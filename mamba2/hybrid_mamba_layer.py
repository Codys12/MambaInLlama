# Copyright (c) 2024, Tri Dao, Albert Gu.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    print("Error importing causalconv1d")
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
except ImportError:
    print("Error importing causalconv1d")
    causal_conv1d_varlen_states = None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

from huggingface_hub import PyTorchModelHubMixin

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class Mamba2(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        d_model,
        d_xb,
        d_inner=None,
        d_state=128,
        d_conv=4,
        conv_init=None,
        expand=2,
        headdim=64,
        d_ssm=None,  # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        ngroups=1,
        A_init_range=(1, 16),
        D_has_hdim=False,
        rmsnorm=True,
        norm_before_gate=False,
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        repeat_kv_before_conv=False,
        bias=False,
        conv_bias=True,
        # Fused kernel and sharding options
        chunk_size=256,
        use_mem_eff_path=True,
        layer_idx=None,  # Absorb kwarg for general module
        process_group=None,
        sequence_parallel=True,
        rotary_emb=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.conv_init = conv_init
        self.expand = expand
        self.process_group = process_group
        self.sequence_parallel = sequence_parallel
        self.world_size = 1 if process_group is None else process_group.size()
        self.local_rank = 0 if process_group is None else process_group.rank()
        self.d_inner = d_inner if d_inner is not None else (self.expand * self.d_model) // self.world_size
        # assert self.d_inner * self.world_size == self.expand * self.d_model
        self.headdim = headdim
        self.d_ssm = self.d_inner if d_ssm is None else d_ssm // self.world_size
        self.scale_factor = ngroups
        print(f"NGROUPS: {ngroups}")
        assert ngroups % self.world_size == 0
        self.ngroups = ngroups // self.world_size
        assert self.d_ssm % self.headdim == 0
        self.nheads = self.d_ssm // self.headdim
        self.D_has_hdim = D_has_hdim
        self.rmsnorm = rmsnorm
        self.norm_before_gate = norm_before_gate
        self.dt_limit = dt_limit
        self.activation = "silu"
        self.chunk_size = chunk_size
        self.use_mem_eff_path = use_mem_eff_path
        self.layer_idx = layer_idx
        self.d_xb = d_xb
        self.repeat_group = self.d_inner // self.d_xb
        self.repeat_kv_before_conv = repeat_kv_before_conv
        self.rotary_emb = rotary_emb
        assert self.d_inner == self.ngroups * self.d_state
        assert self.d_inner == self.d_ssm
        
        self.nheads = self.ngroups
        self.headdim = self.d_state
        self.count = 0

        # Order: [z, x, B, C, dt]
        # [hidden_dim, hidden_dim, d_state]
        d_in_proj = self.d_inner + self.d_xb + self.d_xb + self.d_inner + self.nheads
        # d_in_proj = 2 * self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        if self.process_group is None:
            self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)
        else:
            self.in_proj = ColumnParallelLinear(self.d_model, d_in_proj * self.world_size, bias=bias,
                                                process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                                **factory_kwargs)

        # conv_dim = self.d_ssm + 2 * self.ngroups * self.d_state

        if self.repeat_kv_before_conv:
            conv_dim = self.d_inner + self.d_inner + self.d_inner
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                **factory_kwargs,
            )
        else:
            conv_dim = self.d_inner + self.d_xb + self.d_xb
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=conv_dim,
                padding=d_conv - 1,
                **factory_kwargs,
            )
        
        if self.conv_init is not None:
            nn.init.uniform_(self.conv1d.weight, -self.conv_init, self.conv_init)

        self.act = nn.SiLU()

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_ssm if self.D_has_hdim else self.nheads, device=device))
        self.D._no_weight_decay = True

        if self.rmsnorm:
            assert RMSNormGated is not None
            self.norm = RMSNormGated(self.d_ssm, eps=1e-5, norm_before_gate=self.norm_before_gate,
                                     group_size=self.d_ssm // ngroups, **factory_kwargs)

        if self.process_group is None:
            self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        else:
            self.out_proj = RowParallelLinear(self.d_inner * self.world_size, self.d_model, bias=bias,
                                              process_group=self.process_group, sequence_parallel=self.sequence_parallel,
                                              **factory_kwargs)

    def forward(self, u, seqlen=None, seq_idx=None, cu_seqlens=None, inference_params=None):
        """
        u: (batch, seqlen, hidden_dim) if seqlen=None.
            If seqlen is not None, u is (batch * seqlen, hidden_dim). This is so that when we
            split u during sequence parallel, we split the batch * seqlen dimension
            (in case batch is small).
        Returns: same shape as u
        """
        seqlen_og = seqlen
        if seqlen is None:
            batch, seqlen, dim = u.shape
        else:
            batch_seqlen, dim = u.shape
            batch = batch_seqlen // seqlen

        conv_state, ssm_state = None, None
        if inference_params is not None:
            inference_batch = cu_seqlens.shape[0] - 1 if cu_seqlens is not None else batch
            conv_state, ssm_state = self._get_states_from_cache(inference_params, inference_batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(u, conv_state, ssm_state)
                return out

        zxbcdt = self.in_proj(u)  # (B, L, d_in_proj) or (B * L, d_in_proj)
        if seqlen_og is not None:
            zxbcdt = rearrange(zxbcdt, "(b l) d -> b l d", l=seqlen)
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        A = -torch.exp(self.A_log.float())  # (nheads) or (d_inner, d_state)
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)
        
        # [z, x, B, C, dt]
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_inner - 2 * self.d_xb - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.d_xb, self.nheads],
            dim=-1
        )

        x, B, C = torch.split(xBC, [self.d_xb, self.d_xb, self.ngroups * self.d_state], dim=-1)
        q = C[:, :, :self.d_inner]
        k = B
        v = x

        q = q.view(batch, seqlen, -1, self.headdim).transpose(1, 2)
        k = k.view(batch, seqlen, -1, self.headdim).transpose(1, 2)
        v = v.view(batch, seqlen, -1, self.headdim).transpose(1, 2)

        _, num_key_value_heads, _, _ = v.shape

        position_ids = torch.arange(seqlen, dtype=torch.long, device=q.device)
        position_ids = position_ids.unsqueeze(0).expand(batch, -1)

        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        num_key_value_groups = self.nheads // num_key_value_heads
        k = repeat_kv(k, num_key_value_groups)
        v = repeat_kv(v, num_key_value_groups)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            #attn_mask=causal_mask,
            #dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seqlen, -1)

        attn_output = self.out_proj(attn_output)
        return attn_output


        if self.repeat_kv_before_conv:
            x, B, C = torch.split(xBC, [self.d_xb, self.d_xb, self.ngroups * self.d_state], dim=-1)
            # minic the GQA
            x = rearrange(x, "b l (xb_group dstate) -> b xb_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.repeat_group)
            # x shape: (bsz, n_group, l, dim)
            B = rearrange(B, "b l (xb_group dstate) -> b xb_group l dstate", dstate=self.d_state)
            B = repeat_kv(B, self.repeat_group)
            # combine x, B, C
            x = rearrange(x, "b g l p -> b l (g p)")
            B = rearrange(B, "b g l p -> b l (g p)")
            xBC = torch.cat((x, B, C), dim=-1)

        if conv_state is not None:
            if cu_seqlens is None:
                # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                xBC_t = rearrange(xBC, "b l d -> b d l")
                conv_state.copy_(F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0)))  # Update state (B D W)
            else:
                assert causal_conv1d_varlen_states is not None, "varlen inference requires causal_conv1d package"
                assert batch == 1, "varlen inference only supports batch dimension 1"
                conv_varlen_states = causal_conv1d_varlen_states(
                    xBC.squeeze(0), cu_seqlens, state_len=conv_state.shape[-1]
                )
                conv_state.copy_(conv_varlen_states)
        assert self.activation in ["silu", "swish"]

        if causal_conv1d_fn is None or self.activation not in ["silu", "swish"]:
            print("USING SLOW causalconv1d")
            assert seq_idx is None, "varlen conv1d requires the causal_conv1d package"
            xBC = self.act(
                self.conv1d(xBC.transpose(1, 2)).transpose(1, 2)[:, -(self.dconv - 1):]
            )  # (B, L, self.d_ssm + 2 * ngroups * d_state)
        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2),
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=seq_idx,
            ).transpose(1, 2)
        
        if self.repeat_kv_before_conv:
            x, B, C = torch.split(xBC, [self.ngroups * self.d_state, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)

            y = mamba_chunk_scan_combined(
                rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                dt,
                A,
                rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )

        else:
            # self.d_xb + self.d_xb + self.d_inner
            x, B, C = torch.split(xBC, [self.d_xb, self.d_xb, self.ngroups * self.d_state], dim=-1)
            
            # minic the GQA
            x = rearrange(x, "b l (xb_group dstate) -> b xb_group l dstate", dstate=self.d_state)
            x = repeat_kv(x, self.repeat_group)
            # x shape: (bsz, n_group, l, dim)
            
            B = rearrange(B, "b l (xb_group dstate) -> b xb_group l dstate", dstate=self.d_state)
            B = repeat_kv(B, self.repeat_group)

            y = mamba_chunk_scan_combined(
                # rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
                rearrange(x, "b g l p -> b l g p"),
                dt,
                A,
                # rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
                rearrange(B, "b g l n -> b l g n"),
                rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
                chunk_size=self.chunk_size,
                D=rearrange(self.D, "(h p) -> h p", p=self.headdim) if self.D_has_hdim else self.D,
                z=rearrange(z, "b l (h p) -> b l h p", p=self.headdim) if not self.rmsnorm else None,
                dt_bias=self.dt_bias,
                dt_softplus=True,
                seq_idx=seq_idx,
                cu_seqlens=cu_seqlens,
                **dt_limit_kwargs,
                return_final_states=ssm_state is not None,
                return_varlen_states=cu_seqlens is not None and inference_params is not None,
            )

        if ssm_state is not None:
            y, last_state, *rest = y
            if cu_seqlens is None:
                ssm_state.copy_(last_state)
            else:
                varlen_states = rest[0]
                ssm_state.copy_(varlen_states)
        y = rearrange(y, "b l h p -> b l (h p)")
        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        if seqlen_og is not None:
            y = rearrange(y, "b l d -> (b l) d")
        out = self.out_proj(y)

        # attn_norm = torch.norm(attn_output)
        # mamba_norm = torch.norm(out)
        
        # # Find the smaller of the two norms
        # min_norm = torch.min(attn_norm, mamba_norm)
        
        # # Compute the scaling factor (1/2 of the smallest vector's scale)
        # scaling_factor = 0.5 * min_norm
        
        # # Scale both vectors
        # attn_output = attn_output * (scaling_factor / attn_norm)
        # out = out * (scaling_factor / mamba_norm)

        #if(self.layer_idx == 16 and (self.count % 64 == 0 or self.count == 0)):
            #print(f"attn_output magnitude: {torch.norm(attn_output)}")
            # print(f"mamba_output magnitude: {torch.norm(out)}")
            #print(f"combined magnitude: {torch.norm(out + attn_output)}")
            #print(f"distance: {torch.norm(out - attn_output)}")

        # self.count += 1

        #out = out + attn_output
        out = out / (2.0 * self.scale_factor)
        #out = out.div(2.0 * self.scale_factor)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        zxbcdt = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        d_mlp = (zxbcdt.shape[-1] - 2 * self.d_inner - 2 * self.d_xb - self.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, self.d_ssm, self.d_ssm + 2 * self.d_xb, self.nheads],
            dim=-1
        )
        x, B, C = torch.split(xBC, [self.d_xb, self.d_xb, self.ngroups * self.d_state], dim=-1)
        q = C[:, :, :self.d_inner]
        k = x
        v = B
        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(dtype=dtype)
        else:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        A = -torch.exp(self.A_log.float())  # (nheads,)

        x, B, C = torch.split(xBC, [self.d_xb, self.d_xb, self.ngroups * self.d_state], dim=-1)
        
        
        # minic the GQA
        x = rearrange(x, "b (xb_group dstate) -> b xb_group dstate", dstate=self.d_state)
        x_reshaped = torch.repeat_interleave(x, dim=1, repeats=self.repeat_group)

        B = rearrange(B, "b (xb_group dstate) -> b xb_group dstate", dstate=self.d_state)
        B = torch.repeat_interleave(B, dim=1, repeats=self.repeat_group)
        
        # SSM step
        assert selective_state_update is not None
            
        A = repeat(A, "h -> h p n", p=self.headdim, n=self.d_state).to(dtype=torch.float32)
        dt = repeat(dt, "b h -> b h p", p=self.headdim)
        dt_bias = repeat(self.dt_bias, "h -> h p", p=self.headdim)
        D = repeat(self.D, "h -> h p", p=self.headdim)
        # B = rearrange(B, "b (g n) -> b g n", g=self.ngroups)
        C = rearrange(C, "b (g n) -> b g n", g=self.ngroups)
        # x_reshaped = rearrange(x, "b (h p) -> b h p", p=self.headdim)
        if not self.rmsnorm:
            z = rearrange(z, "b (h p) -> b h p", p=self.headdim)
        y = selective_state_update(
            ssm_state, x_reshaped, dt, A, B, C, D, z=z if not self.rmsnorm else None,
            dt_bias=dt_bias, dt_softplus=True
        )
        y = rearrange(y, "b h p -> b (h p)")

        if self.rmsnorm:
            y = self.norm(y, z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        out = self.out_proj(y)
        out = out / (2.0 * self.scale_factor)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_conv, self.conv1d.weight.shape[0], device=device, dtype=conv_dtype
        ).transpose(1, 2)
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.nheads, self.headdim, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_conv,
                self.conv1d.weight.shape[0],
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            ).transpose(1, 2)
            ssm_state = torch.zeros(
                batch_size,
                self.nheads,
                self.headdim,
                self.d_state,
                device=self.in_proj.weight.device,
                dtype=self.in_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state