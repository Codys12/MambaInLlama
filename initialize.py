import os
import torch
from transformers import AutoConfig, AutoTokenizer
from mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
from mamba2.hybrid_mamba_config import MambaConfig

def main():
    model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'
    output_dir = './baseline'
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    config = AutoConfig.from_pretrained(model_name)
    
    if not hasattr(config, 'head_dim'):
        d_xb = config.num_key_value_heads * (config.hidden_size // config.num_attention_heads)
        d_inner = config.hidden_size
        d_state = config.hidden_size // config.num_attention_heads
    else:
        d_xb = config.num_key_value_heads * config.head_dim
        d_inner = config.num_attention_heads * config.head_dim
        d_state = config.head_dim

    ssm_layers = []  # All layers are attention layers
    attn_layers = [i for i in range(config.num_hidden_layers) if i not in ssm_layers]

    mamba_config = MambaConfig(
        d_model=config.hidden_size,
        ssm_cfg={"expand": 1, "ngroups": config.num_attention_heads, "d_state": d_state},
        rms_norm_eps=config.rms_norm_eps,
        d_inner=d_inner,
        d_xb=d_xb,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        n_layer=config.num_hidden_layers,
        attn_layers=attn_layers,
    )

    # Initialize the hybrid model
    student_model = MambaTransformerHybridModelWrapper.init_distillation(
        None,
        model_name,
        mamba_config,
        attn_layers=attn_layers,
        init_with_kqvo=False,
        attn_implementation='flash_attention_2'
    )

    # Save the model and tokenizer
    student_model.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    student_model.save_config(output_dir)

if __name__ == "__main__":
    main()
