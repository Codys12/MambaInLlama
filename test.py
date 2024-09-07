import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
from mamba2.hybrid_mamba_config import MambaConfig

# Load the transformer model
model_name = "meta-llama/Meta-Llama-3.1-8B"
transformer_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the hybrid model
config = transformer_model.config

if not hasattr(config, 'head_dim'):
    d_xb = config.num_key_value_heads * \
        (config.hidden_size // config.num_attention_heads)
    d_inner = config.hidden_size
    d_state = config.hidden_size//config.num_attention_heads
else:
    # to handle gemma2
    d_xb = config.num_key_value_heads * config.head_dim
    d_inner = config.num_attention_heads * config.head_dim
    d_state = config.head_dim

ssm_layers = [i for i in range(32)]
attn_layers = [i for i in range(config.num_hidden_layers) if i not in ssm_layers]

mamba_config = MambaConfig(
    config.hidden_size,
    {"expand": 1, "ngroups":config.num_attention_heads, "d_state": d_state},
    config.rms_norm_eps,
    d_inner=d_inner,
    d_xb=d_xb,
    intermediate_size=config.intermediate_size,
    hidden_act=config.hidden_act,
    n_layer=config.num_hidden_layers,
    attn_layers=attn_layers,
)
hybrid_model = MambaTransformerHybridModelWrapper.init_distillation(None, model_name, mamba_config, attn_layers, torch.bfloat16)

transformer_model.to("cuda")
hybrid_model.to("cuda")

# Prepare dummy input
dummy_input = "Hello, world!"
input_ids = tokenizer(dummy_input, return_tensors="pt").input_ids
input_ids = input_ids.to("cuda")

# Generate with transformer model to get KV cache
with torch.no_grad():
    outputs = transformer_model.generate(
        input_ids,
        max_length=20,
        use_cache=True,
        return_dict_in_generate=True,
        output_attentions=True
    )

# Print KV cache shape
kv_cache = outputs.past_key_values
#print(f"KV cache shape: {[(k.shape, v.shape) for k, v in kv_cache[0]]}")
print(kv_cache[0][0].shape)
print(kv_cache[0][0])

print(kv_cache[0][1].shape)
print(kv_cache[0][1])

with torch.no_grad():
    ssm_outputs = hybrid_model.generate(
        input_ids,
        max_length=20,
        #use_cache=True,
        return_dict_in_generate=True,
        output_attentions=True
    )
