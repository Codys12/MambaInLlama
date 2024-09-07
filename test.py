import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
from mamba2.hybrid_mamba_config import MambaConfig

# Load the transformer model
model_name = "microsoft/Phi-3-mini-128k-instruct"
transformer_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the hybrid model
config = transformer_model.config
mamba_config = MambaConfig(
    hidden_size=config.hidden_size,
    expansion_config={"expand": 1, "ngroups": config.num_attention_heads, "d_state": config.hidden_size // config.num_attention_heads},
    norm_epsilon=config.layer_norm_epsilon,
    d_inner=config.intermediate_size,
    d_xb=config.num_key_value_heads * (config.hidden_size // config.num_attention_heads),
    intermediate_size=config.intermediate_size,
    hidden_act=config.hidden_act,
    n_layer=config.num_hidden_layers,
    attn_layers=list(range(config.num_hidden_layers))  # All layers as attention for this example
)
hybrid_model = MambaTransformerHybridModelWrapper.init_distillation(None, model_name, mamba_config)

# Prepare dummy input
dummy_input = "Hello, world!"
input_ids = tokenizer(dummy_input, return_tensors="pt").input_ids

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
print(f"KV cache shape: {[(k.shape, v.shape) for k, v in kv_cache[0]]}")