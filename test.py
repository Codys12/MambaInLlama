import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
from mamba2.hybrid_mamba_config import MambaConfig

# Load the transformer model
model_name = "./llama3_0.25_mamba2"
#transformer_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the hybrid model
#config = transformer_model.config

# if not hasattr(config, 'head_dim'):
#     d_xb = config.num_key_value_heads * \
#         (config.hidden_size // config.num_attention_heads)
#     d_inner = config.hidden_size
#     d_state = config.hidden_size//config.num_attention_heads
# else:
#     # to handle gemma2
#     d_xb = config.num_key_value_heads * config.head_dim
#     d_inner = config.num_attention_heads * config.head_dim
#     d_state = config.head_dim

# ssm_layers = [0, 4, 8, 12, 16, 20, 24, 28]#[i for i in range(32)]
# attn_layers = [i for i in range(config.num_hidden_layers) if i not in ssm_layers]

# mamba_config = MambaConfig(
#     config.hidden_size,
#     {"expand": 1, "ngroups":config.num_attention_heads, "d_state": d_state},
#     config.rms_norm_eps,
#     d_inner=d_inner,
#     d_xb=d_xb,
#     intermediate_size=config.intermediate_size,
#     hidden_act=config.hidden_act,
#     n_layer=config.num_hidden_layers,
#     attn_layers=attn_layers,
# )
#hybrid_model = MambaTransformerHybridModelWrapper.init_distillation(None, model_name, mamba_config, attn_layers, torch.bfloat16)

hybrid_model = MambaTransformerHybridModelWrapper.from_pretrained(model_name)

#transformer_model.to("cuda")
hybrid_model.to("cuda")

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(hybrid_model.device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = hybrid_model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))