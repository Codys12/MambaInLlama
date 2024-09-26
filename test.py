import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
from mamba2.hybrid_mamba_config import MambaConfig

# Load the transformer model
model_name = "./llama3_0.50_mamba2_progressive"
tokenizer = AutoTokenizer.from_pretrained(model_name)

hybrid_model = MambaTransformerHybridModelWrapper.from_pretrained(model_name)

hybrid_model.to("cuda")

# Pre-defined system message
system_message = {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"}

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

while True:
    # Get user input
    user_input = input("User: ")
    
    # Exit the loop if the user types 'exit'
    if user_input.lower() == "exit":
        break
    
    # Create the message structure
    messages = [
        system_message,
        {"role": "user", "content": user_input},
    ]
    
    # Prepare input for the model
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    # Generate the response
    outputs = hybrid_model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    
    # Decode and print the response
    response = outputs[0][input_ids.shape[-1]:]
    print("Pirate Bot:", tokenizer.decode(response, skip_special_tokens=True))
