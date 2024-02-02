
from transformers import ( 
    AutoTokenizer
)
import torch

def get_tokenzation():
    model_name = "tiiuae/falcon-7b"
    instructions_path = f"../../data/prompts/Deganoetal2024_instructions.txt"
    with open(instructions_path, "r") as f:
        instructions = f.read()

    tokenizer = AutoTokenizer.from_pretrained(model_name, add_bos_token=True)

    input_ids = tokenizer.encode(
        instructions,
        #return_tensors="pt",
        add_special_tokens=False
    )
    print(len(input_ids), input_ids)
    fixed_input_ids = [tokenizer.eos_token_id] + input_ids
    print(tokenizer.bos_token_id)
    print(tokenizer.decode(tokenizer.eos_token_id))
    print("Input ids ", fixed_input_ids)
    print("shape ", len(fixed_input_ids))
    print(tokenizer.decode(fixed_input_ids, skip_special_tokens=False))

    print("Falcon check ", tokenizer.decode([204,   193, 15073], skip_special_tokens=False))
if __name__ == "__main__":
    get_tokenzation()
