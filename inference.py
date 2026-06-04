import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "./lora-1b-fine-tuned"

print("Loading base model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading and merging your fine-tuned LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

# Craft a prompt using the exact same structure the model trained on
prompt = "### Instruction:\nWrite a haiku about debugging code.\n\n### Response:\n"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print("\nGenerating response...")
with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=100, 
        temperature=0.7, 
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )

# Decode and view the output
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n" + response)