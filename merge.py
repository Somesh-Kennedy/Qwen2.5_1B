import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = "./lora-1b-fine-tuned"
OUTPUT_DIR = "./qwen2.5-1.5b-alpaca-fused"

def merge_and_export():
    print("--> 1. Loading tokenizer from adapter directory...")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    print("--> 2. Loading base model in Float16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("--> 3. Attaching the fine-tuned LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)

    print("--> 4. Merging layers permanently (Fusing LoRA weights back into the base matrix)...")
    # This combines the delta adapter weights straight into the base model weights
    fused_model = model.merge_and_unload()

    print(f"--> 5. Saving unified standalone model to {OUTPUT_DIR}...")
    fused_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n========================================================")
    print(f"SUCCESS! Standalone model saved to: {OUTPUT_DIR}")
    print(f"You can now use this folder directly with Ollama, LM Studio, or vLLM!")
    print(f"========================================================\n")

if __name__ == "__main__":
    merge_and_export()