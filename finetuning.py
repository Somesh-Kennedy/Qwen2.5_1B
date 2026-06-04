import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"  
DATASET_ID = "tatsu-lab/alpaca"            
OUTPUT_DIR = "./lora-1b-fine-tuned"

def format_prompts(batch):
    formatted_texts = []
    for i in range(len(batch['instruction'])):
        text = f"### Instruction:\n{batch['instruction'][i]}\n\n### Response:\n{batch['output'][i]}"
        formatted_texts.append(text)
    return {"text": formatted_texts}

if __name__ == "__main__":
    print("--> 1. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print("--> 2. Loading model in pure Float16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,  # Standard half-precision math
        device_map="auto"           # Pushes automatically to your GPU
    )

    print("--> 3. Configuring LoRA...")
    peft_config = LoraConfig(
        r=16,                       
        lora_alpha=32,              
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    print("--> 4. Loading dataset...")
    dataset = load_dataset(DATASET_ID, split="train")
    dataset = dataset.select(range(2000))  

    print("--> 5. Formatting dataset text...")
    dataset = dataset.map(format_prompts, batched=True)

    print("--> 6. Setting up SFTConfig...")
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,       
        gradient_accumulation_steps=16,      
        learning_rate=2e-4,
        logging_steps=1,                     
        max_steps=50,                  
        optim="adamw_torch",                 
        fp16=True,                      
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        
        # CHANGED: Swapped back to max_seq_length for version 0.11.4 compatibility
        max_seq_length=256,                      
        
        dataset_text_field="text",
        dataloader_num_workers=0             
    )

    print("--> 7. Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        
        # CHANGED: Swapped back to tokenizer for trl 0.11.4 compatibility
        tokenizer=tokenizer, 
        
        args=training_args,
    )

    print("--> 8. Starting training loop...")
    trainer.train()

    print("--> 9. Saving fine-tuned adapter weights...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"--> Training complete! Adapter saved to {OUTPUT_DIR}")