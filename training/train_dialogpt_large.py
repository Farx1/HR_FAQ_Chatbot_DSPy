"""
Training script for DialoGPT-large with LoRA fine-tuning
"""

import os
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import json
import warnings
warnings.filterwarnings("ignore")

# Set seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Model configuration
MODEL_NAME = "microsoft/DialoGPT-large"
MAX_LENGTH = 256
LEARNING_RATE = 2e-4
NUM_EPOCHS = 2
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
OUTPUT_DIR = "models/hr_faq_dialogpt_large_lora"

# LoRA configuration
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,  # Rank
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]  # DialoGPT attention modules
)

def load_hr_dataset():
    """Load and prepare HR dataset"""
    print("Loading HR dataset...")
    
    # Load the prepared dataset
    with open("data/train_alpaca.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples")
    
    # Convert to Alpaca format
    formatted_data = []
    for item in data:
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item["output"]
        
        # Create prompt in DialoGPT format
        if input_text:
            prompt = f"Human: {instruction}\nContext: {input_text}\nAssistant: {output}"
        else:
            prompt = f"Human: {instruction}\nAssistant: {output}"
        
        formatted_data.append({
            "text": prompt,
            "instruction": instruction,
            "input": input_text,
            "output": output
        })
    
    return Dataset.from_list(formatted_data)

def tokenize_function(examples, tokenizer):
    """Tokenize the dataset"""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

def load_model_and_tokenizer():
    """Load model and tokenizer"""
    print(f"Loading {MODEL_NAME}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Use float32 for CPU
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Apply LoRA
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    
    return model, tokenizer

def train_model():
    """Train the model with LoRA"""
    print("Starting DialoGPT-large LoRA training...")
    print("=" * 50)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Load dataset
    dataset = load_hr_dataset()
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split dataset
    train_size = int(0.8 * len(tokenized_dataset))
    train_dataset = tokenized_dataset.select(range(train_size))
    eval_dataset = tokenized_dataset.select(range(train_size, len(tokenized_dataset)))
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Evaluation examples: {len(eval_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,  # Disable for CPU
        gradient_checkpointing=False,  # Disable for CPU
        dataloader_pin_memory=False,  # Disable for CPU
        report_to="none",  # Disable wandb
        seed=RANDOM_SEED,
        remove_unused_columns=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training completed!")
    return model, tokenizer

def load_trained_model():
    """Load the trained model"""
    print(f"Loading trained model from {OUTPUT_DIR}...")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    # Load LoRA adapters
    model = get_peft_model(model, LORA_CONFIG)
    model.load_adapter(OUTPUT_DIR, "default")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    
    return model, tokenizer

def test_model(model, tokenizer):
    """Test the trained model"""
    print("\nTesting trained model...")
    print("=" * 30)
    
    test_questions = [
        "What is the company's vacation policy?",
        "How do I report workplace harassment?",
        "What are the remote work guidelines?",
        "How do I request time off?",
        "What training opportunities are available?"
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        
        # Create prompt
        prompt = f"Human: {question}\nAssistant:"
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        
        print(f"Response: {response}")

if __name__ == "__main__":
    print("DialoGPT-large LoRA Training for HR FAQ")
    print("=" * 50)
    
    # Check if model already exists
    if os.path.exists(OUTPUT_DIR):
        print("Trained model found. Loading...")
        model, tokenizer = load_trained_model()
    else:
        print("No trained model found. Starting training...")
        model, tokenizer = train_model()
    
    # Test the model
    test_model(model, tokenizer)
    
    print("\nTraining script completed!")
