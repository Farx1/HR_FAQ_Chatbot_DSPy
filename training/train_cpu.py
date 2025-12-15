"""
Simplified training script for CPU testing
Uses a smaller model for demonstration purposes
"""

import os
import json
import torch
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, load_from_disk
import random
from typing import Dict, List, Any
import warnings
warnings.filterwarnings("ignore")

# Set seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# Use a smaller model for CPU testing
MODEL_NAME = "microsoft/DialoGPT-small"  # Much smaller model for CPU
MAX_LENGTH = 128
LEARNING_RATE = 5e-4
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4

def format_prompt(example: Dict[str, str]) -> str:
    """Format example for training"""
    
    instruction = example["instruction"]
    output = example["output"]
    
    # Simple format for smaller model
    formatted_text = f"HR Question: {instruction}\nHR Answer: {output}<|endoftext|>"
    
    return formatted_text

def prepare_dataset(dataset_path: str) -> Dataset:
    """Load and prepare dataset for training"""
    
    print(f"Loading dataset from {dataset_path}")
    
    try:
        dataset = load_from_disk(dataset_path)
    except:
        # Fallback to JSON loading
        with open(dataset_path.replace("_dataset", "_alpaca.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        dataset = Dataset.from_list(data)
    
    print(f"Loaded {len(dataset)} examples")
    
    # Format the dataset
    def format_examples(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            example = {
                "instruction": examples["instruction"][i],
                "input": examples["input"][i],
                "output": examples["output"][i]
            }
            formatted = format_prompt(example)
            texts.append(formatted)
        return {"text": texts}
    
    formatted_dataset = dataset.map(format_examples, batched=True)
    
    return formatted_dataset

def setup_model_and_tokenizer():
    """Setup model and tokenizer"""
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    
    # Load model for CPU
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,  # Smaller rank for CPU
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj"]  # DialoGPT specific modules
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer

def tokenize_function(examples, tokenizer):
    """Tokenize the dataset"""
    
    # Tokenize
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # For causal LM, labels are the same as input_ids
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def train_model():
    """Main training function"""
    
    print("Starting HR FAQ fine-tuning (CPU version)...")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load datasets
    train_dataset = prepare_dataset("data/train_dataset")
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/hr_faq_dialogpt_lora",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=50,
        save_strategy="steps",
        report_to="none",  # Completely disable wandb
        seed=RANDOM_SEED,
        fp16=False,  # Disabled for CPU
        gradient_checkpointing=False,  # Disabled for CPU
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained("models/hr_faq_dialogpt_lora")
    
    # Save LoRA adapters separately
    model.save_pretrained("models/hr_faq_dialogpt_lora_adapters")
    
    print("Training completed!")
    print("Model saved to: models/hr_faq_dialogpt_lora")
    print("LoRA adapters saved to: models/hr_faq_dialogpt_lora_adapters")

def load_trained_model():
    """Load the trained model for inference"""
    
    print("Loading trained model...")
    
    # Try to load tokenizer from saved model, fallback to base model if corrupted
    try:
        tokenizer = AutoTokenizer.from_pretrained("models/hr_faq_dialogpt_lora", use_fast=False)
    except Exception as e:
        print(f"Warning: Could not load tokenizer from saved model, using base model tokenizer: {e}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, "models/hr_faq_dialogpt_lora_adapters")
    
    return model, tokenizer

def generate_response(model, tokenizer, question: str) -> str:
    """Generate response for a given question"""
    
    # Format prompt
    prompt = f"HR Question: {question}\nHR Answer:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part
    response = response.split("HR Answer:")[-1].strip()
    
    return response

if __name__ == "__main__":
    print("HR FAQ Fine-tuning with DialoGPT-small (CPU)")
    print("=" * 50)
    
    # Check if data exists
    if not os.path.exists("data/train_dataset"):
        print("Training data not found. Please run data/prepare_data.py first.")
        exit(1)
    
    # Train the model
    train_model()
    
    print("\nTesting the trained model...")
    
    # Load trained model and test
    model, tokenizer = load_trained_model()
    
    # Test questions
    test_questions = [
        "How many vacation days do I get per year?",
        "What is the remote work policy?",
        "How do I install Python on my computer?",  # OOD question
    ]
    
    for question in test_questions:
        print(f"\nQuestion: {question}")
        response = generate_response(model, tokenizer, question)
        print(f"Response: {response}")
