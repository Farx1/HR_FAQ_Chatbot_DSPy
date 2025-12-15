"""
Training script for HR FAQ fine-tuning with Mistral-7B-Instruct-v0.3
Uses LoRA/QLoRA for parameter-efficient fine-tuning
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
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
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

# Model configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_LENGTH = 256  # Reduced for CPU
LEARNING_RATE = 5e-4  # Slightly higher for CPU
NUM_EPOCHS = 1  # Reduced for CPU
BATCH_SIZE = 1  # Reduced for CPU
GRADIENT_ACCUMULATION_STEPS = 8  # Increased to maintain effective batch size

def format_prompt(example: Dict[str, str]) -> str:
    """Format example in Mistral instruction format"""
    
    system_prompt = "Tu es un assistant RH professionnel. Réponds de façon claire, concise et exacte sur la base des politiques RH disponibles. Si la question sort du périmètre RH ou si l'information manque, indique-le poliment et propose de contacter le service RH."
    
    instruction = example["instruction"]
    output = example["output"]
    
    # Format in Mistral instruction style
    formatted_text = f"<s>[INST] {system_prompt}\n\n{instruction} [/INST] {output}</s>"
    
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
    """Setup model and tokenizer with LoRA configuration"""
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model...")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model without quantization for CPU
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        # Configure quantization for GPU
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
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
    
    print("Starting HR FAQ fine-tuning...")
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Load datasets
    train_dataset = prepare_dataset("data/train_dataset")
    val_dataset = prepare_dataset("data/val_dataset") if os.path.exists("data/val_dataset") else None
    
    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    if val_dataset:
        val_dataset = val_dataset.map(
            lambda x: tokenize_function(x, tokenizer),
            batched=True,
            remove_columns=val_dataset.column_names
        )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="models/hr_faq_mistral_lora",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        evaluation_strategy="steps" if val_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,  # Disable wandb for simplicity
        seed=RANDOM_SEED,
        fp16=device == "cuda",
        gradient_checkpointing=device == "cuda",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print("Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained("models/hr_faq_mistral_lora")
    
    # Save LoRA adapters separately
    model.save_pretrained("models/hr_faq_mistral_lora_adapters")
    
    print("Training completed!")
    print("Model saved to: models/hr_faq_mistral_lora")
    print("LoRA adapters saved to: models/hr_faq_mistral_lora_adapters")

def load_trained_model():
    """Load the trained model for inference"""
    
    print("Loading trained model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/hr_faq_mistral_lora")
    
    # Load base model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cpu":
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, "models/hr_faq_mistral_lora_adapters")
    
    return model, tokenizer

def generate_response(model, tokenizer, question: str) -> str:
    """Generate response for a given question"""
    
    system_prompt = "Tu es un assistant RH professionnel. Réponds de façon claire, concise et exacte sur la base des politiques RH disponibles. Si la question sort du périmètre RH ou si l'information manque, indique-le poliment et propose de contacter le service RH."
    
    # Format prompt
    prompt = f"<s>[INST] {system_prompt}\n\n{question} [/INST]"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.05,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the generated part
    response = response.split("[/INST]")[-1].strip()
    
    return response

if __name__ == "__main__":
    print("HR FAQ Fine-tuning with Mistral-7B-Instruct-v0.3")
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
