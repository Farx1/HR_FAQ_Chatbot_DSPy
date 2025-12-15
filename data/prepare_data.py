"""
Data preparation script for HR FAQ fine-tuning
Loads and cleans the syncora/hr-policies-qa-dataset
Converts to Alpaca format for SFT training
"""

import os
import json
import random
import pandas as pd
from datasets import load_dataset, Dataset
from typing import Dict, List, Any
import re

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing"""
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove empty lines
    text = re.sub(r'\n\s*\n', '\n', text)
    
    return text.strip()

def convert_to_alpaca_format(example: Dict[str, Any]) -> Dict[str, str]:
    """Convert dataset example to Alpaca format"""
    
    # Extract question and answer
    question = clean_text(example.get('question', ''))
    answer = clean_text(example.get('answer', ''))
    
    # Skip if either question or answer is empty
    if not question or not answer:
        return None
    
    # Create instruction (question)
    instruction = question
    
    # Create input (empty for FAQ format)
    input_text = ""
    
    # Create output (answer)
    output = answer
    
    return {
        "instruction": instruction,
        "input": input_text,
        "output": output
    }

def load_and_prepare_dataset():
    """Load and prepare the HR FAQ dataset"""
    
    print("Loading HR FAQ dataset...")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("syncora/hr-policies-qa-dataset")
        
        print(f"Dataset loaded successfully!")
        print(f"Train split: {len(dataset['train'])} examples")
        
        # Convert to Alpaca format
        alpaca_data = []
        
        for example in dataset['train']:
            converted = convert_to_alpaca_format(example)
            if converted:
                alpaca_data.append(converted)
        
        print(f"Converted {len(alpaca_data)} examples to Alpaca format")
        
        # Filter out very short or very long examples
        filtered_data = []
        for example in alpaca_data:
            instruction_len = len(example['instruction'])
            output_len = len(example['output'])
            
            # Keep examples with reasonable lengths
            if 10 <= instruction_len <= 500 and 10 <= output_len <= 1000:
                filtered_data.append(example)
        
        print(f"After filtering: {len(filtered_data)} examples")
        
        # Shuffle the data
        random.shuffle(filtered_data)
        
        # Split into train/validation (90/10)
        split_idx = int(0.9 * len(filtered_data))
        train_data = filtered_data[:split_idx]
        val_data = filtered_data[split_idx:]
        
        print(f"Train: {len(train_data)} examples")
        print(f"Validation: {len(val_data)} examples")
        
        # Save the processed datasets
        os.makedirs("data", exist_ok=True)
        
        # Save as JSON
        with open("data/train_alpaca.json", "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        
        with open("data/val_alpaca.json", "w", encoding="utf-8") as f:
            json.dump(val_data, f, ensure_ascii=False, indent=2)
        
        # Also save as HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        train_dataset.save_to_disk("data/train_dataset")
        val_dataset.save_to_disk("data/val_dataset")
        
        print("Datasets saved successfully!")
        
        # Print some examples
        print("\nSample examples:")
        for i, example in enumerate(train_data[:3]):
            print(f"\nExample {i+1}:")
            print(f"Question: {example['instruction']}")
            print(f"Answer: {example['output']}")
        
        return train_dataset, val_dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Creating a small sample dataset for testing...")
        
        # Create a comprehensive sample dataset for testing
        sample_data = [
            {
                "instruction": "How many vacation days do I get per year?",
                "input": "",
                "output": "Full-time employees receive 25 vacation days per year. Part-time employees receive vacation days proportional to their work schedule."
            },
            {
                "instruction": "What is the remote work policy?",
                "input": "",
                "output": "Employees can work remotely up to 3 days per week with manager approval. All remote work must be pre-approved and documented."
            },
            {
                "instruction": "How do I report workplace harassment?",
                "input": "",
                "output": "Report workplace harassment immediately to your manager, HR department, or through the anonymous reporting hotline. All reports are taken seriously and investigated promptly."
            },
            {
                "instruction": "What is the dress code policy?",
                "input": "",
                "output": "Business casual attire is required in the office. Remote work allows for casual dress unless client meetings are scheduled."
            },
            {
                "instruction": "How do I request time off?",
                "input": "",
                "output": "Submit time-off requests through the HR portal at least 2 weeks in advance. Manager approval is required for all requests."
            },
            {
                "instruction": "What is the sick leave policy?",
                "input": "",
                "output": "Employees can take up to 5 sick days per year with manager approval. Extended sick leave requires medical documentation."
            },
            {
                "instruction": "How do I request a salary review?",
                "input": "",
                "output": "Submit a salary review request through the HR portal with supporting documentation. Reviews are conducted annually or upon promotion."
            },
            {
                "instruction": "What training opportunities are available?",
                "input": "",
                "output": "The company offers various training programs including technical skills, leadership development, and professional certifications. Check the learning portal for available courses."
            },
            {
                "instruction": "What is the maternity leave policy?",
                "input": "",
                "output": "Eligible employees receive 12 weeks of paid maternity leave. Additional unpaid leave may be available under FMLA guidelines."
            },
            {
                "instruction": "How do I report a safety concern?",
                "input": "",
                "output": "Report safety concerns immediately to your supervisor, safety officer, or through the anonymous safety hotline. All reports are investigated promptly."
            },
            {
                "instruction": "What is the probation period for new employees?",
                "input": "",
                "output": "New employees have a 90-day probation period. Performance reviews are conducted at 30, 60, and 90 days to assess fit and progress."
            },
            {
                "instruction": "How do I update my personal information?",
                "input": "",
                "output": "Update personal information through the employee self-service portal or by contacting HR directly. Changes to tax information require additional documentation."
            },
            {
                "instruction": "What is the company's policy on overtime?",
                "input": "",
                "output": "Overtime must be pre-approved by your manager. Non-exempt employees receive time-and-a-half pay for hours worked over 40 per week."
            },
            {
                "instruction": "How do I apply for internal job postings?",
                "input": "",
                "output": "Internal job postings are available on the company intranet. Submit applications through the internal portal and notify your current manager."
            },
            {
                "instruction": "What is the policy on workplace confidentiality?",
                "input": "",
                "output": "All employees must maintain strict confidentiality regarding company information, client data, and colleague personal information. Violations may result in disciplinary action."
            },
            {
                "instruction": "How do I request flexible working hours?",
                "input": "",
                "output": "Submit a flexible work arrangement request through HR with manager approval. Requests are evaluated based on business needs and job requirements."
            },
            {
                "instruction": "What is the company's policy on social media?",
                "input": "",
                "output": "Employees should use social media responsibly and avoid posting confidential company information. Personal opinions should not be attributed to the company."
            },
            {
                "instruction": "How do I access my pay stubs?",
                "input": "",
                "output": "Pay stubs are available through the employee self-service portal. You can also request paper copies from HR if needed."
            },
            {
                "instruction": "What is the policy on workplace diversity?",
                "input": "",
                "output": "The company is committed to creating an inclusive workplace that values diversity. Discrimination based on protected characteristics is strictly prohibited."
            },
            {
                "instruction": "How do I report a workplace injury?",
                "input": "",
                "output": "Report workplace injuries immediately to your supervisor and HR. Complete an incident report form and seek medical attention if necessary."
            }
        ]
        
        # Save sample data
        with open("data/train_alpaca.json", "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        train_dataset = Dataset.from_list(sample_data)
        train_dataset.save_to_disk("data/train_dataset")
        
        print("Sample dataset created successfully!")
        return train_dataset, None

def create_out_of_domain_test_set():
    """Create test set with non-HR questions to test OOD rejection"""
    
    ood_questions = [
        "How do I install Python on my computer?",
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
        "What is machine learning?",
        "How do I fix a broken car engine?",
        "What is the weather like today?",
        "How do I learn Spanish?",
        "What is quantum physics?",
        "How do I invest in stocks?",
        "What is the best programming language?"
    ]
    
    ood_data = []
    for question in ood_questions:
        ood_data.append({
            "instruction": question,
            "input": "",
            "output": "Désolé, cette question semble en dehors du périmètre des politiques RH. Veuillez contacter le service RH."
        })
    
    # Save OOD test set
    with open("data/ood_test.json", "w", encoding="utf-8") as f:
        json.dump(ood_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created OOD test set with {len(ood_data)} examples")

if __name__ == "__main__":
    print("Starting data preparation...")
    
    # Load and prepare the main dataset
    train_dataset, val_dataset = load_and_prepare_dataset()
    
    # Create out-of-domain test set
    create_out_of_domain_test_set()
    
    print("\nData preparation completed!")
    print("Files created:")
    print("- data/train_alpaca.json")
    print("- data/val_alpaca.json")
    print("- data/ood_test.json")
    print("- data/train_dataset/")
    if val_dataset:
        print("- data/val_dataset/")
