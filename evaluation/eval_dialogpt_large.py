"""
Evaluation script for DialoGPT-large fine-tuned model
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import json
import warnings
warnings.filterwarnings("ignore")

# Set seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Model configuration
MODEL_NAME = "microsoft/DialoGPT-large"
OUTPUT_DIR = "models/hr_faq_dialogpt_large_lora"

# LoRA configuration
LORA_CONFIG = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn", "c_proj"]
)

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

def load_test_data():
    """Load test data"""
    print("Loading test data...")
    
    # Load validation data
    with open("data/val_alpaca.json", "r", encoding="utf-8") as f:
        val_data = json.load(f)
    
    # Load OOD test data
    with open("data/ood_test.json", "r", encoding="utf-8") as f:
        ood_data = json.load(f)
    
    print(f"Validation examples: {len(val_data)}")
    print(f"OOD examples: {len(ood_data)}")
    
    return val_data, ood_data

def generate_response(model, tokenizer, question, max_length=100):
    """Generate response for a question"""
    # Create prompt
    prompt = f"Human: {question}\nAssistant:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    
    return response

def evaluate_hr_questions(model, tokenizer, test_data):
    """Evaluate on HR questions"""
    print("\nEvaluating HR Questions...")
    print("=" * 40)
    
    results = []
    
    for i, item in enumerate(test_data):
        question = item["instruction"]
        expected = item["output"]
        
        print(f"\nQuestion {i+1}: {question}")
        print(f"Expected: {expected}")
        
        # Generate response
        response = generate_response(model, tokenizer, question)
        print(f"Generated: {response}")
        
        # Simple evaluation metrics
        response_lower = response.lower()
        expected_lower = expected.lower()
        
        # Check for relevant keywords
        hr_keywords = ["policy", "vacation", "leave", "harassment", "training", "remote", "work", "company", "employee"]
        has_hr_keywords = any(keyword in response_lower for keyword in hr_keywords)
        
        # Check if response is reasonable length
        reasonable_length = 10 <= len(response.split()) <= 50
        
        # Check if response is not empty
        not_empty = len(response.strip()) > 0
        
        result = {
            "question": question,
            "expected": expected,
            "generated": response,
            "has_hr_keywords": has_hr_keywords,
            "reasonable_length": reasonable_length,
            "not_empty": not_empty,
            "overall_score": has_hr_keywords and reasonable_length and not_empty
        }
        
        results.append(result)
        
        print(f"Score: {'PASS' if result['overall_score'] else 'FAIL'}")
    
    return results

def evaluate_ood_questions(model, tokenizer, ood_data):
    """Evaluate on out-of-domain questions"""
    print("\nEvaluating Out-of-Domain Questions...")
    print("=" * 40)
    
    results = []
    
    for i, item in enumerate(ood_data):
        question = item["instruction"]
        expected_refusal = item["output"]
        
        print(f"\nOOD Question {i+1}: {question}")
        print(f"Expected refusal: {expected_refusal}")
        
        # Generate response
        response = generate_response(model, tokenizer, question)
        print(f"Generated: {response}")
        
        # Check if model refuses appropriately
        response_lower = response.lower()
        
        # Look for refusal indicators
        refusal_indicators = [
            "not hr", "not human resources", "not related", "not applicable",
            "not my area", "not my expertise", "not qualified", "not appropriate",
            "contact hr", "ask hr", "speak to hr", "not sure", "don't know"
        ]
        
        has_refusal = any(indicator in response_lower for indicator in refusal_indicators)
        
        # Check if response is short (good refusal should be concise)
        is_short = len(response.split()) <= 20
        
        result = {
            "question": question,
            "expected_refusal": expected_refusal,
            "generated": response,
            "has_refusal": has_refusal,
            "is_short": is_short,
            "overall_score": has_refusal and is_short
        }
        
        results.append(result)
        
        print(f"Score: {'PASS' if result['overall_score'] else 'FAIL'}")
    
    return results

def calculate_metrics(hr_results, ood_results):
    """Calculate overall metrics"""
    print("\nOverall Metrics...")
    print("=" * 30)
    
    # HR metrics
    hr_passed = sum(1 for r in hr_results if r["overall_score"])
    hr_total = len(hr_results)
    hr_accuracy = hr_passed / hr_total if hr_total > 0 else 0
    
    # OOD metrics
    ood_passed = sum(1 for r in ood_results if r["overall_score"])
    ood_total = len(ood_results)
    ood_accuracy = ood_passed / ood_total if ood_total > 0 else 0
    
    # Overall metrics
    total_passed = hr_passed + ood_passed
    total_questions = hr_total + ood_total
    overall_accuracy = total_passed / total_questions if total_questions > 0 else 0
    
    print(f"HR Questions: {hr_passed}/{hr_total} ({hr_accuracy:.1%})")
    print(f"OOD Questions: {ood_passed}/{ood_total} ({ood_accuracy:.1%})")
    print(f"Overall: {total_passed}/{total_questions} ({overall_accuracy:.1%})")
    
    return {
        "hr_accuracy": hr_accuracy,
        "ood_accuracy": ood_accuracy,
        "overall_accuracy": overall_accuracy,
        "hr_passed": hr_passed,
        "hr_total": hr_total,
        "ood_passed": ood_passed,
        "ood_total": ood_total
    }

def save_results(hr_results, ood_results, metrics):
    """Save evaluation results"""
    results = {
        "metrics": metrics,
        "hr_results": hr_results,
        "ood_results": ood_results
    }
    
    with open("evaluation_results_dialogpt_large.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to evaluation_results_dialogpt_large.json")

def main():
    print("DialoGPT-large Evaluation")
    print("=" * 30)
    
    # Check if model exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Trained model not found at {OUTPUT_DIR}")
        print("Please run training first.")
        return
    
    # Load model
    model, tokenizer = load_trained_model()
    
    # Load test data
    val_data, ood_data = load_test_data()
    
    # Evaluate HR questions
    hr_results = evaluate_hr_questions(model, tokenizer, val_data)
    
    # Evaluate OOD questions
    ood_results = evaluate_ood_questions(model, tokenizer, ood_data)
    
    # Calculate metrics
    metrics = calculate_metrics(hr_results, ood_results)
    
    # Save results
    save_results(hr_results, ood_results, metrics)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
