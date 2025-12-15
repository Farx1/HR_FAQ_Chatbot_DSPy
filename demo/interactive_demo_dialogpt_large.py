"""
Interactive demo for DialoGPT-large fine-tuned model
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
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
    
    try:
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
        
        print("Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

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

def is_hr_question(question):
    """Check if question is HR-related"""
    hr_keywords = [
        "vacation", "leave", "time off", "holiday", "sick leave",
        "harassment", "discrimination", "workplace", "employee",
        "training", "development", "career", "promotion",
        "remote work", "work from home", "telecommute",
        "contract", "employment", "salary", "benefits",
        "policy", "procedure", "guideline", "rule",
        "hr", "human resources", "personnel"
    ]
    
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in hr_keywords)

def get_system_prompt():
    """Get the system prompt for the chatbot"""
    return """
You are an HR FAQ chatbot assistant. Your role is to provide helpful, accurate, and professional responses to HR-related questions.

Guidelines:
- Answer HR questions clearly and concisely
- Use professional, neutral language
- Keep responses short and to the point
- If you don't know something, suggest contacting HR directly
- For non-HR questions, politely redirect to HR

HR Topics you can help with:
- Vacation and leave policies
- Workplace harassment reporting
- Remote work guidelines
- Training and development opportunities
- Employment contracts and benefits
- Company policies and procedures

Remember: Always be professional and helpful!
"""

def main():
    print("HR FAQ Chatbot - DialoGPT-large Demo")
    print("=" * 50)
    print(get_system_prompt())
    print("=" * 50)
    
    # Check if model exists
    if not os.path.exists(OUTPUT_DIR):
        print(f"Error: Trained model not found at {OUTPUT_DIR}")
        print("Please run training first.")
        return
    
    # Load model
    model, tokenizer = load_trained_model()
    
    if model is None or tokenizer is None:
        print("Failed to load model. Exiting.")
        return
    
    print("\nModel loaded successfully!")
    print("Type 'quit' to exit, 'help' for examples")
    print("-" * 50)
    
    # Interactive loop
    while True:
        try:
            # Get user input
            question = input("\nYou: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if question.lower() == 'help':
                print("\nExample HR questions:")
                print("- What is the company's vacation policy?")
                print("- How do I report workplace harassment?")
                print("- What are the remote work guidelines?")
                print("- How do I request time off?")
                print("- What training opportunities are available?")
                continue
            
            if not question:
                continue
            
            # Check if it's an HR question
            if not is_hr_question(question):
                print("Bot: This question doesn't seem to be related to HR policies. Please contact the HR department for assistance.")
                continue
            
            # Generate response
            print("Bot: ", end="", flush=True)
            response = generate_response(model, tokenizer, question)
            print(response)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()
