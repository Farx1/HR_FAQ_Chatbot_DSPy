"""
Interactive demo for HR FAQ chatbot (CPU version)
Provides a simple command-line interface to test the fine-tuned model
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# Model configuration
MODEL_NAME = "microsoft/DialoGPT-small"

def load_model():
    """Load the trained model and tokenizer"""
    import json
    from peft import LoraConfig
    
    print("Loading HR FAQ chatbot...")
    
    adapter_path = "models/hr_faq_dialogpt_lora"
    
    try:
        # Try to load tokenizer from saved model, fallback to base model if corrupted
        try:
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, use_fast=False)
        except Exception as e:
            print(f"Warning: Could not load tokenizer from saved model, using base model tokenizer")
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
        
        # Load LoRA adapters with robust config filtering
        try:
            model = PeftModel.from_pretrained(base_model, adapter_path)
        except TypeError as e:
            print("Filtering incompatible config fields...")
            # Load and filter adapter config
            config_path = os.path.join(adapter_path, "adapter_config.json")
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Keep only known valid fields for LoraConfig
            valid_fields = [
                'r', 'lora_alpha', 'lora_dropout', 'target_modules', 'bias',
                'task_type', 'fan_in_fan_out', 'modules_to_save', 'init_lora_weights',
                'layers_to_transform', 'layers_pattern', 'rank_pattern', 'alpha_pattern'
            ]
            filtered_config = {k: v for k, v in config_dict.items() if k in valid_fields}
            
            # Create LoraConfig and load model
            lora_config = LoraConfig(**filtered_config)
            model = PeftModel(base_model, lora_config)
            
            # Load adapter weights
            adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
            if os.path.exists(adapter_weights_path):
                from safetensors.torch import load_file
                state_dict = load_file(adapter_weights_path)
                model.load_state_dict(state_dict, strict=False)
        
        print("Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please make sure you have trained the model first by running:")
        print("  python training/train_cpu.py")
        return None, None

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

def print_welcome():
    """Print welcome message"""
    
    print("=" * 60)
    print("CHATBOT FAQ RH - DIALOGPT (CPU DEMO)")
    print("=" * 60)
    print("Assistant RH professionnel fine-tune sur DialoGPT-small")
    print("")
    print("Exemples de questions:")
    print("  • How many vacation days do I get per year?")
    print("  • What is the remote work policy?")
    print("  • How do I report workplace harassment?")
    print("  • What training opportunities are available?")
    print("")
    print("Questions hors domaine:")
    print("  • How do I install Python?")
    print("  • What is the capital of France?")
    print("")
    print("Tapez 'quit' pour quitter, 'help' pour l'aide")
    print("=" * 60)

def print_help():
    """Print help message"""
    
    print("\nAIDE - CHATBOT FAQ RH")
    print("-" * 30)
    print("Ce chatbot repond aux questions sur les politiques RH de l'entreprise.")
    print("")
    print("Domaines couverts:")
    print("  • Conges et absences")
    print("  • Contrats et salaires")
    print("  • Teletravail")
    print("  • Formation et developpement")
    print("  • Recrutement")
    print("  • Harcelement et securite")
    print("")
    print("Le chatbot refusera poliment les questions hors domaine RH.")
    print("")
    print("Commandes:")
    print("  • 'quit' ou 'exit' : Quitter")
    print("  • 'help' : Afficher cette aide")
    print("  • 'clear' : Effacer l'ecran")
    print("")

def run_demo():
    """Main demo function"""
    
    # Print welcome
    print_welcome()
    
    # Load model
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        return
    
    print("\nPret a repondre a vos questions RH !")
    print("")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            question = input("Votre question RH: ").strip()
            
            # Handle special commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nMerci d'avoir utilise le chatbot RH. A bientot !")
                break
            
            elif question.lower() == 'help':
                print_help()
                continue
            
            elif question.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_welcome()
                continue
            
            elif not question:
                print("Veillez poser une question.")
                continue
            
            # Generate response
            print("\nReponse:")
            print("-" * 40)
            
            response = generate_response(model, tokenizer, question)
            print(response)
            
            print("-" * 40)
            print("")
            
        except KeyboardInterrupt:
            print("\nInterruption detectee. Au revoir !")
            break
        
        except Exception as e:
            print(f"\nErreur: {e}")
            print("Veuillez reessayer.")

def run_batch_test():
    """Run batch test with predefined questions"""
    
    print("Test par lot - Questions predefinies")
    print("=" * 50)
    
    # Load model
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        return
    
    # Test questions
    test_questions = [
        "How many vacation days do I get per year?",
        "What is the remote work policy?",
        "How do I report workplace harassment?",
        "What training opportunities are available?",
        "How do I request time off?",
        "What is the dress code policy?",
        "How do I install Python on my computer?",  # OOD
        "What is the capital of France?",  # OOD
        "How do I bake a chocolate cake?",  # OOD
    ]
    
    print(f"Test de {len(test_questions)} questions...")
    print("")
    
    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}: {question}")
        print("Reponse:")
        
        response = generate_response(model, tokenizer, question)
        print(response)
        
        print("-" * 60)
        print("")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_batch_test()
    else:
        run_demo()
