"""
Interactive demo for HR FAQ chatbot
Provides a simple command-line interface to test the fine-tuned model
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# Model configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

def load_model():
    """Load the trained model and tokenizer"""
    
    print("Loading HR FAQ chatbot...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("models/hr_faq_mistral_lora")
        
        # Load base model with quantization
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
        
        print("✓ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("Please make sure you have trained the model first by running:")
        print("  python training/train.py")
        return None, None

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

def print_welcome():
    """Print welcome message"""
    
    print("=" * 60)
    print("🤖 CHATBOT FAQ RH - MISTRAL AI")
    print("=" * 60)
    print("Assistant RH professionnel fine-tuné sur Mistral-7B-Instruct-v0.3")
    print("")
    print("💡 Exemples de questions:")
    print("  • Combien de jours de congé ai-je par an ?")
    print("  • Quelle est la politique de télétravail ?")
    print("  • Comment signaler un harcèlement ?")
    print("  • Quelles formations sont disponibles ?")
    print("")
    print("❌ Questions hors domaine:")
    print("  • Comment installer Python ?")
    print("  • Quelle est la capitale de la France ?")
    print("")
    print("Tapez 'quit' pour quitter, 'help' pour l'aide")
    print("=" * 60)

def print_help():
    """Print help message"""
    
    print("\n📋 AIDE - CHATBOT FAQ RH")
    print("-" * 30)
    print("Ce chatbot répond aux questions sur les politiques RH de l'entreprise.")
    print("")
    print("🎯 Domaines couverts:")
    print("  • Congés et absences")
    print("  • Contrats et salaires")
    print("  • Télétravail")
    print("  • Formation et développement")
    print("  • Recrutement")
    print("  • Harcèlement et sécurité")
    print("")
    print("⚠️  Le chatbot refusera poliment les questions hors domaine RH.")
    print("")
    print("🔧 Commandes:")
    print("  • 'quit' ou 'exit' : Quitter")
    print("  • 'help' : Afficher cette aide")
    print("  • 'clear' : Effacer l'écran")
    print("")

def run_demo():
    """Main demo function"""
    
    # Print welcome
    print_welcome()
    
    # Load model
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        return
    
    print("\n🚀 Prêt à répondre à vos questions RH !")
    print("")
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            question = input("❓ Votre question RH: ").strip()
            
            # Handle special commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Merci d'avoir utilisé le chatbot RH. À bientôt !")
                break
            
            elif question.lower() == 'help':
                print_help()
                continue
            
            elif question.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_welcome()
                continue
            
            elif not question:
                print("⚠️  Veuillez poser une question.")
                continue
            
            # Generate response
            print("\n🤖 Réponse:")
            print("-" * 40)
            
            response = generate_response(model, tokenizer, question)
            print(response)
            
            print("-" * 40)
            print("")
            
        except KeyboardInterrupt:
            print("\n\n👋 Interruption détectée. Au revoir !")
            break
        
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            print("Veuillez réessayer.")

def run_batch_test():
    """Run batch test with predefined questions"""
    
    print("🧪 Test par lot - Questions prédéfinies")
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
        print("Réponse:")
        
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
