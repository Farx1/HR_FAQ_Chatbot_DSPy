"""
DSPy module for HR FAQ chatbot
Uses DSPy to optimize prompts and improve response quality
"""

import os
import json
import torch
import dspy
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

# Model configuration
MODEL_NAME = "microsoft/DialoGPT-small"


class HRFAQAdapter(dspy.BaseLM):
    """
    Custom DSPy adapter for the fine-tuned HR FAQ model
    Wraps the fine-tuned DialoGPT model to work with DSPy
    """
    
    def __init__(self, model_path="models/hr_faq_dialogpt_lora", adapter_path="models/hr_faq_dialogpt_lora_adapters", **kwargs):
        super().__init__(model="hr_faq_dialogpt", **kwargs)
        self.model_path = model_path
        self.adapter_path = adapter_path
        self.pytorch_model = None  # Renamed to avoid conflict with BaseLM.model
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the fine-tuned model"""
        print("Loading fine-tuned HR FAQ model for DSPy...")
        
        # Load tokenizer from base model (more reliable)
        # If saved tokenizer is corrupted, fallback to base model tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        except Exception as e:
            print(f"Warning: Could not load tokenizer from {self.model_path}, using base model tokenizer: {e}")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        # Load LoRA adapters - handle incompatible config fields
        try:
            self.pytorch_model = PeftModel.from_pretrained(base_model, self.adapter_path)
        except TypeError as e:
            # If config has incompatible fields, try to load with a cleaned config
            print(f"Warning: Config incompatibility detected, attempting to fix: {e}")
            import json
            from peft import LoraConfig, get_peft_model
            
            # Load and clean the config
            config_path = os.path.join(self.adapter_path, "adapter_config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config_dict = json.load(f)
                
                # Only keep valid LoraConfig fields
                valid_lora_fields = {
                    "r", "lora_alpha", "target_modules", "lora_dropout", 
                    "task_type", "bias", "inference_mode", "modules_to_save",
                    "init_lora_weights", "layers_to_transform", "layers_pattern",
                    "rank_pattern", "alpha_pattern", "fan_in_fan_out"
                }
                
                # Filter to only valid fields
                cleaned_config = {k: v for k, v in config_dict.items() 
                                 if k in valid_lora_fields and v is not None}
                
                # Create LoRA config from cleaned dict
                lora_config = LoraConfig(**cleaned_config)
                
                # Apply LoRA and load weights
                self.pytorch_model = get_peft_model(base_model, lora_config)
                adapter_weights_path = os.path.join(self.adapter_path, "adapter_model.bin")
                if not os.path.exists(adapter_weights_path):
                    adapter_weights_path = os.path.join(self.adapter_path, "adapter_model.safetensors")
                
                if os.path.exists(adapter_weights_path):
                    from peft import set_peft_model_state_dict
                    try:
                        state_dict = torch.load(adapter_weights_path, map_location="cpu")
                        set_peft_model_state_dict(self.pytorch_model, state_dict)
                    except Exception:
                        # Try safetensors format
                        try:
                            from safetensors import safe_open
                            state_dict = {}
                            with safe_open(adapter_weights_path, framework="pt", device="cpu") as f:
                                for key in f.keys():
                                    state_dict[key] = f.get_tensor(key)
                            set_peft_model_state_dict(self.pytorch_model, state_dict)
                        except Exception as e2:
                            print(f"Warning: Could not load adapter weights: {e2}")
                else:
                    print("Warning: Could not find adapter weights, using base model")
            else:
                raise e
        else:
            self.pytorch_model = self.model  # If loaded successfully, rename it
        
        self.pytorch_model.eval()
        
        print("Model loaded successfully!")
    
    def forward(self, prompt=None, messages=None, **kwargs):
        """Main forward method for DSPy - returns response object compatible with DSPy adapters"""
        if messages:
            # Convert messages to prompt
            prompt = "\n".join([msg.get("content", "") for msg in messages])
        
        if not prompt:
            raise ValueError("Either prompt or messages must be provided")
        
        # Format prompt for the model
        formatted_prompt = f"HR Question: {prompt}\nHR Answer:"
        
        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        
        # Generate
        with torch.no_grad():
            outputs = self.pytorch_model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", 128),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 0.9),
                repetition_penalty=kwargs.get("repetition_penalty", 1.1),
                do_sample=kwargs.get("do_sample", True),
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part
        response = response.split("HR Answer:")[-1].strip()
        
        # Create response object compatible with DSPy adapters
        # DSPy adapters expect an object with choices[].message.content
        class Response:
            def __init__(self, text):
                class Message:
                    def __init__(self, content):
                        self.content = content
                        self.role = "assistant"
                
                class Choice:
                    def __init__(self, message):
                        self.message = message
                        self.logprobs = None
                        self.finish_reason = "stop"
                
                self.choices = [Choice(Message(text))]
                self.usage = {"total_tokens": len(inputs["input_ids"][0]) + len(outputs[0])}
                self.model = "hr_faq_dialogpt"
        
        return Response(response)


class HRFAQSignature(dspy.Signature):
    """Signature for HR FAQ question answering"""
    
    question: str = dspy.InputField(
        desc="The HR-related question from the employee"
    )
    
    answer: str = dspy.OutputField(
        desc="A clear, concise, and professional answer based on HR policies. If the question is outside HR scope, politely indicate this and suggest contacting HR."
    )


class HRFAQModule(dspy.Module):
    """
    DSPy module for HR FAQ chatbot with improved OOD detection and better prompts
    """
    
    def __init__(self, adapter=None):
        super().__init__()
        self.adapter = adapter
        # Keywords to identify HR-related questions
        self.hr_keywords = [
            "vacation", "sick leave", "salary", "review", "training", "policy", 
            "holiday", "time off", "benefits", "insurance", "pension", "retirement",
            "maternity", "paternity", "leave", "harassment", "discrimination",
            "promotion", "raise", "bonus", "contract", "probation", "dress code",
            "remote work", "telework", "flexible", "hours", "overtime", "payroll",
            "hr", "human resources", "employee", "workplace", "safety", "injury"
        ]
        # Keywords that indicate non-HR questions
        self.non_hr_keywords = [
            "install", "python", "programming", "code", "software", "computer",
            "capital", "country", "city", "bake", "cook", "recipe", "cake",
            "weather", "temperature", "rain", "learn", "language", "spanish",
            "quantum", "physics", "science", "invest", "stocks", "money", "trading",
            "car", "engine", "fix", "repair", "machine learning", "ai", "algorithm"
        ]
    
    def _is_hr_related(self, question: str) -> bool:
        """Improved OOD detection using keyword matching"""
        question_lower = question.lower()
        
        # Check for non-HR keywords first (stronger signal)
        for keyword in self.non_hr_keywords:
            if keyword in question_lower:
                return False
        
        # Check for HR keywords
        for keyword in self.hr_keywords:
            if keyword in question_lower:
                return True
        
        # If question is very short or unclear, default to HR (let model decide)
        if len(question.split()) < 3:
            return True
        
        # Default: assume HR-related if no clear non-HR signal
        return True
    
    def forward(self, question: str):
        """Generate answer for HR question with improved OOD detection"""
        try:
            # First, check if question is HR-related
            is_hr = self._is_hr_related(question)
            
            if not is_hr:
                # Return professional rejection message
                rejection_msg = "I apologize, but this question appears to be outside the scope of HR policies. Please contact the HR department for assistance with non-HR related inquiries."
                return dspy.Prediction(answer=rejection_msg)
            
            # If adapter is available, use it directly
            if self.adapter and hasattr(self.adapter, 'pytorch_model'):
                # Improved prompt with better structure
                formatted_prompt = f"You are a professional HR assistant. Answer clearly and concisely.\n\nHR Question: {question}\nHR Answer:"
                inputs = self.adapter.tokenizer(formatted_prompt, return_tensors="pt")
                
                with torch.no_grad():
                    outputs = self.adapter.pytorch_model.generate(
                        **inputs,
                        max_new_tokens=150,  # Increased for better responses
                        temperature=0.5,  # Lower temperature for more focused answers
                        top_p=0.85,
                        repetition_penalty=1.15,  # Increased to avoid repetition
                        do_sample=True,
                        pad_token_id=self.adapter.tokenizer.eos_token_id
                    )
                
                response = self.adapter.tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = response.split("HR Answer:")[-1].strip()
                
                # Clean up the answer
                if answer:
                    # Remove common artifacts
                    answer = answer.split("\n")[0].strip()  # Take first line
                    # Remove if answer is too short or nonsensical
                    if len(answer) < 5 or answer.lower() in ["yes", "no", "ok", "sure"]:
                        answer = ""
                
                # Improved fallback with better template
                if not answer or len(answer) < 10:
                    # Use a more informative template
                    question_clean = question.lower().replace("?", "").strip()
                    answer = f"According to company HR policies, {question_clean} is handled through established procedures. For specific details, please contact the HR department or consult the employee handbook."
                
                return dspy.Prediction(answer=answer)
            
            # Fallback: use Predict
            generate_answer = dspy.Predict(HRFAQSignature)
            result = generate_answer(question=question)
            if result and hasattr(result, 'answer') and result.answer:
                return result
            else:
                return dspy.Prediction(answer="")
        except Exception as e:
            print(f"Error in HRFAQModule.forward: {e}")
            import traceback
            traceback.print_exc()
            return dspy.Prediction(answer="")


class HRFAQRejectionSignature(dspy.Signature):
    """Signature for detecting out-of-domain questions"""
    
    question: str = dspy.InputField(
        desc="The question to evaluate"
    )
    
    is_hr_related: bool = dspy.OutputField(
        desc="True if the question is related to HR policies, False otherwise"
    )
    
    reasoning: str = dspy.OutputField(
        desc="Brief reasoning for the classification"
    )


class HRFAQWithRejection(dspy.Module):
    """
    Enhanced HR FAQ module with OOD detection
    """
    
    def __init__(self):
        super().__init__()
        self.classify_question = dspy.ChainOfThought(HRFAQRejectionSignature)
        self.generate_answer = dspy.ChainOfThought(HRFAQSignature)
    
    def forward(self, question: str):
        """Generate answer with OOD detection"""
        # First, classify if question is HR-related
        classification = self.classify_question(question=question)
        
        if not classification.is_hr_related:
            # Return rejection message
            return dspy.Prediction(
                answer="Désolé, cette question semble en dehors du périmètre des politiques RH. Veuillez contacter le service RH pour plus d'informations.",
                is_hr_related=False,
                reasoning=classification.reasoning
            )
        
        # Generate answer for HR question
        result = self.generate_answer(question=question)
        return dspy.Prediction(
            answer=result.answer,
            is_hr_related=True,
            reasoning=classification.reasoning
        )


def load_evaluation_data():
    """Load evaluation datasets"""
    hr_data = []
    ood_data = []
    
    # Load validation data
    if os.path.exists("data/val_alpaca.json"):
        with open("data/val_alpaca.json", "r", encoding="utf-8") as f:
            hr_data = json.load(f)
    
    # Load OOD data
    if os.path.exists("data/ood_test.json"):
        with open("data/ood_test.json", "r", encoding="utf-8") as f:
            ood_data = json.load(f)
    
    # Create DSPy examples
    hr_examples = []
    for item in hr_data:
        hr_examples.append(dspy.Example(
            question=item["instruction"],
            answer=item["output"]
        ).with_inputs("question"))
    
    ood_examples = []
    for item in ood_data:
        ood_examples.append(dspy.Example(
            question=item["instruction"],
            answer=item["output"]  # Should be rejection message
        ).with_inputs("question"))
    
    return hr_examples, ood_examples


def create_metric_function():
    """Create evaluation metric for DSPy optimizer"""
    
    def metric(example, prediction, trace=None):
        """Compute metric score for a prediction"""
        # Simple exact match for now
        # Can be enhanced with ROUGE, BLEU, etc.
        expected = example.answer.lower().strip()
        predicted = prediction.answer.lower().strip()
        
        # Exact match
        if expected == predicted:
            return 1.0
        
        # Partial match (check if key words are present)
        expected_words = set(expected.split())
        predicted_words = set(predicted.split())
        
        if len(expected_words) == 0:
            return 0.0
        
        overlap = len(expected_words & predicted_words) / len(expected_words)
        return overlap
    
    return metric

