"""
Configuration file for HR FAQ fine-tuning project
Centralizes all hyperparameters and settings
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_length: int = 512
    learning_rate: float = 2e-4
    num_epochs: int = 2
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

@dataclass
class LoRAConfig:
    """LoRA configuration"""
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "q_proj", "v_proj", "k_proj", "o_proj", 
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass
class QuantizationConfig:
    """Quantization configuration"""
    load_in_4bit: bool = True
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"

@dataclass
class GenerationConfig:
    """Text generation configuration"""
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    do_sample: bool = True

@dataclass
class SystemPromptConfig:
    """System prompt configuration"""
    french_prompt: str = "Tu es un assistant RH professionnel. Réponds de façon claire, concise et exacte sur la base des politiques RH disponibles. Si la question sort du périmètre RH ou si l'information manque, indique-le poliment et propose de contacter le service RH."
    
    english_prompt: str = "You are a professional HR assistant. Answer clearly, concisely and accurately based on available HR policies. If the question is outside the HR scope or if information is missing, politely indicate this and suggest contacting the HR department."
    
    rejection_message: str = "Désolé, cette question semble en dehors du périmètre des politiques RH. Veuillez contacter le service RH."

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    exact_match_threshold: float = 0.7
    ood_rejection_threshold: float = 0.8
    rejection_keywords: list = None
    
    def __post_init__(self):
        if self.rejection_keywords is None:
            self.rejection_keywords = [
                "désolé", "périmètre", "rh", "contacter", "service",
                "sorry", "scope", "hr", "contact", "department"
            ]

@dataclass
class DataConfig:
    """Data configuration"""
    dataset_name: str = "syncora/hr-policies-qa-dataset"
    train_split: float = 0.9
    val_split: float = 0.1
    min_instruction_length: int = 10
    max_instruction_length: int = 500
    min_output_length: int = 10
    max_output_length: int = 1000

@dataclass
class ProjectConfig:
    """Main project configuration"""
    random_seed: int = 42
    project_name: str = "HR FAQ Fine-tuning"
    model_config: ModelConfig = None
    lora_config: LoRAConfig = None
    quantization_config: QuantizationConfig = None
    generation_config: GenerationConfig = None
    system_prompt_config: SystemPromptConfig = None
    evaluation_config: EvaluationConfig = None
    data_config: DataConfig = None
    
    def __post_init__(self):
        if self.model_config is None:
            self.model_config = ModelConfig()
        if self.lora_config is None:
            self.lora_config = LoRAConfig()
        if self.quantization_config is None:
            self.quantization_config = QuantizationConfig()
        if self.generation_config is None:
            self.generation_config = GenerationConfig()
        if self.system_prompt_config is None:
            self.system_prompt_config = SystemPromptConfig()
        if self.evaluation_config is None:
            self.evaluation_config = EvaluationConfig()
        if self.data_config is None:
            self.data_config = DataConfig()

# Global configuration instance
config = ProjectConfig()

# Environment variables
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")
HF_HOME = os.getenv("HF_HOME", "./cache")
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", "./cache")

# Paths
DATA_DIR = "data"
MODELS_DIR = "models"
REPORTS_DIR = "reports"
CACHE_DIR = "cache"

# File paths
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "train_alpaca.json")
VAL_DATA_PATH = os.path.join(DATA_DIR, "val_alpaca.json")
OOD_TEST_PATH = os.path.join(DATA_DIR, "ood_test.json")
MODEL_PATH = os.path.join(MODELS_DIR, "hr_faq_mistral_lora")
ADAPTERS_PATH = os.path.join(MODELS_DIR, "hr_faq_mistral_lora_adapters")
EVALUATION_RESULTS_PATH = os.path.join(REPORTS_DIR, "evaluation_results.json")
EVALUATION_REPORT_PATH = os.path.join(REPORTS_DIR, "evaluation_report.md")

def print_config():
    """Print current configuration"""
    print("🔧 Configuration:")
    print(f"  Model: {config.model_config.model_name}")
    print(f"  Learning Rate: {config.model_config.learning_rate}")
    print(f"  Epochs: {config.model_config.num_epochs}")
    print(f"  Batch Size: {config.model_config.batch_size}")
    print(f"  LoRA Rank: {config.lora_config.r}")
    print(f"  Max Length: {config.model_config.max_length}")
    print(f"  Random Seed: {config.random_seed}")

if __name__ == "__main__":
    print_config()
