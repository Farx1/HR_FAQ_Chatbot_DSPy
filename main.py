"""
Main script to run the complete HR FAQ fine-tuning pipeline
Orchestrates data preparation, training, evaluation, and demo
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_script(script_path: str, description: str):
    """Run a Python script with error handling"""
    
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.getcwd())
        
        if result.returncode == 0:
            print(f"{description} completed successfully!")
            if result.stdout:
                print("Output:")
                print(result.stdout)
        else:
            print(f"{description} failed!")
            print("Error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"Error running {description}: {e}")
        return False
    
    return True

def check_requirements():
    """Check if required packages are installed"""
    
    print("Checking requirements...")
    
    required_packages = [
        "torch", "transformers", "datasets", "peft", 
        "bitsandbytes", "accelerate", "evaluate"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"Package {package} OK")
        except ImportError:
            print(f"Package {package} MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install them with: pip install -r requirements.txt")
        return False
    
    print("All requirements satisfied!")
    return True

def main():
    """Main pipeline function"""
    
    parser = argparse.ArgumentParser(description="HR FAQ Fine-tuning Pipeline")
    parser.add_argument("--step", choices=["data", "train", "eval", "demo", "all"], 
                       default="all", help="Which step to run")
    parser.add_argument("--skip-checks", action="store_true", 
                       help="Skip requirement checks")
    
    args = parser.parse_args()
    
    print("HR FAQ Fine-tuning Pipeline - Mistral AI")
    print("=" * 60)
    
    # Check requirements unless skipped
    if not args.skip_checks:
        if not check_requirements():
            print(f"\nRequirements check failed. Exiting.")
            return
    
    # Create necessary directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    
    success = True
    
    # Run pipeline steps
    if args.step in ["data", "all"]:
        success &= run_script("data/prepare_data.py", "Data Preparation")
    
    if args.step in ["train", "all"] and success:
        success &= run_script("training/train_cpu.py", "Model Training")
    
    if args.step in ["eval", "all"] and success:
        success &= run_script("evaluation/eval_cpu.py", "Model Evaluation")
    
    if args.step in ["demo", "all"] and success:
        print(f"\n{'='*60}")
        print("Starting Interactive Demo")
        print(f"{'='*60}")
        print("Run the demo manually with: python demo/interactive_demo_cpu.py")
        print("Or run batch test with: python demo/interactive_demo_cpu.py test")
    
    # Final summary
    print(f"\n{'='*60}")
    if success:
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("\nGenerated files:")
        
        # List generated files
        if os.path.exists("data/train_alpaca.json"):
            print("  • data/train_alpaca.json")
        if os.path.exists("data/val_alpaca.json"):
            print("  • data/val_alpaca.json")
        if os.path.exists("data/ood_test.json"):
            print("  • data/ood_test.json")
        if os.path.exists("models/hr_faq_mistral_lora"):
            print("  • models/hr_faq_mistral_lora/")
        if os.path.exists("models/hr_faq_mistral_lora_adapters"):
            print("  • models/hr_faq_mistral_lora_adapters/")
        if os.path.exists("reports/evaluation_results.json"):
            print("  • reports/evaluation_results.json")
        if os.path.exists("reports/evaluation_report.md"):
            print("  • reports/evaluation_report.md")
        
        print("\nNext steps:")
        print("  1. Review the evaluation report: reports/evaluation_report.md")
        print("  2. Test the chatbot: python demo/interactive_demo_cpu.py")
        print("  3. Deploy the model for production use")
        
    else:
        print("PIPELINE FAILED!")
        print("Please check the error messages above and fix the issues.")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
