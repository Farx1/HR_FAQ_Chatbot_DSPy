"""
DSPy optimization script for HR FAQ chatbot
Uses DSPy optimizers to improve prompt quality
"""

import os
import json
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dspy_module.hr_faq_dspy import (
    HRFAQAdapter,
    HRFAQModule,
    load_evaluation_data,
    create_metric_function
)
import dspy


def optimize_module():
    """Optimize DSPy module using MIPRO or BootstrapFewShot"""
    print("DSPy Optimization for HR FAQ Chatbot")
    print("="*60)
    
    # Configure DSPy with custom adapter
    print("\n1. Configuring DSPy with fine-tuned model...")
    adapter = HRFAQAdapter()
    dspy.configure(lm=adapter)
    
    # Load evaluation data
    print("\n2. Loading evaluation data...")
    hr_examples, ood_examples = load_evaluation_data()
    
    if len(hr_examples) < 3:
        print("Warning: Not enough examples for optimization. Creating sample data...")
        hr_examples = [
            dspy.Example(
                question="What is the company's sick leave policy?",
                answer="Employees can take up to 5 sick days per year with manager approval. Extended sick leave requires medical documentation."
            ).with_inputs("question"),
            dspy.Example(
                question="How do I request a salary review?",
                answer="Submit a salary review request through the HR portal with supporting documentation. Reviews are conducted annually or upon promotion."
            ).with_inputs("question"),
            dspy.Example(
                question="What training opportunities are available?",
                answer="The company offers various training programs including technical skills, leadership development, and professional certifications. Check the learning portal for available courses."
            ).with_inputs("question")
        ]
    
    # Split into train/val (80/20 for prompt optimization)
    split_idx = int(0.8 * len(hr_examples))
    trainset = hr_examples[:split_idx] if split_idx > 0 else hr_examples
    valset = hr_examples[split_idx:] if split_idx < len(hr_examples) else []
    
    print(f"  Train set: {len(trainset)} examples")
    print(f"  Validation set: {len(valset)} examples")
    
    # Create metric function
    print("\n3. Creating evaluation metric...")
    metric = create_metric_function()
    
    # Initialize module
    print("\n4. Initializing DSPy module...")
    module = HRFAQModule()
    
    # Evaluate before optimization
    print("\n5. Evaluating before optimization...")
    if valset:
        from dspy.evaluate import Evaluate
        evaluate = Evaluate(devset=valset, metric=metric, num_threads=1, display_progress=True)
        baseline_score = evaluate(module)
        print(f"  Baseline score: {baseline_score:.3f}")
    
    # Optimize using BootstrapFewShot (simpler, faster)
    print("\n6. Optimizing module with BootstrapFewShot...")
    try:
        from dspy.teleprompt import BootstrapFewShot
        
        optimizer = BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=4,
            max_labeled_demos=2,
            max_rounds=1
        )
        
        optimized_module = optimizer.compile(
            module,
            trainset=trainset,
            valset=valset if valset else trainset
        )
        
        print("  Optimization completed!")
        
        # Evaluate after optimization
        if valset:
            print("\n7. Evaluating after optimization...")
            optimized_score = evaluate(optimized_module)
            print(f"  Optimized score: {optimized_score:.3f}")
            print(f"  Improvement: {optimized_score - baseline_score:+.3f}")
        
        # Save optimized module
        print("\n8. Saving optimized module...")
        os.makedirs("models", exist_ok=True)
        optimized_module.save("models/dspy_optimized_module.json")
        print("  Saved to: models/dspy_optimized_module.json")
        
        return optimized_module
        
    except Exception as e:
        print(f"  Error during optimization: {e}")
        print("  Falling back to non-optimized module...")
        return module


def main():
    """Main optimization function"""
    try:
        optimized_module = optimize_module()
        print("\n" + "="*60)
        print("Optimization completed successfully!")
        print("="*60)
        print("\nYou can now use the optimized module in benchmark_dspy.py")
        print("by setting optimized=True")
        
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: Optimization requires sufficient training data.")
        print("The module will still work without optimization.")


if __name__ == "__main__":
    main()

