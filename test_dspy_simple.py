"""
Simple test script to verify DSPy integration works
"""

import os
import sys

# Add dspy_module to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from dspy_module.hr_faq_dspy import HRFAQAdapter, HRFAQModule
    import dspy
    
    print("Testing DSPy integration...")
    print("="*50)
    
    # Check if model exists
    if not os.path.exists("models/hr_faq_dialogpt_lora"):
        print("ERROR: Model not found. Please train the model first:")
        print("  python training/train_cpu.py")
        sys.exit(1)
    
    # Configure DSPy
    print("\n1. Configuring DSPy...")
    adapter = HRFAQAdapter()
    dspy.configure(lm=adapter)
    print("   [OK] DSPy configured")
    
    # Create module
    print("\n2. Creating DSPy module...")
    module = HRFAQModule()
    print("   [OK] Module created")
    
    # Test with a simple question
    print("\n3. Testing with a sample question...")
    test_question = "How many vacation days do I get per year?"
    print(f"   Question: {test_question}")
    
    try:
        result = module(question=test_question)
        print(f"   Answer: {result.answer}")
        print("   [OK] Test successful!")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "="*50)
    print("DSPy integration test completed successfully!")
    print("You can now run: python dspy_module/benchmark_dspy.py")
    
except ImportError as e:
    print(f"ERROR: Failed to import DSPy modules: {e}")
    print("Please install DSPy: pip install dspy-ai")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

