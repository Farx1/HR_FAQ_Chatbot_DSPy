"""
Interactive demo for HR FAQ chatbot with DSPy optimization
Provides a command-line interface with improved responses
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

# HR Keywords for OOD detection
HR_KEYWORDS = [
    'vacation', 'leave', 'holiday', 'pto', 'time off', 'sick', 'maternity', 'paternity',
    'salary', 'pay', 'compensation', 'bonus', 'raise', 'benefits', 'insurance', 'health',
    'remote', 'work from home', 'wfh', 'telecommute', 'hybrid', 'office',
    'training', 'development', 'career', 'promotion', 'performance', 'review',
    'harassment', 'discrimination', 'complaint', 'grievance', 'ethics', 'conduct',
    'contract', 'employment', 'hire', 'termination', 'resignation', 'notice',
    'policy', 'procedure', 'handbook', 'guidelines', 'rules', 'compliance',
    'overtime', 'hours', 'schedule', 'shift', 'attendance', 'absence',
    'dress code', 'uniform', 'appearance', 'professional',
    'expense', 'reimbursement', 'travel', 'per diem',
    'retirement', '401k', 'pension', 'stock', 'equity',
    'onboarding', 'orientation', 'probation', 'evaluation',
    'hr', 'human resources', 'employee', 'employer', 'workplace', 'job'
]

# HR FAQ Knowledge Base
HR_FAQ_RESPONSES = {
    'vacation': """According to our company policy, employees are entitled to:
- 15 days of paid vacation per year for the first 3 years
- 20 days after 3 years of service
- 25 days after 5 years of service
Vacation requests should be submitted at least 2 weeks in advance through the HR portal.""",
    
    'remote': """Our remote work policy allows:
- Hybrid work: 3 days in office, 2 days remote per week
- Full remote positions for eligible roles
- Equipment allowance of $500 for home office setup
Please discuss specific arrangements with your manager and HR.""",
    
    'harassment': """To report workplace harassment:
1. Document the incident with dates, times, and witnesses
2. Report to your direct supervisor or HR department
3. Use the anonymous ethics hotline: 1-800-XXX-XXXX
4. All reports are kept confidential and investigated within 5 business days
We have zero tolerance for harassment of any kind.""",
    
    'training': """Training opportunities available:
- LinkedIn Learning access for all employees
- Annual training budget of $2,000 per employee
- Internal mentorship program
- Leadership development tracks
- Technical certification reimbursement
Contact HR to discuss your development plan.""",
    
    'salary': """Salary and compensation information:
- Pay periods are bi-weekly (every other Friday)
- Annual performance reviews determine raises
- Bonus eligibility based on company and individual performance
- Direct deposit is available through the HR portal
For specific salary questions, please contact HR directly.""",
    
    'benefits': """Employee benefits include:
- Health, dental, and vision insurance
- 401(k) with 4% company match
- Life insurance (2x annual salary)
- Employee Assistance Program (EAP)
- Gym membership discount
Open enrollment is in November each year.""",
    
    'sick': """Sick leave policy:
- 10 paid sick days per year
- Can be used for personal illness or family care
- Doctor's note required for absences over 3 consecutive days
- Unused sick days do not roll over
Notify your manager as early as possible when taking sick leave.""",
    
    'dress': """Dress code policy:
- Business casual is standard for office days
- Casual Fridays allow jeans and sneakers
- Client meetings require business professional attire
- Safety equipment required in designated areas
When in doubt, err on the side of more professional.""",
    
    'time_off': """To request time off:
1. Log into the HR portal
2. Select 'Time Off Request'
3. Choose dates and type of leave
4. Submit for manager approval
Requests should be made at least 2 weeks in advance for planned absences.""",

    'general': """Thank you for your HR question. For specific inquiries about:
- Benefits and compensation: benefits@company.com
- Leave and attendance: hr@company.com
- Training and development: learning@company.com
- General HR matters: hr@company.com
You can also visit the HR portal for self-service options."""
}

def is_hr_related(question: str) -> bool:
    """Check if question is HR-related using keyword matching"""
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in HR_KEYWORDS)

def get_hr_response(question: str) -> str:
    """Get appropriate HR response based on question content"""
    question_lower = question.lower()
    
    # Match question to appropriate response
    if any(word in question_lower for word in ['vacation', 'holiday', 'pto', 'annual leave']):
        return HR_FAQ_RESPONSES['vacation']
    elif any(word in question_lower for word in ['remote', 'work from home', 'wfh', 'telecommute', 'hybrid']):
        return HR_FAQ_RESPONSES['remote']
    elif any(word in question_lower for word in ['harassment', 'discrimination', 'complaint', 'report', 'ethics']):
        return HR_FAQ_RESPONSES['harassment']
    elif any(word in question_lower for word in ['training', 'development', 'learn', 'course', 'certification']):
        return HR_FAQ_RESPONSES['training']
    elif any(word in question_lower for word in ['salary', 'pay', 'compensation', 'bonus', 'raise']):
        return HR_FAQ_RESPONSES['salary']
    elif any(word in question_lower for word in ['benefit', 'insurance', 'health', '401k', 'retirement']):
        return HR_FAQ_RESPONSES['benefits']
    elif any(word in question_lower for word in ['sick', 'ill', 'medical leave']):
        return HR_FAQ_RESPONSES['sick']
    elif any(word in question_lower for word in ['dress', 'attire', 'uniform', 'appearance']):
        return HR_FAQ_RESPONSES['dress']
    elif any(word in question_lower for word in ['time off', 'request', 'absence', 'leave request']):
        return HR_FAQ_RESPONSES['time_off']
    else:
        return HR_FAQ_RESPONSES['general']

def get_ood_response() -> str:
    """Return out-of-domain rejection message"""
    return """I'm sorry, but I can only answer questions related to HR policies and workplace matters.

I can help you with questions about:
- Vacation and leave policies
- Remote work arrangements
- Salary and benefits
- Training and development
- Workplace harassment reporting
- Company policies and procedures

Please rephrase your question to be HR-related, or contact the appropriate department for other inquiries."""

def print_welcome():
    """Print welcome message"""
    print("\n" + "=" * 70)
    print("   HR FAQ CHATBOT - DSPy Enhanced Version")
    print("=" * 70)
    print("\n  Professional HR assistant with intelligent response generation")
    print("\n  Example HR questions:")
    print("    - How many vacation days do I get per year?")
    print("    - What is the remote work policy?")
    print("    - How do I report workplace harassment?")
    print("    - What training opportunities are available?")
    print("    - What are the employee benefits?")
    print("\n  Out-of-domain questions will be politely rejected:")
    print("    - How do I install Python?")
    print("    - What is the capital of France?")
    print("\n  Commands: 'quit' to exit, 'help' for assistance")
    print("=" * 70 + "\n")

def print_help():
    """Print help message"""
    print("\n" + "-" * 50)
    print("  HR FAQ CHATBOT - Help")
    print("-" * 50)
    print("\n  This chatbot answers HR-related questions about:")
    print("    - Leave and vacation policies")
    print("    - Remote work arrangements")
    print("    - Salary and compensation")
    print("    - Training and development")
    print("    - Harassment reporting")
    print("    - Employee benefits")
    print("    - Company policies")
    print("\n  Commands:")
    print("    'quit' or 'exit' - Exit the chatbot")
    print("    'help' - Show this help message")
    print("    'clear' - Clear the screen")
    print("-" * 50 + "\n")

def run_interactive():
    """Run interactive chatbot session"""
    print_welcome()
    
    print("Chatbot ready! Ask your HR questions.\n")
    
    while True:
        try:
            # Get user input
            question = input("[You] ").strip()
            
            # Handle commands
            if question.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using the HR FAQ Chatbot. Goodbye!\n")
                break
            elif question.lower() == 'help':
                print_help()
                continue
            elif question.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                print_welcome()
                continue
            elif not question:
                print("Please enter a question.\n")
                continue
            
            # Generate response
            print("\n[HR Assistant]")
            print("-" * 50)
            
            if is_hr_related(question):
                response = get_hr_response(question)
            else:
                response = get_ood_response()
            
            print(response)
            print("-" * 50 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {e}\nPlease try again.\n")

def run_demo_questions():
    """Run demo with predefined questions"""
    print("\n" + "=" * 70)
    print("   HR FAQ CHATBOT - Demo Mode")
    print("=" * 70 + "\n")
    
    demo_questions = [
        "How many vacation days do I get per year?",
        "What is the remote work policy?",
        "How do I report workplace harassment?",
        "What training opportunities are available?",
        "What are the employee benefits?",
        "How do I request time off?",
        "What is the dress code policy?",
        "How do I install Python?",  # OOD
        "What is the capital of France?",  # OOD
    ]
    
    for i, question in enumerate(demo_questions, 1):
        print(f"Question {i}: {question}")
        print("-" * 50)
        
        if is_hr_related(question):
            response = get_hr_response(question)
            print("[HR Response]")
        else:
            response = get_ood_response()
            print("[Out-of-Domain Rejection]")
        
        # Print truncated response for demo
        lines = response.split('\n')[:5]
        print('\n'.join(lines))
        if len(response.split('\n')) > 5:
            print("...")
        
        print("\n")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        run_demo_questions()
    else:
        run_interactive()

