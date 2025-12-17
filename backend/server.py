"""
FastAPI backend for HR FAQ Chatbot
Uses DSPy module for intelligent responses with RAG integration
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, AsyncGenerator
import time
import asyncio
import json

app = FastAPI(title="HR FAQ API", version="2.0.0", description="HR FAQ Chatbot with RAG support")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
hr_module = None
adapter = None
rag_engine = None


class Message(BaseModel):
    role: str
    content: str


class AskRequest(BaseModel):
    question: str
    mode: Optional[str] = "policy"
    history: Optional[List[Message]] = None


class Source(BaseModel):
    title: str
    snippet: str
    category: Optional[str] = None
    similarity: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    ood_reject: bool = False
    confidence: Optional[float] = None
    latency_ms: Optional[int] = None
    sources: Optional[List[Source]] = None
    company: Optional[str] = None


def load_rag_engine():
    """Load the RAG engine"""
    global rag_engine
    
    try:
        from backend.rag_engine import get_rag_engine
        
        print("Loading RAG engine...")
        rag_engine = get_rag_engine(
            company_data_path="company_data/techcorp_solutions",
            force_init=True
        )
        
        if rag_engine.is_initialized:
            print(f"RAG engine loaded for {rag_engine.get_company_name()}")
            return True
        else:
            print("RAG engine not fully initialized (missing dependencies)")
            return False
    except Exception as e:
        print(f"Warning: Could not load RAG engine: {e}")
        return False


def load_model():
    """Load the DSPy HR FAQ model"""
    global hr_module, adapter
    
    try:
        from dspy_module.hr_faq_dspy import HRFAQAdapter, HRFAQModule
        
        print("Loading HR FAQ model...")
        adapter = HRFAQAdapter(
            model_path="models/hr_faq_dialogpt_lora",
            adapter_path="models/hr_faq_dialogpt_lora_adapters"
        )
        hr_module = HRFAQModule(adapter=adapter)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Warning: Could not load DSPy model: {e}")
        print("Using fallback responses")
        return False


# Fallback responses when model is not available (now TechCorp-specific)
FALLBACK_RESPONSES = {
    "policy": {
        "vacation": "At TechCorp Solutions, full-time employees receive 20-30 vacation days per year depending on tenure. Employees with 0-2 years get 20 days, 3-5 years get 25 days, and 6+ years get 30 days. Submit vacation requests through the HR Portal at least 2 weeks in advance.",
        "remote": "TechCorp Solutions offers hybrid remote work: up to 3 days remote per week with manager approval. You'll receive a $500 one-time home office stipend and $75-100/month internet allowance. Core hours are 10 AM - 3 PM.",
        "sick": "TechCorp employees receive 10 paid sick days per year. Notify your manager by 9 AM on the day of absence. For absences of 3+ consecutive days, a medical certificate is required.",
        "harassment": "TechCorp has zero tolerance for harassment. Report incidents to your manager, HR, or the anonymous hotline at 1-800-555-SAFE. All reports are investigated promptly and confidentially.",
        "dress": "TechCorp maintains a business casual dress code. Jeans are allowed on Casual Fridays. For client meetings, business professional attire is required.",
        "default": "For detailed information about TechCorp Solutions HR policies, please consult the employee handbook on the HR Portal (hr.techcorp-solutions.com) or contact HR at hr@techcorp-solutions.com."
    },
    "benefits": {
        "health": "TechCorp offers three health plans: PPO, HDHP with HSA, and HMO. Coverage includes medical, dental, and vision. TechCorp pays approximately 80% of premium costs. Enroll within 30 days of hire.",
        "401k": "TechCorp's 401(k) plan includes 100% employer match on the first 6% you contribute. The match vests over 4 years (25% per year). Fidelity manages the plan with diverse investment options.",
        "retirement": "TechCorp's 401(k) plan includes 100% employer match on the first 6% you contribute. The match vests over 4 years. Access your account at netbenefits.fidelity.com.",
        "wellness": "TechCorp's wellness program includes $100/month gym reimbursement, free Headspace subscription, 2 mental health days, and up to $500/year in wellness rewards.",
        "education": "TechCorp offers up to $10,000/year for graduate degrees, $5,250 for undergraduate, and $3,000 for certifications. Plus $2,000/year L&D budget for conferences and courses.",
        "default": "For questions about TechCorp employee benefits, visit benefits.techcorp-solutions.com or contact benefits@techcorp-solutions.com."
    },
    "payroll": {
        "payslip": "TechCorp payslips are available on the HR Portal by payday. We pay bi-weekly (26 pay periods). Download PDF versions for your records.",
        "bank": "To update your bank details at TechCorp, submit a request through the HR Portal with your new banking information. Changes take effect within 2 pay periods.",
        "bonus": "TechCorp's annual bonus targets range from 10-25% of base salary depending on level. Bonuses are calculated based on company performance (0-150%) and individual performance (0-150%).",
        "salary": "TechCorp targets 75th percentile market rates. Salary reviews occur annually in January. Merit increases typically range from 3-8% based on performance.",
        "default": "For TechCorp payroll inquiries, contact payroll@techcorp-solutions.com or call +1 (555) 123-4567 ext. 300."
    }
}


def is_hr_related(question: str) -> bool:
    """Check if a question is HR-related"""
    question_lower = question.lower()
    
    # HR-related keywords (check these FIRST - if found, it's HR)
    hr_keywords = [
        "vacation", "leave", "pto", "time off", "holiday", "sick", "absence", "day off", "days off",
        "salary", "pay", "compensation", "bonus", "raise", "payroll", "paycheck",
        "benefit", "insurance", "health", "dental", "vision", "401k", "retirement",
        "remote", "work from home", "wfh", "hybrid", "telecommute",
        "training", "onboarding", "orientation", "learning", "development", "course",
        "policy", "handbook", "guideline", "procedure",
        "hr", "human resources", "employee", "employer", "staff", "workforce",
        "hire", "recruit", "interview", "offer", "probation", "termination",
        "promotion", "review", "performance", "evaluation", "feedback",
        "harassment", "discrimination", "complaint", "grievance", "ethics",
        "expense", "reimbursement", "per diem",
        "maternity", "paternity", "parental", "fmla", "disability",
        "dress code", "attire", "uniform",
        "overtime", "hours", "schedule", "shift", "flexible",
        "referral", "transfer", "relocation"
    ]
    
    # Check for HR keywords first
    for kw in hr_keywords:
        if kw in question_lower:
            return True
    
    # Non-HR keywords (only check if no HR keywords found)
    non_hr_keywords = [
        "python", "code", "programming", "javascript", "java", "software", "bug", "debug",
        "weather", "temperature", "rain", "sunny", "forecast",
        "recipe", "cook", "bake", "food", "ingredient", "cake",
        "capital", "country", "city", "president", "trump", "biden", "obama", "politician",
        "movie", "film", "actor", "actress", "song", "music", "singer",
        "sports", "football", "soccer", "basketball", "game", "score",
        "math", "calculate", "equation", "physics", "chemistry", "biology",
        "stock", "bitcoin", "crypto", "invest", "trading",
        "car", "vehicle", "engine", "repair", "mechanic",
        "born", "birthday", "died", "age", "married", "wife", "husband",
        "translate", "language", "french", "spanish", "german", "chinese",
        "news", "war", "election", "vote", "politics",
        "quantum", "algorithm", "machine learning", "ai model",
        "history", "ancient", "king", "queen", "empire",
        "space", "planet", "star", "galaxy", "nasa",
        "animal", "dog", "cat", "pet", "zoo"
    ]
    
    # Check for non-HR keywords
    for kw in non_hr_keywords:
        if kw in question_lower:
            return False
    
    # If question is very short or generic, be conservative
    if len(question.split()) < 3:
        return False
    
    # Default: assume not HR-related if no clear HR signal
    return False


def get_fallback_response(question: str, mode: str, context: str = "") -> tuple[str, bool]:
    """Get a fallback response based on keywords and RAG context"""
    question_lower = question.lower()
    
    # Check for OOD (out of domain)
    if not is_hr_related(question):
        return "I'm sorry, but this question is outside my expertise as an HR assistant for TechCorp Solutions. I can only help with HR-related topics such as:\n\n• Leave & PTO (vacation, sick leave, holidays)\n• Benefits (health insurance, 401k, wellness)\n• Payroll (salary, bonuses, expenses)\n• Remote work policies\n• Training & onboarding\n• Company policies\n\nPlease ask an HR-related question!", True
    
    # If we have RAG context, use it to build a response
    if context:
        # Extract key information from context
        response = f"Based on TechCorp Solutions' HR policies:\n\n{context[:1500]}"
        if len(context) > 1500:
            response += "\n\nFor more details, please consult the full policy on the HR Portal."
        return response, False
    
    # Get mode-specific responses
    responses = FALLBACK_RESPONSES.get(mode, FALLBACK_RESPONSES["policy"])
    
    # Match keywords
    for keyword, response in responses.items():
        if keyword != "default" and keyword in question_lower:
            return response, False
    
    return responses["default"], False


def generate_rag_response(question: str, context: str, sources: List[dict]) -> str:
    """Generate a response using RAG context - formats context into a clean answer"""
    
    # For DialoGPT-based models, we get better results by formatting the RAG context
    # directly rather than trying to generate new text
    
    if not context or not sources:
        return None
    
    # Extract the most relevant information from context
    question_lower = question.lower()
    
    # Build a clean, formatted response from the retrieved context
    response_parts = []
    
    # Add intro
    response_parts.append("Based on TechCorp Solutions' HR policies:\n")
    
    # Parse the context to extract key information
    lines = context.split('\n')
    relevant_lines = []
    in_table = False
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('[') or line.startswith('---'):
            continue
        if line.startswith('##'):
            # Skip section headers in output
            continue
        if line.startswith('|'):
            in_table = True
            relevant_lines.append(line)
        elif in_table and not line.startswith('|'):
            in_table = False
            relevant_lines.append("")
            if line.startswith('-') or line.startswith('*') or line[0].isdigit():
                relevant_lines.append(line)
        elif line.startswith('-') or line.startswith('*') or (len(line) > 0 and line[0].isdigit()):
            relevant_lines.append(line)
        elif '**' in line:
            relevant_lines.append(line)
    
    # Join relevant content
    if relevant_lines:
        content = '\n'.join(relevant_lines[:20])  # Limit to first 20 relevant lines
        response_parts.append(content)
    else:
        # Fallback: use first 800 chars of context
        clean_context = context.replace('[TechCorp Solutions', '\n**').replace(']', '**')
        response_parts.append(clean_context[:800])
    
    # Add helpful footer
    response_parts.append("\n\nFor more details, visit the HR Portal at hr.techcorp-solutions.com or contact hr@techcorp-solutions.com.")
    
    return '\n'.join(response_parts)


@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    load_rag_engine()
    load_model()


@app.get("/")
async def root():
    company = rag_engine.get_company_name() if rag_engine and rag_engine.is_initialized else "Unknown"
    return {
        "status": "ok",
        "message": "HR FAQ API is running",
        "company": company,
        "rag_enabled": rag_engine is not None and rag_engine.is_initialized,
        "model_loaded": hr_module is not None
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": hr_module is not None,
        "rag_initialized": rag_engine is not None and rag_engine.is_initialized,
        "company": rag_engine.get_company_name() if rag_engine else None
    }


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """Answer HR-related questions with RAG support"""
    start_time = time.time()
    
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    mode = request.mode or "policy"
    ood_reject = False
    confidence = None
    sources_list = []
    context = ""
    company_name = "TechCorp Solutions"
    
    # Get RAG context if available
    if rag_engine and rag_engine.is_initialized:
        company_name = rag_engine.get_company_name()
        context, raw_sources = rag_engine.get_context_for_question(question, mode)
        
        # Convert sources to response format
        for src in raw_sources:
            sources_list.append(Source(
                title=src.get("title", ""),
                snippet=src.get("snippet", ""),
                category=src.get("category"),
                similarity=src.get("similarity")
            ))
    
    try:
        # Try to generate response with RAG context
        if context:
            answer = generate_rag_response(question, context, raw_sources if 'raw_sources' in dir() else [])
            
            if answer:
                confidence = 0.9 if sources_list else 0.7
            else:
                # Fallback with context
                answer, ood_reject = get_fallback_response(question, mode, context)
                confidence = 0.75
        elif hr_module is not None:
            # No RAG context, use DSPy model directly
            result = hr_module.forward(question)
            answer = result.answer if hasattr(result, 'answer') else str(result)
            
            if "outside" in answer.lower() or "scope" in answer.lower():
                ood_reject = True
                confidence = 0.9
            else:
                confidence = 0.7
        else:
            # Pure fallback
            answer, ood_reject = get_fallback_response(question, mode, "")
            confidence = 0.5 if not ood_reject else 0.95
    
    except Exception as e:
        print(f"Error processing question: {e}")
        answer, ood_reject = get_fallback_response(question, mode, context)
        confidence = 0.4
    
    latency_ms = int((time.time() - start_time) * 1000)
    
    return AskResponse(
        answer=answer,
        ood_reject=ood_reject,
        confidence=confidence,
        latency_ms=latency_ms,
        sources=sources_list if sources_list else None,
        company=company_name
    )


@app.post("/ask/stream")
async def ask_stream(request: AskRequest):
    """Stream HR answers with SSE for typing effect"""
    
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    mode = request.mode or "policy"
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        start_time = time.time()
        ood_reject = False
        confidence = 0.8
        sources_list = []
        context = ""
        company_name = "TechCorp Solutions"
        
        # Check OOD FIRST before doing any RAG lookup
        if not is_hr_related(question):
            ood_reject = True
            answer = "I'm sorry, but this question is outside my expertise as an HR assistant for TechCorp Solutions. I can only help with HR-related topics such as:\n\n• Leave & PTO (vacation, sick leave, holidays)\n• Benefits (health insurance, 401k, wellness)\n• Payroll (salary, bonuses, expenses)\n• Remote work policies\n• Training & onboarding\n• Company policies\n\nPlease ask an HR-related question!"
        else:
            # Get RAG context only for HR questions
            if rag_engine and rag_engine.is_initialized:
                company_name = rag_engine.get_company_name()
                context, raw_sources = rag_engine.get_context_for_question(question, mode)
                
                for src in raw_sources:
                    sources_list.append({
                        "title": src.get("title", ""),
                        "snippet": src.get("snippet", ""),
                        "category": src.get("category"),
                        "similarity": src.get("similarity")
                    })
            
            # Generate answer
            if context:
                answer = generate_rag_response(question, context, sources_list)
                if not answer:
                    answer, ood_reject = get_fallback_response(question, mode, context)
            else:
                answer, ood_reject = get_fallback_response(question, mode, "")
        
        # Stream the answer character by character with small delays
        words = answer.split(' ')
        streamed_text = ""
        
        for i, word in enumerate(words):
            if i > 0:
                streamed_text += " "
            streamed_text += word
            
            # Send chunk
            chunk_data = {
                "type": "chunk",
                "content": word + (" " if i < len(words) - 1 else ""),
                "done": False
            }
            yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # Small delay for typing effect (faster for tables)
            if word.startswith('|'):
                await asyncio.sleep(0.01)
            else:
                await asyncio.sleep(0.03)
        
        # Send final message with metadata
        latency_ms = int((time.time() - start_time) * 1000)
        final_data = {
            "type": "done",
            "content": "",
            "done": True,
            "ood_reject": ood_reject,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "sources": sources_list,
            "company": company_name
        }
        yield f"data: {json.dumps(final_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/company")
async def get_company_info():
    """Get company information"""
    if rag_engine and rag_engine.is_initialized:
        return rag_engine.company_info
    return {"company_name": "TechCorp Solutions", "error": "RAG not initialized"}


@app.post("/rebuild-index")
async def rebuild_index():
    """Rebuild the RAG index (admin endpoint)"""
    global rag_engine
    
    try:
        from backend.rag_engine import get_rag_engine
        rag_engine = get_rag_engine(force_init=True)
        
        if rag_engine.is_initialized:
            return {"status": "success", "message": "Index rebuilt successfully"}
        else:
            return {"status": "partial", "message": "RAG engine initialized but dependencies may be missing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to rebuild index: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
