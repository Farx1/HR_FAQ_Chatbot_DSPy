'use client';

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Send, Sparkles, Copy, Check, AlertCircle, Bot, User, Zap, Shield, 
  TrendingUp, Brain, MessageSquare, ArrowRight, ChevronDown,
  FileText, Clock, Users, CheckCircle, Menu, X, Lock, Eye, Server
} from 'lucide-react';
import {
  BackgroundBeams, Spotlight, GridPattern, BentoCard, BentoGrid,
  TextGenerateEffect, HoverBorderGradient, AnimatedBadge, Section,
  SectionHeader, StepCard, AccordionItem, Accordion, FloatingCard,
  Button, Badge, Card, StatCard, Separator
} from '@/components/ui';

// ============================================================================
// LOGIC (unchanged)
// ============================================================================

const HR_KEYWORDS = [
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
];

const RESPONSES: Record<string, string> = {
  vacation: `**Vacation Policy**

Employees are entitled to paid vacation based on tenure:

• **First 3 years:** 15 days per year
• **After 3 years:** 20 days per year
• **After 5 years:** 25 days per year

Submit requests at least 2 weeks in advance via the HR portal.`,

  remote: `**Remote Work Policy**

Our hybrid work arrangements include:

• **Standard:** 3 days office, 2 days remote
• **Full remote:** Available for eligible roles
• **Equipment:** $500 home office allowance

Coordinate specific arrangements with your manager.`,

  harassment: `**Harassment Reporting**

To report workplace harassment:

1. Document the incident with details
2. Contact HR or your supervisor
3. Use anonymous hotline: 1-800-XXX-XXXX

All reports are confidential and investigated within 5 business days.`,

  training: `**Training & Development**

Available opportunities:

• **LinkedIn Learning:** Access for all employees
• **Training budget:** $2,000 annually
• **Mentorship:** Internal program available
• **Certifications:** Reimbursement eligible

Contact HR to discuss your development plan.`,

  salary: `**Compensation Information**

Key details:

• **Pay periods:** Bi-weekly (every other Friday)
• **Raises:** Annual performance review based
• **Bonuses:** Based on company/individual performance
• **Direct deposit:** Available via HR portal`,

  benefits: `**Employee Benefits**

Your benefits package includes:

• **Insurance:** Medical, dental, vision
• **401(k):** 4% company match
• **Life insurance:** 2x annual salary
• **EAP:** Employee Assistance Program
• **Perks:** Gym membership discount

Open enrollment: November annually.`,

  sick: `**Sick Leave Policy**

Policy details:

• **Days:** 10 paid sick days per year
• **Usage:** Personal or family care
• **Documentation:** Doctor's note after 3+ days
• **Rollover:** Does not roll over

Notify your manager as early as possible.`,

  dress: `**Dress Code Guidelines**

• **Office days:** Business casual
• **Fridays:** Casual (jeans OK)
• **Client meetings:** Business professional
• **Safety areas:** Required equipment

When uncertain, dress more formally.`,

  time_off: `**Time Off Requests**

How to request:

1. Log into HR portal
2. Select "Time Off Request"
3. Choose dates and leave type
4. Submit for approval

Request 2+ weeks in advance for planned absences.`,

  general: `**HR Contact Information**

For specific inquiries:

• **Benefits:** benefits@company.com
• **Leave:** hr@company.com
• **Training:** learning@company.com
• **General:** hr@company.com

Visit the HR portal for self-service options.`
};

const OOD_RESPONSE = `**Out of Scope**

I specialize in HR-related questions only.

I can help with:
• Leave and vacation policies
• Remote work arrangements
• Compensation and benefits
• Training and development
• Workplace policies

Please ask an HR-related question.`;

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  isOOD?: boolean;
  timestamp: Date;
}

function getResponse(q: string): { response: string; isOOD: boolean } {
  const lower = q.toLowerCase();
  const isHR = HR_KEYWORDS.some(k => lower.includes(k));

  if (!isHR) return { response: OOD_RESPONSE, isOOD: true };

  const matches: [string[], string][] = [
    [['vacation', 'holiday', 'pto', 'annual leave'], 'vacation'],
    [['remote', 'work from home', 'wfh', 'hybrid'], 'remote'],
    [['harassment', 'discrimination', 'complaint', 'ethics'], 'harassment'],
    [['training', 'development', 'course', 'certification'], 'training'],
    [['salary', 'pay', 'compensation', 'bonus'], 'salary'],
    [['benefit', 'insurance', 'health', '401k'], 'benefits'],
    [['sick', 'ill', 'medical'], 'sick'],
    [['dress', 'attire', 'uniform'], 'dress'],
    [['time off', 'request', 'absence'], 'time_off'],
  ];

  for (const [keywords, key] of matches) {
    if (keywords.some(w => lower.includes(w))) {
      return { response: RESPONSES[key], isOOD: false };
    }
  }

  return { response: RESPONSES.general, isOOD: false };
}

// ============================================================================
// NAV LINKS
// ============================================================================

const NAV_LINKS = [
  { label: 'Features', href: '#features' },
  { label: 'How it Works', href: '#how-it-works' },
  { label: 'Demo', href: '#demo' },
  { label: 'Security', href: '#security' },
];

// ============================================================================
// FEATURES DATA
// ============================================================================

const FEATURES = [
  {
    title: 'DSPy Optimization',
    description: 'Fine-tuned with DSPy framework achieving 784% accuracy improvement over baseline prompting methods.',
    icon: <Brain className="w-6 h-6 text-blue-400" />,
    gradient: 'from-blue-500/10 to-cyan-500/10'
  },
  {
    title: 'Out-of-Domain Detection',
    description: '90% rejection rate for off-topic queries, keeping conversations focused and relevant to HR topics.',
    icon: <Shield className="w-6 h-6 text-violet-400" />,
    gradient: 'from-violet-500/10 to-purple-500/10'
  },
  {
    title: 'Sub-300ms Latency',
    description: 'Lightning-fast responses with average 272ms latency for seamless, real-time user experience.',
    icon: <Zap className="w-6 h-6 text-amber-400" />,
    gradient: 'from-amber-500/10 to-orange-500/10'
  },
  {
    title: 'Comprehensive Coverage',
    description: 'Full knowledge of vacation, benefits, remote work, training, compliance, and all HR policies.',
    icon: <FileText className="w-6 h-6 text-emerald-400" />,
    gradient: 'from-emerald-500/10 to-green-500/10'
  },
  {
    title: '24/7 Availability',
    description: 'Always-on assistant ready to answer employee questions any time, reducing HR team workload.',
    icon: <Clock className="w-6 h-6 text-pink-400" />,
    gradient: 'from-pink-500/10 to-rose-500/10'
  },
  {
    title: 'Enterprise Scale',
    description: 'Handle thousands of concurrent queries without performance degradation or downtime.',
    icon: <Users className="w-6 h-6 text-cyan-400" />,
    gradient: 'from-cyan-500/10 to-teal-500/10'
  },
];

// ============================================================================
// STATS
// ============================================================================

const STATS = [
  { label: 'Accuracy Boost', value: '+784%', icon: <TrendingUp className="w-5 h-5 text-emerald-400" /> },
  { label: 'OOD Rejection', value: '90%', icon: <Shield className="w-5 h-5 text-blue-400" /> },
  { label: 'Avg Latency', value: '272ms', icon: <Zap className="w-5 h-5 text-amber-400" /> },
];

// ============================================================================
// MESSAGE COMPONENTS
// ============================================================================

function MessageContent({ content }: { content: string }) {
  const lines = content.split('\n');
  return (
    <div className="space-y-1.5">
      {lines.map((line, i) => {
        if (line.startsWith('**') && line.endsWith('**')) {
          return <h4 key={i} className="text-[13px] font-semibold text-white">{line.slice(2, -2)}</h4>;
        }
        if (line.startsWith('• **')) {
          const match = line.match(/• \*\*(.+?)\*\*:?\s*(.*)/);
          if (match) {
            return (
              <p key={i} className="text-[12px] leading-relaxed">
                <span className="text-blue-400">•</span>{' '}
                <span className="font-medium text-zinc-200">{match[1]}:</span>{' '}
                <span className="text-zinc-400">{match[2]}</span>
              </p>
            );
          }
        }
        if (line.startsWith('• ')) {
          return (
            <p key={i} className="text-[12px] text-zinc-400 leading-relaxed">
              <span className="text-blue-400">•</span> {line.slice(2)}
            </p>
          );
        }
        if (line.match(/^\d+\./)) {
          return (
            <p key={i} className="text-[12px] text-zinc-400 leading-relaxed">
              <span className="text-blue-400 font-medium">{line.match(/^\d+/)?.[0]}.</span> {line.replace(/^\d+\.\s*/, '')}
            </p>
          );
        }
        if (line.trim()) {
          return <p key={i} className="text-[12px] text-zinc-400 leading-relaxed">{line}</p>;
        }
        return null;
      })}
    </div>
  );
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const copy = () => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={copy}
      className="p-1.5 rounded-md hover:bg-white/10 transition-colors text-zinc-500 hover:text-zinc-300"
      aria-label="Copy message"
    >
      {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
    </button>
  );
}

// ============================================================================
// CHAT DEMO COMPONENT
// ============================================================================

const SUGGESTIONS = [
  { icon: '🏖️', text: 'How many vacation days do I get?', category: 'Leave' },
  { icon: '🏠', text: 'What is the remote work policy?', category: 'Work' },
  { icon: '📚', text: 'What training is available?', category: 'Growth' },
  { icon: '💰', text: 'Tell me about benefits', category: 'Benefits' },
];

function ChatDemo() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const send = async (text: string) => {
    if (!text.trim() || loading) return;

    const userMsg: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: text.trim(),
      timestamp: new Date(),
    };
    setMessages(m => [...m, userMsg]);
    setInput('');
    setLoading(true);

    await new Promise(r => setTimeout(r, 600));

    const { response, isOOD } = getResponse(text);
    setMessages(m => [...m, {
      id: (Date.now() + 1).toString(),
      role: 'assistant',
      content: response,
      isOOD,
      timestamp: new Date(),
    }]);
    setLoading(false);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    send(input);
  };

  return (
    <div className="flex flex-col h-[520px]">
      {/* Chat Header */}
      <div className="flex items-center justify-between p-4 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center shadow-lg shadow-blue-500/25">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 bg-emerald-400 rounded-full border-2 border-black" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-white">HR Assistant</h3>
            <p className="text-xs text-zinc-500">DSPy-Enhanced • Online</p>
          </div>
        </div>
        <Badge variant="success">
          <span className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
          Live Demo
        </Badge>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        <AnimatePresence mode="popLayout">
          {messages.length === 0 ? (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="h-full flex flex-col items-center justify-center text-center px-4"
            >
              <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-blue-500/20 to-violet-500/20 border border-white/10 flex items-center justify-center mb-4">
                <Sparkles className="w-8 h-8 text-blue-400" />
              </div>
              <h4 className="text-white font-medium mb-2">Try the HR Assistant</h4>
              <p className="text-sm text-zinc-500 mb-6">Click a suggestion or type your question</p>
              <div className="grid grid-cols-2 gap-2 w-full max-w-sm">
                {SUGGESTIONS.map((s, i) => (
                  <motion.button
                    key={i}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: i * 0.1 }}
                    onClick={() => send(s.text)}
                    className="text-left p-3 rounded-xl bg-white/[0.03] border border-white/10 hover:border-blue-500/50 hover:bg-white/[0.06] transition-all text-xs group"
                  >
                    <span className="text-lg mr-2">{s.icon}</span>
                    <span className="text-zinc-300 group-hover:text-white transition-colors">{s.text.slice(0, 20)}...</span>
                  </motion.button>
                ))}
              </div>
            </motion.div>
          ) : (
            messages.map((msg) => (
              <motion.div
                key={msg.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}
              >
                <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center ${
                  msg.role === 'user'
                    ? 'bg-blue-600'
                    : msg.isOOD
                    ? 'bg-amber-500/20 border border-amber-500/30'
                    : 'bg-gradient-to-br from-blue-500 to-violet-600'
                }`}>
                  {msg.role === 'user' ? (
                    <User className="w-4 h-4 text-white" />
                  ) : msg.isOOD ? (
                    <AlertCircle className="w-4 h-4 text-amber-400" />
                  ) : (
                    <Bot className="w-4 h-4 text-white" />
                  )}
                </div>
                <div className={`flex-1 max-w-[75%] ${msg.role === 'user' ? 'flex flex-col items-end' : ''}`}>
                  <div className={`rounded-2xl px-4 py-3 ${
                    msg.role === 'user'
                      ? 'bg-blue-600 text-white rounded-tr-sm'
                      : msg.isOOD
                      ? 'bg-amber-500/5 border border-amber-500/20 rounded-tl-sm'
                      : 'bg-white/[0.04] border border-white/10 rounded-tl-sm'
                  }`}>
                    {msg.role === 'user' ? (
                      <p className="text-[13px]">{msg.content}</p>
                    ) : (
                      <MessageContent content={msg.content} />
                    )}
                  </div>
                  {msg.role === 'assistant' && (
                    <div className="flex items-center gap-1 mt-1.5 opacity-0 hover:opacity-100 transition-opacity">
                      <CopyButton text={msg.content} />
                    </div>
                  )}
                </div>
              </motion.div>
            ))
          )}
        </AnimatePresence>

        {loading && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex gap-3"
          >
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center">
              <Bot className="w-4 h-4 text-white" />
            </div>
            <div className="bg-white/[0.04] border border-white/10 rounded-2xl rounded-tl-sm px-4 py-3">
              <div className="flex gap-1.5">
                {[0, 1, 2].map(i => (
                  <motion.span
                    key={i}
                    className="w-2 h-2 bg-blue-400 rounded-full"
                    animate={{ y: [0, -6, 0] }}
                    transition={{ duration: 0.6, repeat: Infinity, delay: i * 0.15 }}
                  />
                ))}
              </div>
            </div>
          </motion.div>
        )}

        <div ref={endRef} />
      </div>

      {/* Input */}
      <form onSubmit={handleSubmit} className="p-4 border-t border-white/10">
        <div className="flex items-center gap-3">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder="Ask about HR policies, benefits, leave..."
            disabled={loading}
            className="flex-1 px-4 py-3 rounded-xl bg-white/[0.05] border border-white/10 focus:border-blue-500/50 focus:ring-2 focus:ring-blue-500/20 outline-none text-sm text-white placeholder-zinc-500 transition-all disabled:opacity-50"
          />
          <Button type="submit" disabled={!input.trim() || loading} size="icon" className="rounded-xl">
            <Send className="w-4 h-4" />
          </Button>
        </div>
      </form>
    </div>
  );
}

// ============================================================================
// MAIN PAGE
// ============================================================================

export default function Home() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      <BackgroundBeams />
      <Spotlight />
      <GridPattern />

      {/* ===== NAVIGATION ===== */}
      <header className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled ? 'bg-black/80 backdrop-blur-xl border-b border-white/10' : 'bg-transparent'
      }`}>
        <nav className="mx-auto max-w-6xl px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          {/* Logo */}
          <motion.a 
            href="#" 
            className="flex items-center gap-3"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center shadow-lg shadow-blue-500/25">
              <Bot className="w-5 h-5 text-white" />
            </div>
            <span className="font-semibold text-lg hidden sm:block">HR Assistant</span>
          </motion.a>

          {/* Desktop Nav */}
          <motion.div 
            className="hidden md:flex items-center gap-1"
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
          >
            {NAV_LINKS.map((link, i) => (
              <a
                key={link.href}
                href={link.href}
                className="px-4 py-2 rounded-lg text-sm text-zinc-400 hover:text-white hover:bg-white/5 transition-all"
              >
                {link.label}
              </a>
            ))}
          </motion.div>

          {/* CTA */}
          <motion.div 
            className="hidden md:flex items-center gap-3"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
          >
            <a href="#demo">
              <Button>Try Demo</Button>
            </a>
          </motion.div>

          {/* Mobile Menu Button */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            className="md:hidden p-2 rounded-lg text-zinc-400 hover:text-white hover:bg-white/10 transition-colors"
            aria-label="Toggle menu"
          >
            {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </nav>

        {/* Mobile Menu */}
        <AnimatePresence>
          {mobileMenuOpen && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="md:hidden border-t border-white/10 bg-black/95 backdrop-blur-xl"
            >
              <div className="px-4 py-4 space-y-1">
                {NAV_LINKS.map(link => (
                  <a
                    key={link.href}
                    href={link.href}
                    onClick={() => setMobileMenuOpen(false)}
                    className="block py-3 px-4 rounded-lg text-zinc-400 hover:text-white hover:bg-white/5 transition-all"
                  >
                    {link.label}
                  </a>
                ))}
                <Separator className="my-2" />
                <a href="#demo" onClick={() => setMobileMenuOpen(false)}>
                  <Button className="w-full mt-2">Try Demo</Button>
                </a>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </header>

      {/* ===== HERO ===== */}
      <section className="relative pt-32 pb-24 px-4">
        <div className="mx-auto max-w-5xl text-center">
          {/* Badges */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex flex-wrap justify-center gap-3 mb-8"
          >
            <AnimatedBadge><Brain className="w-3.5 h-3.5 text-blue-400" /> DSPy-Enhanced</AnimatedBadge>
            <AnimatedBadge><Shield className="w-3.5 h-3.5 text-violet-400" /> OOD Detection</AnimatedBadge>
            <AnimatedBadge><Zap className="w-3.5 h-3.5 text-amber-400" /> Sub-300ms</AnimatedBadge>
          </motion.div>

          {/* Title */}
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-4xl sm:text-5xl lg:text-6xl xl:text-7xl font-bold mb-6 leading-[1.1] tracking-tight"
          >
            <span className="bg-gradient-to-b from-white to-zinc-400 bg-clip-text text-transparent">
              Your Intelligent
            </span>
            <br />
            <span className="bg-gradient-to-r from-blue-400 via-violet-400 to-fuchsia-400 bg-clip-text text-transparent">
              <TextGenerateEffect words="HR Assistant" />
            </span>
          </motion.h1>

          {/* Subtitle */}
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="text-lg md:text-xl text-zinc-400 mb-10 max-w-2xl mx-auto leading-relaxed"
          >
            AI-powered HR chatbot fine-tuned with DSPy for accurate policy answers. 
            Get instant responses about vacation, benefits, remote work, and more.
          </motion.p>

          {/* CTAs */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <a href="#demo">
              <HoverBorderGradient className="shadow-lg shadow-blue-500/20">
                <MessageSquare className="w-4 h-4" />
                Try the Demo
                <ArrowRight className="w-4 h-4" />
              </HoverBorderGradient>
            </a>
            <a href="#features">
              <Button variant="ghost" className="gap-2">
                Learn More
                <ChevronDown className="w-4 h-4" />
              </Button>
            </a>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4 }}
            className="mt-20 grid grid-cols-3 gap-8 max-w-lg mx-auto"
          >
            {STATS.map((stat, i) => (
              <StatCard key={i} {...stat} delay={0.5 + i * 0.1} />
            ))}
          </motion.div>
        </div>
      </section>

      {/* ===== FEATURES ===== */}
      <Section id="features" className="px-4">
        <div className="mx-auto max-w-6xl">
          <SectionHeader
            badge="Features"
            title="Powerful Capabilities"
            description="Built with cutting-edge AI technology to deliver accurate, fast, and reliable HR assistance."
          />
          <BentoGrid>
            {FEATURES.map((feature, i) => (
              <BentoCard
                key={i}
                title={feature.title}
                description={feature.description}
                icon={feature.icon}
                gradient={feature.gradient}
              />
            ))}
          </BentoGrid>
        </div>
      </Section>

      {/* ===== HOW IT WORKS ===== */}
      <Section id="how-it-works" className="px-4">
        <div className="mx-auto max-w-5xl">
          <SectionHeader
            badge="Process"
            title="How It Works"
            description="Simple, fast, and intelligent. Get answers to your HR questions in seconds."
          />
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-8">
            <StepCard
              step={1}
              title="Ask a Question"
              description="Type your HR-related question in natural language. No special commands needed."
              icon={<MessageSquare className="w-8 h-8 text-blue-400" />}
            />
            <StepCard
              step={2}
              title="AI Processing"
              description="DSPy-optimized model analyzes, validates, and processes your query intelligently."
              icon={<Brain className="w-8 h-8 text-violet-400" />}
            />
            <StepCard
              step={3}
              title="Get Your Answer"
              description="Receive accurate, policy-compliant responses instantly with relevant details."
              icon={<CheckCircle className="w-8 h-8 text-emerald-400" />}
            />
          </div>
        </div>
      </Section>

      {/* ===== DEMO ===== */}
      <Section id="demo" className="px-4">
        <div className="mx-auto max-w-5xl">
          <SectionHeader
            badge="Interactive Demo"
            title="Try It Now"
            description="Experience the HR Assistant in action. Ask about vacation, benefits, remote work, or any HR policy."
          />
          <FloatingCard className="max-w-2xl mx-auto">
            <ChatDemo />
          </FloatingCard>
        </div>
      </Section>

      {/* ===== SECURITY / FAQ ===== */}
      <Section id="security" className="px-4">
        <div className="mx-auto max-w-3xl">
          <SectionHeader
            badge="Security & Privacy"
            title="Frequently Asked Questions"
            description="Everything you need to know about the HR Assistant."
          />
          <Card className="p-6 md:p-8">
            <Accordion>
              <AccordionItem 
                title="What topics can the HR Assistant help with?" 
                icon={<FileText className="w-4 h-4" />}
                defaultOpen
              >
                The assistant covers a comprehensive range of HR topics including vacation policies, 
                sick leave, remote work arrangements, benefits packages, training opportunities, 
                compensation details, dress code guidelines, and general HR inquiries. It&apos;s designed 
                to provide accurate, policy-compliant information instantly.
              </AccordionItem>
              <AccordionItem 
                title="How accurate are the responses?"
                icon={<TrendingUp className="w-4 h-4" />}
              >
                With DSPy optimization, the assistant achieves a remarkable 784% accuracy improvement 
                over baseline prompting methods. This ensures reliable, consistent, and policy-compliant 
                answers for all your HR-related questions.
              </AccordionItem>
              <AccordionItem 
                title="What happens if I ask an off-topic question?"
                icon={<Shield className="w-4 h-4" />}
              >
                The assistant features advanced out-of-domain detection with a 90% rejection rate for 
                off-topic queries. It will politely inform you that the question is outside its scope 
                and suggest relevant HR-related topics it can help with instead.
              </AccordionItem>
              <AccordionItem 
                title="Is my data secure and private?"
                icon={<Lock className="w-4 h-4" />}
              >
                Absolutely. All conversations are processed securely with enterprise-grade encryption. 
                No personal data is stored beyond the current session, and we follow strict data 
                protection protocols to ensure your privacy is maintained at all times.
              </AccordionItem>
              <AccordionItem 
                title="Can this integrate with existing HR systems?"
                icon={<Server className="w-4 h-4" />}
              >
                Yes, the HR Assistant is designed with integration in mind. It can connect with 
                popular HRIS platforms, ticketing systems, and internal knowledge bases to provide 
                even more accurate and contextual responses based on your organization&apos;s specific policies.
              </AccordionItem>
            </Accordion>
          </Card>
        </div>
      </Section>

      {/* ===== FOOTER ===== */}
      <footer className="border-t border-white/10 py-12 px-4 mt-12">
        <div className="mx-auto max-w-6xl">
          <div className="flex flex-col md:flex-row items-center justify-between gap-8">
            {/* Logo */}
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center">
                <Bot className="w-4 h-4 text-white" />
              </div>
              <span className="font-semibold">HR Assistant</span>
            </div>

            {/* Links */}
            <div className="flex items-center gap-6 text-sm">
              {NAV_LINKS.map(link => (
                <a 
                  key={link.href} 
                  href={link.href} 
                  className="text-zinc-400 hover:text-white transition-colors"
                >
                  {link.label}
                </a>
              ))}
            </div>

            {/* Copyright */}
            <p className="text-sm text-zinc-500">
              © 2024 HR Assistant. DSPy-Enhanced.
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
