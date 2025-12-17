'use client';

import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';
import { useEffect, useRef, useState, useCallback, ReactNode, forwardRef, ButtonHTMLAttributes } from 'react';
import { cn } from '@/lib/utils';

// ============================================================================
// BUTTON (shadcn-style)
// ============================================================================

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'default' | 'secondary' | 'outline' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'default', size = 'default', ...props }, ref) => {
    const variants = {
      default: 'bg-blue-600 text-white hover:bg-blue-500 shadow-lg shadow-blue-500/25',
      secondary: 'bg-white/10 text-white hover:bg-white/20 border border-white/10',
      outline: 'border border-white/20 bg-transparent hover:bg-white/5 text-white',
      ghost: 'hover:bg-white/10 text-zinc-400 hover:text-white',
      link: 'text-blue-400 underline-offset-4 hover:underline',
    };
    const sizes = {
      default: 'h-11 px-6 py-2',
      sm: 'h-9 px-4 text-sm',
      lg: 'h-12 px-8 text-lg',
      icon: 'h-10 w-10',
    };
    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center gap-2 rounded-xl font-medium transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-black disabled:pointer-events-none disabled:opacity-50',
          variants[variant],
          sizes[size],
          className
        )}
        {...props}
      />
    );
  }
);
Button.displayName = 'Button';

// ============================================================================
// BADGE (shadcn-style)
// ============================================================================

interface BadgeProps {
  children: ReactNode;
  variant?: 'default' | 'secondary' | 'outline' | 'success' | 'warning';
  className?: string;
}

export function Badge({ children, variant = 'default', className }: BadgeProps) {
  const variants = {
    default: 'bg-blue-500/10 text-blue-400 border-blue-500/20',
    secondary: 'bg-white/5 text-zinc-300 border-white/10',
    outline: 'bg-transparent text-zinc-400 border-white/20',
    success: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    warning: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  };
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium border transition-colors',
        variants[variant],
        className
      )}
    >
      {children}
    </span>
  );
}

// ============================================================================
// CARD (shadcn-style)
// ============================================================================

interface CardProps {
  children: ReactNode;
  className?: string;
  hover?: boolean;
}

export function Card({ children, className, hover = false }: CardProps) {
  return (
    <div
      className={cn(
        'rounded-2xl bg-white/[0.02] border border-white/10 backdrop-blur-sm',
        hover && 'transition-all duration-300 hover:bg-white/[0.04] hover:border-white/20 hover:-translate-y-1 hover:shadow-xl hover:shadow-blue-500/5',
        className
      )}
    >
      {children}
    </div>
  );
}

export function CardHeader({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('p-6 pb-4', className)}>{children}</div>;
}

export function CardContent({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('p-6 pt-0', className)}>{children}</div>;
}

export function CardTitle({ children, className }: { children: ReactNode; className?: string }) {
  return <h3 className={cn('text-lg font-semibold text-white', className)}>{children}</h3>;
}

export function CardDescription({ children, className }: { children: ReactNode; className?: string }) {
  return <p className={cn('text-sm text-zinc-400 mt-1.5', className)}>{children}</p>;
}

// ============================================================================
// SEPARATOR
// ============================================================================

export function Separator({ className, orientation = 'horizontal' }: { className?: string; orientation?: 'horizontal' | 'vertical' }) {
  return (
    <div
      className={cn(
        'bg-white/10',
        orientation === 'horizontal' ? 'h-px w-full' : 'h-full w-px',
        className
      )}
    />
  );
}

// ============================================================================
// TOOLTIP
// ============================================================================

interface TooltipProps {
  children: ReactNode;
  content: string;
}

export function Tooltip({ children, content }: TooltipProps) {
  return (
    <div className="relative group">
      {children}
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-1.5 bg-zinc-900 border border-white/10 rounded-lg text-xs text-white opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-50">
        {content}
        <div className="absolute top-full left-1/2 -translate-x-1/2 border-4 border-transparent border-t-zinc-900" />
      </div>
    </div>
  );
}

// ============================================================================
// ACCORDION (shadcn-style)
// ============================================================================

interface AccordionItemProps {
  title: string;
  children: ReactNode;
  defaultOpen?: boolean;
  icon?: ReactNode;
}

export function AccordionItem({ title, children, defaultOpen = false, icon }: AccordionItemProps) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border-b border-white/10 last:border-b-0">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full py-5 flex items-center justify-between text-left group"
        aria-expanded={isOpen}
      >
        <div className="flex items-center gap-3">
          {icon && <span className="text-blue-400">{icon}</span>}
          <span className="font-medium text-white group-hover:text-blue-400 transition-colors">{title}</span>
        </div>
        <motion.div
          animate={{ rotate: isOpen ? 180 : 0 }}
          transition={{ duration: 0.2 }}
          className="text-zinc-500 group-hover:text-zinc-300 transition-colors"
        >
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M4 6L8 10L12 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        </motion.div>
      </button>
      <motion.div
        initial={false}
        animate={{ 
          height: isOpen ? 'auto' : 0,
          opacity: isOpen ? 1 : 0 
        }}
        transition={{ duration: 0.3, ease: 'easeInOut' }}
        className="overflow-hidden"
      >
        <div className="pb-5 text-sm text-zinc-400 leading-relaxed pr-8">
          {children}
        </div>
      </motion.div>
    </div>
  );
}

export function Accordion({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('divide-y divide-white/10', className)}>{children}</div>;
}

// ============================================================================
// BACKGROUND BEAMS (Aceternity-style)
// ============================================================================

export function BackgroundBeams() {
  return (
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      {/* Gradient overlay */}
      <div className="absolute inset-0 bg-gradient-to-b from-blue-950/30 via-transparent to-violet-950/30" />
      
      {/* Animated beams */}
      <svg className="absolute inset-0 w-full h-full" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <linearGradient id="beam-gradient-1" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="rgba(59, 130, 246, 0)" />
            <stop offset="50%" stopColor="rgba(59, 130, 246, 0.4)" />
            <stop offset="100%" stopColor="rgba(59, 130, 246, 0)" />
          </linearGradient>
          <linearGradient id="beam-gradient-2" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="rgba(139, 92, 246, 0)" />
            <stop offset="50%" stopColor="rgba(139, 92, 246, 0.4)" />
            <stop offset="100%" stopColor="rgba(139, 92, 246, 0)" />
          </linearGradient>
          <linearGradient id="beam-gradient-3" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="rgba(236, 72, 153, 0)" />
            <stop offset="50%" stopColor="rgba(236, 72, 153, 0.3)" />
            <stop offset="100%" stopColor="rgba(236, 72, 153, 0)" />
          </linearGradient>
        </defs>
        {[...Array(8)].map((_, i) => (
          <motion.line
            key={i}
            x1={`${5 + i * 12}%`}
            y1="-10%"
            x2={`${25 + i * 12}%`}
            y2="110%"
            stroke={`url(#beam-gradient-${(i % 3) + 1})`}
            strokeWidth="1"
            initial={{ pathLength: 0, opacity: 0 }}
            animate={{ 
              pathLength: [0, 1, 0],
              opacity: [0, 0.6, 0]
            }}
            transition={{
              duration: 3 + i * 0.3,
              repeat: Infinity,
              delay: i * 0.5,
              ease: 'easeInOut'
            }}
          />
        ))}
      </svg>

      {/* Floating orbs */}
      <motion.div
        className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-500/10 rounded-full blur-3xl"
        animate={{
          x: [0, 50, 0],
          y: [0, 30, 0],
          scale: [1, 1.1, 1],
        }}
        transition={{ duration: 8, repeat: Infinity, ease: 'easeInOut' }}
      />
      <motion.div
        className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-violet-500/10 rounded-full blur-3xl"
        animate={{
          x: [0, -40, 0],
          y: [0, -20, 0],
          scale: [1, 1.15, 1],
        }}
        transition={{ duration: 10, repeat: Infinity, ease: 'easeInOut' }}
      />
    </div>
  );
}

// ============================================================================
// SPOTLIGHT (Aceternity-style)
// ============================================================================

export function Spotlight() {
  const [position, setPosition] = useState({ x: 50, y: 50 });

  const handleMouseMove = useCallback((e: MouseEvent) => {
    setPosition({
      x: (e.clientX / window.innerWidth) * 100,
      y: (e.clientY / window.innerHeight) * 100,
    });
  }, []);

  useEffect(() => {
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, [handleMouseMove]);

  return (
    <div
      className="absolute inset-0 pointer-events-none transition-all duration-300"
      style={{
        background: `radial-gradient(800px circle at ${position.x}% ${position.y}%, rgba(59, 130, 246, 0.08), transparent 40%)`
      }}
    />
  );
}

// ============================================================================
// GRID PATTERN
// ============================================================================

export function GridPattern() {
  return (
    <div 
      className="absolute inset-0 pointer-events-none"
      style={{
        backgroundImage: `
          linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)
        `,
        backgroundSize: '64px 64px',
        maskImage: 'radial-gradient(ellipse 80% 50% at 50% 0%, black 70%, transparent 100%)'
      }}
    />
  );
}

// ============================================================================
// BENTO GRID (Aceternity-style)
// ============================================================================

interface BentoCardProps {
  title: string;
  description: string;
  icon: ReactNode;
  className?: string;
  gradient?: string;
  size?: 'default' | 'large';
}

export function BentoCard({ title, description, icon, className = '', gradient = 'from-blue-500/10 to-violet-500/10', size = 'default' }: BentoCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-50px' }}
      whileHover={{ y: -6, transition: { duration: 0.2 } }}
      className={cn(
        'group relative p-6 rounded-2xl bg-white/[0.02] border border-white/10 hover:border-white/20 transition-all duration-300 overflow-hidden',
        size === 'large' && 'md:col-span-2',
        className
      )}
    >
      {/* Hover gradient */}
      <div className={cn('absolute inset-0 bg-gradient-to-br opacity-0 group-hover:opacity-100 transition-opacity duration-500', gradient)} />
      
      {/* Shine effect */}
      <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500">
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/5 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-1000" />
      </div>

      <div className="relative z-10">
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500/20 to-violet-500/20 border border-white/10 flex items-center justify-center mb-4 group-hover:scale-110 group-hover:border-white/20 transition-all duration-300">
          {icon}
        </div>
        <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-blue-400 transition-colors">{title}</h3>
        <p className="text-sm text-zinc-400 leading-relaxed">{description}</p>
      </div>
    </motion.div>
  );
}

export function BentoGrid({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn('grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4', className)}>
      {children}
    </div>
  );
}

// ============================================================================
// TEXT GENERATE EFFECT (Aceternity-style)
// ============================================================================

export function TextGenerateEffect({ words, className = '' }: { words: string; className?: string }) {
  const [displayedText, setDisplayedText] = useState('');
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isComplete, setIsComplete] = useState(false);

  useEffect(() => {
    if (currentIndex < words.length) {
      const timeout = setTimeout(() => {
        setDisplayedText(words.slice(0, currentIndex + 1));
        setCurrentIndex(currentIndex + 1);
      }, 40);
      return () => clearTimeout(timeout);
    } else {
      setIsComplete(true);
    }
  }, [currentIndex, words]);

  return (
    <span className={className}>
      {displayedText}
      {!isComplete && (
        <motion.span 
          className="inline-block w-[3px] h-[1em] bg-blue-400 ml-1 align-middle"
          animate={{ opacity: [1, 0] }}
          transition={{ duration: 0.5, repeat: Infinity }}
        />
      )}
    </span>
  );
}

// ============================================================================
// HOVER BORDER GRADIENT (Aceternity-style)
// ============================================================================

interface HoverBorderGradientProps {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
  as?: 'button' | 'a';
  href?: string;
}

export function HoverBorderGradient({ children, className = '', onClick, as = 'button', href }: HoverBorderGradientProps) {
  const Component = as;
  
  return (
    <Component
      onClick={onClick}
      href={href}
      className={cn(
        'relative group px-6 py-3 rounded-xl bg-black overflow-hidden transition-all duration-300',
        className
      )}
    >
      {/* Animated gradient border */}
      <div className="absolute inset-0 rounded-xl p-[1px] bg-gradient-to-r from-blue-500 via-violet-500 to-fuchsia-500 opacity-50 group-hover:opacity-100 transition-opacity">
        <div className="absolute inset-[1px] bg-black rounded-xl" />
      </div>
      
      {/* Rotating gradient (on hover) */}
      <motion.div 
        className="absolute inset-[-100%] bg-gradient-conic from-blue-500 via-violet-500 to-blue-500 opacity-0 group-hover:opacity-100 transition-opacity"
        animate={{ rotate: 360 }}
        transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
      />
      <div className="absolute inset-[1px] bg-black rounded-xl" />
      
      <span className="relative z-10 flex items-center justify-center gap-2 font-medium text-white">
        {children}
      </span>
    </Component>
  );
}

// ============================================================================
// ANIMATED BADGE
// ============================================================================

export function AnimatedBadge({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9, y: 10 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      className={cn(
        'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-white/5 border border-white/10 text-zinc-300 backdrop-blur-sm',
        className
      )}
    >
      {children}
    </motion.div>
  );
}

// ============================================================================
// SECTION WRAPPER
// ============================================================================

interface SectionProps {
  children: ReactNode;
  className?: string;
  id?: string;
}

export function Section({ children, className = '', id }: SectionProps) {
  return (
    <motion.section
      id={id}
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className={cn('py-24', className)}
    >
      {children}
    </motion.section>
  );
}

// ============================================================================
// SECTION HEADER
// ============================================================================

interface SectionHeaderProps {
  badge?: string;
  title: string;
  description?: string;
  className?: string;
}

export function SectionHeader({ badge, title, description, className }: SectionHeaderProps) {
  return (
    <div className={cn('text-center mb-16', className)}>
      {badge && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mb-4"
        >
          <Badge variant="default">{badge}</Badge>
        </motion.div>
      )}
      <motion.h2
        initial={{ opacity: 0, y: 10 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ delay: 0.1 }}
        className="text-3xl md:text-4xl font-bold text-white mb-4"
      >
        {title}
      </motion.h2>
      {description && (
        <motion.p
          initial={{ opacity: 0, y: 10 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ delay: 0.2 }}
          className="text-zinc-400 max-w-2xl mx-auto text-lg"
        >
          {description}
        </motion.p>
      )}
    </div>
  );
}

// ============================================================================
// STEP CARD
// ============================================================================

interface StepCardProps {
  step: number;
  title: string;
  description: string;
  icon: ReactNode;
}

export function StepCard({ step, title, description, icon }: StepCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay: step * 0.15 }}
      className="relative"
    >
      {/* Connector line */}
      {step < 3 && (
        <div className="hidden md:block absolute top-10 left-[60%] w-full h-px bg-gradient-to-r from-white/20 to-transparent" />
      )}
      
      <div className="flex flex-col items-center text-center p-6">
        {/* Step number */}
        <div className="absolute -top-3 left-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-violet-500 flex items-center justify-center text-sm font-bold text-white shadow-lg shadow-blue-500/25">
          {step}
        </div>
        
        {/* Icon */}
        <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-white/5 to-white/[0.02] border border-white/10 flex items-center justify-center mb-6 mt-4">
          {icon}
        </div>
        
        <h3 className="text-xl font-semibold text-white mb-3">{title}</h3>
        <p className="text-sm text-zinc-400 leading-relaxed max-w-xs">{description}</p>
      </div>
    </motion.div>
  );
}

// ============================================================================
// FLOATING CARD (for demo preview)
// ============================================================================

export function FloatingCard({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      className={cn('relative', className)}
    >
      {/* Glow effect */}
      <div className="absolute -inset-4 bg-gradient-to-r from-blue-500/20 via-violet-500/20 to-fuchsia-500/20 rounded-3xl blur-2xl opacity-50" />
      
      {/* Card */}
      <div className="relative rounded-2xl bg-gradient-to-b from-white/[0.08] to-white/[0.02] border border-white/10 p-1 shadow-2xl">
        <div className="bg-black/90 rounded-xl overflow-hidden backdrop-blur-xl">
          {children}
        </div>
      </div>
    </motion.div>
  );
}

// ============================================================================
// STAT CARD
// ============================================================================

interface StatCardProps {
  value: string;
  label: string;
  icon: ReactNode;
  delay?: number;
}

export function StatCard({ value, label, icon, delay = 0 }: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay }}
      className="text-center"
    >
      <div className="flex justify-center mb-3">
        <div className="w-10 h-10 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center">
          {icon}
        </div>
      </div>
      <div className="text-3xl font-bold text-white mb-1">{value}</div>
      <div className="text-sm text-zinc-500">{label}</div>
    </motion.div>
  );
}
