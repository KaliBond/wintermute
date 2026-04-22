import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SectionEyebrow } from '../components/SectionEyebrow';

gsap.registerPlugin(ScrollTrigger);

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  color: string;
  side: 'left' | 'right';
}

export function DivergenceFieldSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const leftRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const left = leftRef.current;
    if (!section || !left) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const lines = left.querySelectorAll('.dvg-line');

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: section,
        start: 'top 70%',
        toggleActions: 'play none none none',
      },
    });

    tl.fromTo(lines, { x: -20, opacity: 0 }, { x: 0, opacity: 1, duration: 0.8, stagger: 0.15, ease: 'power3.out' });

    return () => { tl.kill(); };
  }, []);

  // Canvas particle system
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    let animId: number;
    const particles: Particle[] = [];
    let time = 0;

    const resize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (rect) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }
    };
    resize();
    window.addEventListener('resize', resize);

    const emitParticle = () => {
      const side = Math.random() > 0.5 ? 'left' : 'right';
      particles.push({
        x: canvas.width / 2,
        y: canvas.height / 2,
        vx: side === 'left' ? -0.3 - Math.random() * 0.5 : 0.3 + Math.random() * 0.5,
        vy: (Math.random() - 0.5) * 0.3,
        life: 0,
        maxLife: 150 + Math.random() * 100,
        color: side === 'left' ? '#5A7A6A' : '#8A4A4A',
        side,
      });
    };

    const draw = () => {
      if (!ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw center emission point
      ctx.beginPath();
      ctx.arc(canvas.width / 2, canvas.height / 2, 3, 0, Math.PI * 2);
      ctx.fillStyle = '#E8E4DF';
      ctx.fill();

      // Draw faint dividing line
      ctx.beginPath();
      ctx.moveTo(canvas.width / 2, 20);
      ctx.lineTo(canvas.width / 2, canvas.height - 20);
      ctx.strokeStyle = '#3A4148';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Labels
      ctx.font = '300 11px "JetBrains Mono"';
      ctx.fillStyle = '#5A7A6A';
      ctx.textAlign = 'right';
      ctx.fillText('From V_i', canvas.width / 2 - 10, 30);
      ctx.fillStyle = '#8A4A4A';
      ctx.textAlign = 'left';
      ctx.fillText('From ' + '\u03c3' + '_i', canvas.width / 2 + 10, 30);

      for (let i = particles.length - 1; i >= 0; i--) {
        const p = particles[i];
        p.life++;

        // Divergence force
        const divergenceFactor = p.life / p.maxLife;
        p.vx += (p.side === 'left' ? -0.008 : 0.008) * divergenceFactor;
        p.vy += Math.sin(time * 0.02 + p.life * 0.05) * 0.003;

        p.x += p.vx;
        p.y += p.vy;

        const alpha = 1 - p.life / p.maxLife;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 2 * alpha + 0.5, 0, Math.PI * 2);
        ctx.fillStyle = p.color;
        ctx.globalAlpha = alpha * 0.8;
        ctx.fill();
        ctx.globalAlpha = 1;

        // Trail
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x - p.vx * 4, p.y - p.vy * 4);
        ctx.strokeStyle = p.color;
        ctx.globalAlpha = alpha * 0.3;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.globalAlpha = 1;

        if (p.life >= p.maxLife) {
          particles.splice(i, 1);
        }
      }

      if (!prefersReducedMotion && particles.length < 120) {
        emitParticle();
        if (Math.random() > 0.5) emitParticle();
      }

      time++;
      animId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <section ref={sectionRef} className="bg-charcoal section-padding">
      <div className="content-max-width">
        <div className="flex flex-col lg:flex-row gap-16 lg:gap-20">
          {/* Left Column */}
          <div className="lg:w-[45%]" ref={leftRef}>
            <div className="dvg-line">
              <SectionEyebrow text="07 / Pathological Decoupling" color="text-crimson" className="mb-4 block" />
            </div>
            <h2 className="dvg-line font-serif text-parchment text-[clamp(1.8rem,3vw,2.5rem)] tracking-[-0.03em]">
              The Divergence Field
            </h2>
            <div className="dvg-line mt-8">
              <p className="font-mono text-parchment text-[clamp(1.2rem,2vw,1.8rem)] font-light">
                <var>{'\u0394'}</var><sub><var>i</var></sub> = log(|<var>{'\u03c3'}</var><sub><var>i</var></sub>| + <var>{'\u03b5'}</var>) - <var>V̄</var><sub><var>i</var></sub>
              </p>
            </div>
            <div className="dvg-line mt-4">
              <p className="font-mono text-steel text-[0.8rem] font-light">
                <var>{'\u03b5'}</var> = 0.1 regularizes zero-crossings.
              </p>
            </div>
            <p className="dvg-line font-sans font-light text-steel text-[1rem] max-w-[440px] mt-6 leading-[1.7]">
              A large positive divergence indicates an entity mathematically divorced from its own material substrate — surviving purely on systemic momentum. This diagnoses pathological decoupling before it becomes visible.
            </p>
          </div>

          {/* Right Column - Canvas */}
          <div className="lg:w-[55%]">
            <div className="relative w-full h-[400px] bg-charcoal-surface rounded-sm overflow-hidden">
              <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
            </div>
            <div className="flex justify-between mt-4">
              <span className="font-sans font-light text-[0.7rem] text-sage">{'\u2192'} cohesive material trajectory</span>
              <span className="font-sans font-light text-[0.7rem] text-crimson">divergent symbolic trajectory {'\u2192'}</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
