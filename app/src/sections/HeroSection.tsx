import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ConstellationCanvas } from '../components/ConstellationCanvas';

export function HeroSection() {
  const eyebrowRef = useRef<HTMLSpanElement>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const subtitleRef = useRef<HTMLParagraphElement>(null);
  const ctaRef = useRef<HTMLAnchorElement>(null);
  const hintRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });

    tl.fromTo(eyebrowRef.current, { opacity: 0 }, { opacity: 1, duration: 0.8, delay: 0.3 })
      .fromTo(headingRef.current, { opacity: 0, y: 30 }, { opacity: 1, y: 0, duration: 1 }, 0.5)
      .fromTo(subtitleRef.current, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.8 }, 0.8)
      .fromTo(ctaRef.current, { opacity: 0 }, { opacity: 1, duration: 0.6 }, 1.1)
      .fromTo(hintRef.current, { opacity: 0 }, { opacity: 1, duration: 0.6 }, 1.4);

    return () => { tl.kill(); };
  }, []);

  const handleExplore = (e: React.MouseEvent) => {
    e.preventDefault();
    document.querySelector('#manifesto')?.scrollIntoView({ behavior: 'smooth' });
  };

  return (
    <section className="relative w-full min-h-[100dvh] flex items-center justify-center overflow-hidden">
      <ConstellationCanvas />

      <div className="relative z-10 flex flex-col items-center text-center px-6 max-w-[700px]">
        <span
          ref={eyebrowRef}
          className="accent-label text-steel mb-6 opacity-0"
        >
          CAMS v3.2-R-ESCH
        </span>

        <h1
          ref={headingRef}
          className="font-serif text-parchment text-[clamp(2.5rem,6vw,5rem)] tracking-[-0.03em] leading-[1.1] opacity-0"
          style={{ textShadow: '0 2px 30px rgba(26,29,33,0.9)' }}
        >
          Macroscopic Societal Thermodynamics
        </h1>

        <p
          ref={subtitleRef}
          className="font-sans font-light text-steel text-[clamp(0.9rem,1.5vw,1.15rem)] mt-6 max-w-[560px] leading-[1.7] opacity-0"
          style={{ textShadow: '0 2px 20px rgba(26,29,33,0.8)' }}
        >
          A scale-invariant mathematical framework for reading societies as living coordination systems.
        </p>

        <a
          ref={ctaRef}
          href="#manifesto"
          onClick={handleExplore}
          className="mt-10 font-sans font-medium text-[0.75rem] uppercase tracking-[0.12em] text-sage border border-sage px-8 py-3 rounded-[2px] hover:bg-sage hover:text-charcoal transition-all duration-300 opacity-0"
        >
          Explore the Framework
        </a>

        <div ref={hintRef} className="mt-16 flex flex-col items-center gap-3 opacity-0">
          <span className="font-sans font-light text-[0.6rem] text-steel uppercase tracking-[0.15em]">
            Scroll to begin
          </span>
          <div className="w-10 h-[1px] bg-divider relative">
            <svg
              className="absolute left-1/2 -translate-x-1/2 -bottom-3 w-3 h-3 text-steel animate-bounce-hint"
              fill="none"
              viewBox="0 0 12 12"
              stroke="currentColor"
              strokeWidth="1.5"
            >
              <path d="M2 4L6 8L10 4" />
            </svg>
          </div>
        </div>
      </div>
    </section>
  );
}
