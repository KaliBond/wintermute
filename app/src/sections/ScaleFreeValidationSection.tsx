import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SectionEyebrow } from '../components/SectionEyebrow';

gsap.registerPlugin(ScrollTrigger);

export function ScaleFreeValidationSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);
  const leftCardRef = useRef<HTMLDivElement>(null);
  const rightCardRef = useRef<HTMLDivElement>(null);
  const bannerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const header = headerRef.current;
    const leftCard = leftCardRef.current;
    const rightCard = rightCardRef.current;
    const banner = bannerRef.current;
    if (!section || !header || !leftCard || !rightCard || !banner) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: section,
        start: 'top 75%',
        toggleActions: 'play none none none',
      },
    });

    tl.fromTo(header, { y: 30, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, ease: 'power3.out' })
      .fromTo(leftCard, { x: -40, opacity: 0 }, { x: 0, opacity: 1, duration: 1, ease: 'power3.out' }, 0.2)
      .fromTo(rightCard, { x: 40, opacity: 0 }, { x: 0, opacity: 1, duration: 1, ease: 'power3.out' }, 0.2)
      .fromTo(banner, { y: 20, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, ease: 'power3.out' }, 0.8);

    return () => { tl.kill(); };
  }, []);

  return (
    <section ref={sectionRef} className="bg-charcoal-light section-padding">
      <div className="content-max-width">
        <div ref={headerRef}>
          <SectionEyebrow text="08 / Empirical Validation" color="text-amber" className="mb-4 block" />
          <h2 className="font-serif text-parchment text-[clamp(1.8rem,3.5vw,2.8rem)] tracking-[-0.03em]">
            Scale-Free Validation: The Operator Translates Reality
          </h2>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mt-16">
          {/* Card 1 - SpaceX */}
          <div
            ref={leftCardRef}
            className="bg-charcoal-surface p-12"
            style={{ borderTop: '3px solid #5A7A6A' }}
          >
            <span className="font-sans font-medium text-[0.75rem] uppercase tracking-[0.1em] text-sage">
              Entity 1: SpaceX (2005)
            </span>
            <p className="font-sans font-light text-parchment text-[0.9rem] mt-4">
              Data: Lore
            </p>
            <p className="font-mono text-sage text-[clamp(1.5rem,3vw,2.2rem)] font-light mt-4">
              <var>{'\u03c3'}</var> = 128
            </p>
            <p className="font-mono text-steel text-[0.8rem] font-light mt-3">
              All material nodes (Flow, Hands, Shield) &lt; 0
            </p>
            <p className="font-sans font-light text-parchment text-[0.9rem] mt-6 leading-[1.7]">
              <strong className="font-medium text-sage">Diagnosis:</strong> Lore-first activation. High symbolic coherence driving an entity before material capacity exists.
            </p>
          </div>

          {/* Card 2 - Roman Empire */}
          <div
            ref={rightCardRef}
            className="bg-charcoal-surface p-12"
            style={{ borderTop: '3px solid #8A4A4A' }}
          >
            <span className="font-sans font-medium text-[0.75rem] uppercase tracking-[0.1em] text-crimson">
              Entity 2: Western Roman Empire (CE 450)
            </span>
            <p className="font-sans font-light text-parchment text-[0.9rem] mt-4">
              Data: Terminal Lore Persistence.
            </p>
            <div className="font-mono text-parchment text-[0.9rem] font-light mt-4 space-y-1">
              <p>Lore <var>A</var> × <var>C</var> = 45;</p>
              <p>Energetic Margin (<var>K</var>-<var>S</var>) = -2;</p>
              <p>Yielding <var>{'\u03c3'}</var> = -90</p>
            </div>
            <p className="font-sans font-light text-parchment text-[0.9rem] mt-6 leading-[1.7]">
              <strong className="font-medium text-crimson">Diagnosis:</strong> The symbolic mind continues to hallucinate function even as the thermodynamic body fails.
            </p>
          </div>
        </div>

        {/* Conclusion Banner */}
        <div ref={bannerRef} className="mt-16 bg-charcoal p-10 text-center">
          <p className="font-serif italic text-parchment text-[clamp(1rem,1.8vw,1.3rem)] leading-[1.6]">
            The mathematical trajectory — not the descriptive ideology — is the universal empirical truth of complex systems.
          </p>
        </div>
      </div>
    </section>
  );
}
