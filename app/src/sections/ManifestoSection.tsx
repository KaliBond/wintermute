import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SectionEyebrow } from '../components/SectionEyebrow';

gsap.registerPlugin(ScrollTrigger);

export function ManifestoSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const headingLinesRef = useRef<HTMLDivElement>(null);
  const rightColRef = useRef<HTMLDivElement>(null);
  const dividerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const headingLines = headingLinesRef.current;
    const rightCol = rightColRef.current;
    const divider = dividerRef.current;
    if (!section || !headingLines || !rightCol || !divider) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const lines = headingLines.querySelectorAll('.heading-line');

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: section,
        start: 'top 80%',
        toggleActions: 'play none none none',
      },
    });

    tl.fromTo(lines, { y: 60, opacity: 0 }, { y: 0, opacity: 1, duration: 1, stagger: 0.12, ease: 'power3.out' })
      .fromTo(rightCol, { y: 40, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, ease: 'power3.out' }, 0.3)
      .fromTo(divider, { scaleX: 0 }, { scaleX: 1, duration: 0.6, ease: 'power2.out' }, 0.5);

    return () => { tl.kill(); };
  }, []);

  return (
    <section
      id="manifesto"
      ref={sectionRef}
      className="bg-charcoal section-padding"
    >
      <div className="content-max-width">
        <div className="flex flex-col lg:flex-row gap-12 lg:gap-20">
          {/* Left Column - 55% */}
          <div className="lg:w-[55%]">
            <SectionEyebrow text="01 / The Premise" color="text-sage" className="mb-6 block" />
            <div ref={headingLinesRef}>
              <span className="heading-line block font-serif text-parchment text-[clamp(2rem,4vw,3.5rem)] tracking-[-0.03em] leading-[1.1]">
                Societies are not
              </span>
              <span className="heading-line block font-serif text-parchment text-[clamp(2rem,4vw,3.5rem)] tracking-[-0.03em] leading-[1.1]">
                built by ideology.
              </span>
              <span className="heading-line block font-serif text-parchment text-[clamp(2rem,4vw,3.5rem)] tracking-[-0.03em] leading-[1.1] mt-2">
                They are held together by thermodynamic coupling.
              </span>
            </div>
          </div>

          {/* Right Column - 45% */}
          <div className="lg:w-[45%] lg:pt-12" ref={rightColRef}>
            <p className="font-sans font-light text-steel text-[1rem] leading-[1.8]">
              Societies maintain viability when the fast loop — material throughput, craft, hands, shield — and the slow loop — symbolic coordination, archive, lore, helm — remain energetically coupled. Crisis is not a failure of leadership. It is temporal desynchronization.
            </p>
            <div ref={dividerRef} className="w-[60px] h-[1px] bg-divider my-8 origin-left" />
            <p className="font-sans font-light text-steel text-[1rem] leading-[1.8]">
              The mathematical trajectory — not descriptive ideology — is the universal empirical truth of complex systems. From the Western Roman Empire to SpaceX, the same thermodynamic signatures recur at every scale.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
