import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SectionEyebrow } from '../components/SectionEyebrow';

gsap.registerPlugin(ScrollTrigger);

export function ActivationIndexSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const formulaRef = useRef<HTMLDivElement>(null);
  const leftBracketRef = useRef<SVGPathElement>(null);
  const rightBracketRef = useRef<SVGPathElement>(null);
  const labelsRef = useRef<HTMLDivElement>(null);
  const descRef = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const formula = formulaRef.current;
    const leftBracket = leftBracketRef.current;
    const rightBracket = rightBracketRef.current;
    const labels = labelsRef.current;
    const desc = descRef.current;
    if (!section || !formula || !leftBracket || !rightBracket || !labels || !desc) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    // Set initial stroke state
    gsap.set([leftBracket, rightBracket], { strokeDasharray: 200, strokeDashoffset: 200 });

    const labelEls = labels.querySelectorAll('.bracket-label');

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: section,
        start: 'top 75%',
        toggleActions: 'play none none none',
      },
    });

    tl.fromTo(formula, { opacity: 0, scale: 0.95 }, { opacity: 1, scale: 1, duration: 1.2, ease: 'power3.out' })
      .to([leftBracket, rightBracket], { strokeDashoffset: 0, duration: 1.5, ease: 'power2.out' }, 0.5)
      .fromTo(labelEls, { opacity: 0 }, { opacity: 1, duration: 0.6, stagger: 0.2 }, 1.2)
      .fromTo(desc, { y: 20, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, ease: 'power3.out' }, 1.5);

    return () => { tl.kill(); };
  }, []);

  return (
    <section ref={sectionRef} className="bg-charcoal section-padding">
      <div className="content-max-width">
        <div className="text-center">
          <SectionEyebrow text="05 / Trajectory Morphology" color="text-amber" className="mb-8 block" />

          <div ref={formulaRef} className="relative inline-block">
            <p className="font-mono text-parchment text-[clamp(1.8rem,4vw,3.5rem)] font-light">
              <var>{'\u03c3'}</var><sub><var>i</var></sub> = (<var>A</var><sub><var>i</var></sub> × <var>C</var><sub><var>i</var></sub>) × (<var>K</var><sub><var>i</var></sub> - <var>S</var><sub><var>i</var></sub>)
            </p>

            {/* Bracket SVGs */}
            <svg className="absolute -bottom-20 left-0 w-full h-20 pointer-events-none" viewBox="0 0 600 80" fill="none">
              {/* Left bracket under (A_i x C_i) */}
              <path
                ref={leftBracketRef}
                d="M 80 5 Q 80 40 80 65 Q 80 75 100 75 L 240 75 Q 260 75 260 65"
                stroke="#5A7A6A"
                strokeWidth="1.5"
                fill="none"
              />
              {/* Right bracket under (K_i - S_i) */}
              <path
                ref={rightBracketRef}
                d="M 340 65 Q 340 75 360 75 L 500 75 Q 520 75 520 65 Q 520 40 520 5"
                stroke="#C8943A"
                strokeWidth="1.5"
                fill="none"
              />
            </svg>
          </div>

          <div ref={labelsRef} className="mt-24 flex flex-col md:flex-row justify-center gap-12 md:gap-24">
            <div className="bracket-label text-left max-w-[200px]">
              <p className="font-sans font-light text-sage text-[0.8rem] leading-[1.6]">
                <strong className="font-medium">Symbolic Integration:</strong> How coherently the node processes narrative and abstraction.
              </p>
            </div>
            <div className="bracket-label text-left max-w-[200px]">
              <p className="font-sans font-light text-amber text-[0.8rem] leading-[1.6]">
                <strong className="font-medium">Energetic Surplus:</strong> The material margin of error.
              </p>
            </div>
          </div>

          <p ref={descRef} className="font-sans font-light text-steel text-[1rem] max-w-[600px] mx-auto mt-12 leading-[1.7]">
            Highly non-linear. A high-cognition node will massively amplify a small affective deficit (K &lt; S), generating a deeply negative <var>{'\u03c3'}</var><sub><var>i</var></sub>. This is cognitive inversion — the point where symbolic capacity becomes destructive.
          </p>
        </div>
      </div>
    </section>
  );
}
