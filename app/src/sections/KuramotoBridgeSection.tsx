import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SectionEyebrow } from '../components/SectionEyebrow';

gsap.registerPlugin(ScrollTrigger);

export function KuramotoBridgeSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const formulaRef = useRef<HTMLDivElement>(null);
  const dividerRef = useRef<HTMLDivElement>(null);
  const labelsRef = useRef<HTMLDivElement>(null);
  const descRef = useRef<HTMLParagraphElement>(null);
  const theoremRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const formula = formulaRef.current;
    const divider = dividerRef.current;
    const labels = labelsRef.current;
    const desc = descRef.current;
    const theorem = theoremRef.current;
    if (!section || !formula || !divider || !labels || !desc || !theorem) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const labelEls = labels.querySelectorAll('.kura-label');

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: section,
        start: 'top 75%',
        toggleActions: 'play none none none',
      },
    });

    tl.fromTo(formula, { opacity: 0, scale: 0.9 }, { opacity: 1, scale: 1, duration: 1.2, ease: 'power3.out' })
      .fromTo(divider, { scaleX: 0 }, { scaleX: 1, duration: 0.6, ease: 'power2.out' }, 0.5)
      .fromTo(labelEls, { opacity: 0 }, { opacity: 1, duration: 0.5, stagger: 0.15 }, 0.8)
      .fromTo(desc, { y: 20, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, ease: 'power3.out' }, 1)
      .fromTo(theorem, { y: 20, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, ease: 'power3.out' }, 1.2);

    return () => { tl.kill(); };
  }, []);

  return (
    <section ref={sectionRef} className="bg-charcoal-light section-padding">
      <div className="content-max-width">
        <div className="text-center">
          <SectionEyebrow text="10 / Criticality" color="text-plum" className="mb-4 block" />
          <h2 className="font-serif text-parchment text-[clamp(1.8rem,3vw,2.5rem)] tracking-[-0.03em]">
            The Kuramoto Criticality Bridge
          </h2>

          <div ref={formulaRef} className="mt-10 inline-flex flex-col items-center">
            <p className="font-mono text-parchment text-[clamp(1.8rem,4vw,3.5rem)] font-light">
              <var>{'\u03ba'}</var>(<var>t</var>) ≈
            </p>

            {/* Fraction */}
            <div className="flex flex-col items-center mt-2">
              <span className="font-mono text-amber text-[clamp(1.6rem,3.5vw,3rem)] font-light">
                <var>{'\u03a9'}</var>(<var>t</var>)
              </span>
              <div ref={dividerRef} className="w-32 h-[2px] bg-parchment my-2 origin-center" />
              <span className="font-mono text-sage text-[clamp(1.6rem,3.5vw,3rem)] font-light">
                <var>{'\u039b'}</var>(<var>t</var>)
              </span>
            </div>
          </div>

          <div ref={labelsRef} className="mt-10 flex flex-col md:flex-row justify-center gap-12 md:gap-20">
            <div className="kura-label text-left max-w-[220px]">
              <p className="font-sans font-light text-amber text-[0.8rem] leading-[1.6]">
                <strong className="font-medium">Numerator:</strong> Rate dispersion across Fast/Slow loops.
              </p>
            </div>
            <div className="kura-label text-left max-w-[220px]">
              <p className="font-sans font-light text-sage text-[0.8rem] leading-[1.6]">
                <strong className="font-medium">Denominator:</strong> Coordination bond strength.
              </p>
            </div>
          </div>

          <p ref={descRef} className="font-sans font-light text-steel text-[1rem] max-w-[560px] mx-auto mt-8 leading-[1.7]">
            Equates the core of societal fracture to the disorder-to-coupling ratio of the Kuramoto model. When the dispersive force exceeds the binding force, the network structurally ruptures.
          </p>

          {/* Theorem Box */}
          <div ref={theoremRef} className="mt-12 bg-charcoal border-l-[3px] border-crimson px-10 py-8 max-w-[800px] mx-auto text-left">
            <p className="font-sans font-normal text-parchment text-[1rem] leading-[1.7]">
              <strong>Theorem:</strong> When <var>{'\u03a9'}</var>(<var>t</var>) exceeds <var>{'\u039b'}</var>(<var>t</var>), the network structurally ruptures in a non-agentic phase transition. Fracture is driven by measurable thermodynamic desynchronization, not individual leadership failures.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
