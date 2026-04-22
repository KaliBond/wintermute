import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SectionEyebrow } from '../components/SectionEyebrow';

gsap.registerPlugin(ScrollTrigger);

const PATHOLOGIES = [
  {
    title: 'Healthy Differentiated',
    condition: 'V_i > 0, \u03c3_i > 0',
    description: 'Capacity exceeds stress; normal activation. The system is thermodynamically viable, with coherent symbolic and material coupling.',
    color: '#5A7A6A',
  },
  {
    title: 'Symbolic Persistence (Sisu)',
    condition: 'V_i \u2193, mod \u03c3_i, \u0394_i >> 0',
    description: 'Material substrate collapsing, but high A \u00d7 C hallucinates stability. The symbolic mind continues to function even as the thermodynamic body fails.',
    color: '#C8943A',
  },
  {
    title: 'Cognitive Inversion',
    condition: 'mod V_i, \u03c3_i << 0, \u0394_i << 0',
    description: 'High abstraction weaponized against a material deficit. Destructive activation. A high-cognition node massively amplifies a small affective deficit.',
    color: '#8A4A4A',
  },
  {
    title: 'Thermodynamic Freeze',
    condition: 'V_i < 0, \u03c3_i \u2248 0, \u0394_i \u2248 0',
    description: 'Total collapse of material capacity and memory. Reset. The system has lost both energetic potential and symbolic coordination.',
    color: '#8A9499',
  },
  {
    title: 'Executive Decoupling',
    condition: 'V_Helm < 6',
    description: 'Impending failure of the state\'s central steering mechanism. The helm node can no longer coordinate the system\'s slow loop.',
    color: '#7A6A8A',
  },
];

export function DiagnosticPathologySection() {
  const sectionRef = useRef<HTMLElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const header = headerRef.current;
    const grid = gridRef.current;
    if (!section || !header || !grid) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const cards = grid.querySelectorAll('.pathology-card');

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: section,
        start: 'top 70%',
        toggleActions: 'play none none none',
      },
    });

    tl.fromTo(header, { y: 30, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, ease: 'power3.out' })
      .fromTo(cards, { y: 40, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, stagger: 0.1, ease: 'power3.out' }, 0.2);

    return () => { tl.kill(); };
  }, []);

  return (
    <section id="diagnostics" ref={sectionRef} className="bg-charcoal-light section-padding">
      <div className="content-max-width">
        <div ref={headerRef}>
          <SectionEyebrow text="04 / Systemic States" color="text-crimson" className="mb-4 block" />
          <h2 className="font-serif text-parchment text-[clamp(1.8rem,3.5vw,2.8rem)] tracking-[-0.03em]">
            The Diagnostic Pathology Matrix
          </h2>
          <p className="font-sans font-light text-steel text-[1rem] max-w-[640px] mt-4 leading-[1.7]">
            Five fundamental states of societal thermodynamic health. Each diagnosis reveals a distinct coupling relationship between material capacity and symbolic integration.
          </p>
        </div>

        <div ref={gridRef} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6 mt-16">
          {PATHOLOGIES.map((p) => (
            <div
              key={p.title}
              className="pathology-card bg-charcoal-surface p-8"
              style={{ borderTop: `3px solid ${p.color}` }}
            >
              <h3 className="font-sans font-medium text-parchment text-[1.1rem]">
                {p.title}
              </h3>
              <p className="font-mono text-[0.9rem] mt-3 font-light" style={{ color: p.color }}>
                {p.condition}
              </p>
              <p className="font-sans font-light text-steel text-[0.85rem] leading-[1.7] mt-4">
                {p.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
