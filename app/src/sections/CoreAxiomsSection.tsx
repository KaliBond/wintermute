import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SectionEyebrow } from '../components/SectionEyebrow';

gsap.registerPlugin(ScrollTrigger);

const AXIOMS = [
  {
    variable: 'C_i',
    title: 'Coherence',
    description: 'Internal alignment and integration. How well a node\'s components coordinate with each other. High coherence means the node\'s subsystems are synchronized and reinforcing.',
    color: '#5A7A6A',
  },
  {
    variable: 'K_i',
    title: 'Capacity',
    description: 'Material and functional efficacy. The node\'s ability to perform work, produce output, and maintain physical infrastructure. Capacity is the energetic potential of the system.',
    color: '#C8943A',
  },
  {
    variable: 'S_i',
    title: 'Stress',
    description: 'Affective and resource load. The entropy burden on the system — the cost of maintaining order against dissipation. Rising stress signals increasing thermodynamic pressure.',
    color: '#8A4A4A',
  },
  {
    variable: 'A_i',
    title: 'Abstraction',
    description: 'Symbolic and narrative decoupling. The distance between a node\'s self-representation and its material reality. High abstraction enables coordination at scale but risks cognitive inversion.',
    color: '#7A6A8A',
  },
];

export function CoreAxiomsSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const headingRef = useRef<HTMLDivElement>(null);
  const gridRef = useRef<HTMLDivElement>(null);
  const bannerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const heading = headingRef.current;
    const grid = gridRef.current;
    const banner = bannerRef.current;
    if (!section || !heading || !grid || !banner) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const cards = grid.querySelectorAll('.axiom-card');
    const borders = grid.querySelectorAll('.axiom-border');

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: section,
        start: 'top 75%',
        toggleActions: 'play none none none',
      },
    });

    tl.fromTo(heading, { y: 40, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, ease: 'power3.out' })
      .fromTo(cards, { y: 50, opacity: 0 }, { y: 0, opacity: 1, duration: 0.9, stagger: 0.15, ease: 'power3.out' }, 0.2)
      .fromTo(borders, { scaleY: 0 }, { scaleY: 1, duration: 0.6, stagger: 0.15, ease: 'power2.out' }, 0.5)
      .fromTo(banner, { y: 30, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, ease: 'power3.out' }, 0.8);

    return () => { tl.kill(); };
  }, []);

  return (
    <section ref={sectionRef} className="bg-charcoal-light section-padding">
      <div className="content-max-width">
        <div ref={headingRef}>
          <SectionEyebrow text="02 / State Vector" color="text-amber" className="mb-4 block" />
          <h2 className="font-serif text-parchment text-[clamp(1.8rem,3.5vw,2.8rem)] tracking-[-0.03em]">
            The Four Dimensions of Societal State
          </h2>
        </div>

        <div ref={gridRef} className="grid grid-cols-1 md:grid-cols-2 gap-8 mt-16">
          {AXIOMS.map((axiom) => (
            <div key={axiom.variable} className="axiom-card relative bg-charcoal-surface p-10 overflow-hidden">
              <div
                className="axiom-border absolute left-0 top-0 bottom-0 w-[3px] origin-top"
                style={{ backgroundColor: axiom.color }}
              />
              <span className="font-mono text-[1.5rem] font-light" style={{ color: axiom.color }}>
                {axiom.variable}
              </span>
              <h3 className="font-sans font-medium text-parchment text-[1rem] mt-3">
                {axiom.title}
              </h3>
              <p className="font-sans font-light text-steel text-[0.85rem] leading-[1.7] mt-3">
                {axiom.description}
              </p>
            </div>
          ))}
        </div>

        {/* Formula Banner */}
        <div ref={bannerRef} className="mt-16 bg-charcoal p-10 text-center">
          <p className="font-mono text-parchment text-[clamp(1.2rem,2.5vw,2rem)] font-light">
            <var>m</var><sub><var>i</var></sub>(<var>t</var>) = (<var>C</var><sub><var>i</var></sub>, <var>K</var><sub><var>i</var></sub>, <var>S</var><sub><var>i</var></sub>, <var>A</var><sub><var>i</var></sub>) ∈ [0, 10]<sup>4</sup>
          </p>
          <p className="font-sans font-light text-steel text-[0.85rem] mt-4">
            For each node <var>i</var> ∈ <var>N</var> at discrete time <var>t</var>, this vector captures the raw operational state before structural amplification.
          </p>
        </div>
      </div>
    </section>
  );
}
