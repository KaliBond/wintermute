import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SectionEyebrow } from '../components/SectionEyebrow';

gsap.registerPlugin(ScrollTrigger);

const BAR_SEGMENTS = [
  { label: 'C_i', value: 2.1, color: '#5A7A6A' },
  { label: 'K_i', value: 3.4, color: '#C8943A' },
  { label: '-S_i', value: -1.8, color: '#8A4A4A' },
  { label: '0.5A_i', value: 1.5, color: '#7A6A8A' },
];

export function ThermodynamicBaselineSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const leftRef = useRef<HTMLDivElement>(null);
  const barRef = useRef<HTMLDivElement>(null);

  const [hoveredSeg, setHoveredSeg] = useState<number | null>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const left = leftRef.current;
    const bar = barRef.current;
    if (!section || !left || !bar) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const lines = left.querySelectorAll('.formula-line');
    const segments = bar.querySelectorAll('.bar-segment');

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: section,
        start: 'top 70%',
        toggleActions: 'play none none none',
      },
    });

    tl.fromTo(lines, { x: -30, opacity: 0 }, { x: 0, opacity: 1, duration: 1, stagger: 0.2, ease: 'power3.out' })
      .fromTo(segments, { scaleY: 0 }, {
        scaleY: 1,
        duration: 0.8,
        stagger: 0.25,
        ease: 'power2.out',
      }, 0.3);

    return () => { tl.kill(); };
  }, []);

  const totalPositive = BAR_SEGMENTS.filter(s => s.value > 0).reduce((a, b) => a + b.value, 0);
  const maxHeight = 280;

  return (
    <section ref={sectionRef} className="bg-charcoal section-padding">
      <div className="content-max-width">
        <div className="flex flex-col lg:flex-row gap-16 lg:gap-20">
          {/* Left Column */}
          <div className="lg:w-1/2" ref={leftRef}>
            <div className="formula-line">
              <SectionEyebrow text="03 / The Thermodynamic Baseline" color="text-sage" className="mb-4 block" />
            </div>
            <h2 className="formula-line font-serif text-parchment text-[clamp(1.8rem,3vw,2.5rem)] tracking-[-0.03em]">
              Canonical Node Value
            </h2>
            <div className="formula-line mt-8">
              <p className="font-mono text-parchment text-[clamp(1.4rem,2.5vw,2.2rem)] font-light">
                <var>V</var><sub><var>i</var></sub>(<var>t</var>) ={' '}
                <span className="bg-sage/10 px-1 rounded"><var>C</var><sub><var>i</var></sub></span> +{' '}
                <span className="bg-amber/10 px-1 rounded"><var>K</var><sub><var>i</var></sub></span> -{' '}
                <span className="bg-crimson/10 px-1 rounded"><var>S</var><sub><var>i</var></sub></span> +{' '}
                <span className="bg-plum/10 px-1 rounded">0.5<var>A</var><sub><var>i</var></sub></span>
              </p>
            </div>
            <div className="formula-line mt-6">
              <p className="font-mono text-steel text-[1.2rem] font-light">
                <var>V</var><sub><var>i</var></sub> = 0.530
              </p>
            </div>
            <p className="formula-line font-sans font-light text-steel text-[1rem] max-w-[480px] mt-8 leading-[1.7]">
              Measures the absolute institutional health and potential energy of the node. It is additive, linear, and calibrated for aggregate structural resilience. Capacity plus coherence, minus stress, plus half the abstraction premium.
            </p>
          </div>

          {/* Right Column - Animated Bar */}
          <div className="lg:w-1/2 flex items-center justify-center" ref={barRef}>
            <div className="relative flex items-end gap-16">
              {/* Grid lines */}
              <div className="absolute inset-0 flex flex-col justify-between pointer-events-none" style={{ height: maxHeight }}>
                {[0, 1, 2, 3, 4].map(i => (
                  <div key={i} className="w-full h-[1px] bg-divider" />
                ))}
              </div>

              {/* Stacked bar */}
              <div className="relative flex flex-col items-center" style={{ height: maxHeight }}>
                {/* Positive segments */}
                <div className="flex flex-col-reverse w-20 origin-bottom">
                  {BAR_SEGMENTS.map((seg, i) => {
                    const h = Math.abs(seg.value) / totalPositive * (maxHeight * 0.7);
                    return (
                      <div
                        key={seg.label}
                        className="bar-segment w-full relative cursor-pointer origin-bottom transition-opacity duration-200"
                        style={{
                          height: `${h}px`,
                          backgroundColor: seg.color,
                          opacity: hoveredSeg !== null && hoveredSeg !== i ? 0.5 : 1,
                        }}
                        onMouseEnter={() => setHoveredSeg(i)}
                        onMouseLeave={() => setHoveredSeg(null)}
                      >
                        {hoveredSeg === i && (
                          <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-charcoal-surface px-2 py-1 rounded text-[0.7rem] font-mono text-parchment whitespace-nowrap border border-divider z-10">
                            {seg.label} = {seg.value}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>

                {/* V_i marker */}
                <div className="absolute w-28 h-[2px] bg-parchment left-full ml-2" style={{ bottom: `${(totalPositive - 1.8 + 1.5) / totalPositive * (maxHeight * 0.7)}px` }}>
                  <span className="absolute left-full ml-2 font-mono text-parchment text-[0.8rem] -translate-y-1/2">
                    <var>V</var><sub><var>i</var></sub>
                  </span>
                </div>

                {/* Labels on the left */}
                <div className="absolute right-full mr-4 flex flex-col justify-around h-full">
                  {BAR_SEGMENTS.map((seg) => (
                    <span key={seg.label} className="font-mono text-[0.75rem] text-right" style={{ color: seg.color }}>
                      {seg.label}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
