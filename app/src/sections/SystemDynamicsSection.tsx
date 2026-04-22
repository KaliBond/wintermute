import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SectionEyebrow } from '../components/SectionEyebrow';

gsap.registerPlugin(ScrollTrigger);

const NETWORK_NODES = [
  { x: 150, y: 60 }, { x: 240, y: 30 }, { x: 300, y: 90 },
  { x: 270, y: 160 }, { x: 180, y: 170 }, { x: 90, y: 120 },
];

const NETWORK_EDGES = [
  [0, 1], [0, 2], [0, 3], [0, 4], [0, 5],
  [1, 2], [1, 3], [2, 3], [2, 4], [3, 4], [3, 5], [4, 5],
];

export function SystemDynamicsSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const headerRef = useRef<HTMLDivElement>(null);
  const leftPanelRef = useRef<HTMLDivElement>(null);
  const rightPanelRef = useRef<HTMLDivElement>(null);
  const dividerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const header = headerRef.current;
    const leftPanel = leftPanelRef.current;
    const rightPanel = rightPanelRef.current;
    const divider = dividerRef.current;
    if (!section || !header || !leftPanel || !rightPanel || !divider) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const nodes = leftPanel.querySelectorAll('.net-node');
    const edges = leftPanel.querySelectorAll('.net-edge');
    const arrows = rightPanel.querySelectorAll('.disp-arrow');

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: section,
        start: 'top 70%',
        toggleActions: 'play none none none',
      },
    });

    tl.fromTo(header, { y: 30, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, ease: 'power3.out' })
      .fromTo(nodes, { scale: 0 }, { scale: 1, duration: 0.5, stagger: 0.08, ease: 'back.out(1.7)' }, 0.3)
      .fromTo(edges, { opacity: 0 }, { opacity: 1, duration: 0.4 }, 0.8)
      .fromTo(divider, { scaleY: 0 }, { scaleY: 1, duration: 0.8, ease: 'power2.out' }, 0.2)
      .fromTo(arrows, { x: 0, opacity: 0 }, { x: (i: number) => (i % 2 === 0 ? 20 : -20), opacity: 1, duration: 0.6, stagger: 0.1, ease: 'power2.out' }, 0.4);

    return () => { tl.kill(); };
  }, []);

  return (
    <section ref={sectionRef} className="bg-charcoal section-padding">
      <div className="content-max-width">
        <div ref={headerRef}>
          <SectionEyebrow text="11 / Forces" color="text-amber" className="mb-4 block" />
          <h2 className="font-serif text-parchment text-[clamp(1.8rem,3.5vw,2.8rem)] tracking-[-0.03em]">
            Systemic Dynamics: Bond Strength vs. Shear Force
          </h2>
        </div>

        <div className="flex flex-col lg:flex-row mt-16 relative">
          {/* Vertical divider */}
          <div ref={dividerRef} className="hidden lg:block absolute left-1/2 top-0 bottom-0 w-[1px] bg-divider origin-top" />

          {/* Left Panel - Bond Strength */}
          <div ref={leftPanelRef} className="lg:w-1/2 lg:pr-12 pb-12 lg:pb-0">
            <p className="font-mono text-sage text-[1.4rem] font-light">
              <var>B</var><sub><var>ij</var></sub> = <var>f</var>(<var>V</var><sub><var>i</var></sub>, <var>V</var><sub><var>j</var></sub>)
            </p>
            <p className="font-sans font-light text-steel text-[0.85rem] leading-[1.7] mt-3 max-w-[320px]">
              The undirected coordination conductance between any two nodes. The system's glue.
            </p>

            {/* Network diagram */}
            <svg viewBox="0 0 360 220" className="w-full max-w-[400px] mt-6">
              {NETWORK_EDGES.map(([a, b], i) => {
                const na = NETWORK_NODES[a];
                const nb = NETWORK_NODES[b];
                const strength = 0.3 + (Math.sin(i * 1.5) * 0.3 + 0.3);
                return (
                  <line
                    key={`edge-${i}`}
                    className="net-edge"
                    x1={na.x} y1={na.y} x2={nb.x} y2={nb.y}
                    stroke="#5A7A6A"
                    strokeWidth={strength * 2.5}
                    opacity={strength}
                  />
                );
              })}
              {NETWORK_NODES.map((n, i) => (
                <g key={`node-${i}`}>
                  <circle className="net-node" cx={n.x} cy={n.y} r="10" fill={i < 3 ? '#5A7A6A' : '#C8943A'} opacity="0.9" />
                  <circle cx={n.x} cy={n.y} r="4" fill="#1A1D21" />
                </g>
              ))}
            </svg>
          </div>

          {/* Right Panel - Shear Force */}
          <div ref={rightPanelRef} className="lg:w-1/2 lg:pl-12 pt-12 lg:pt-0 border-t lg:border-t-0 border-divider">
            <p className="font-mono text-amber text-[1.4rem] font-light">
              <var>{'\u03a9'}</var>(<var>t</var>) = std(<var>{'\u0394'}</var><var>S</var><sub><var>i</var></sub>)
            </p>
            <p className="font-mono text-amber text-[1.2rem] font-light mt-1">
              <var>{'\u03a9'}</var><sub><var>{'\u03c3'}</var></sub>(<var>t</var>) = std(<var>{'\u0394'}</var><var>{'\u03c3'}</var><sub><var>i</var></sub>)
            </p>
            <p className="font-sans font-light text-steel text-[0.85rem] leading-[1.7] mt-3 max-w-[320px]">
              The dispersive force tearing the network apart via structural dispersion.
            </p>

            {/* Dispersion diagram */}
            <svg viewBox="0 0 360 220" className="w-full max-w-[400px] mt-6">
              {/* Central node */}
              <circle cx={180} cy={110} r="12" fill="#E8E4DF" />
              <circle cx={180} cy={110} r="5" fill="#1A1D21" />

              {/* Dispersing arrows */}
              {[
                { angle: -60, len: 70, color: '#8A4A4A' },
                { angle: -30, len: 55, color: '#8A4A4A' },
                { angle: 0, len: 80, color: '#8A4A4A' },
                { angle: 30, len: 60, color: '#8A4A4A' },
                { angle: 60, len: 75, color: '#8A4A4A' },
                { angle: 120, len: 40, color: '#5A7A6A' },
                { angle: 150, len: 35, color: '#5A7A6A' },
                { angle: 180, len: 45, color: '#5A7A6A' },
                { angle: 210, len: 30, color: '#5A7A6A' },
                { angle: 240, len: 38, color: '#5A7A6A' },
              ].map((arrow, i) => {
                const rad = (arrow.angle * Math.PI) / 180;
                const x2 = 180 + Math.cos(rad) * arrow.len;
                const y2 = 110 + Math.sin(rad) * arrow.len;
                const headLen = 8;
                const headAngle1 = rad + Math.PI * 0.8;
                const headAngle2 = rad - Math.PI * 0.8;
                return (
                  <g key={`arrow-${i}`}>
                    <line
                      className="disp-arrow"
                      x1={180} y1={110}
                      x2={x2 - Math.cos(rad) * 12}
                      y2={y2 - Math.sin(rad) * 12}
                      stroke={arrow.color}
                      strokeWidth="2"
                      strokeLinecap="round"
                    />
                    <polygon
                      points={`${x2},${y2} ${x2 - headLen * Math.cos(headAngle1)},${y2 - headLen * Math.sin(headAngle1)} ${x2 - headLen * Math.cos(headAngle2)},${y2 - headLen * Math.sin(headAngle2)}`}
                      fill={arrow.color}
                    />
                  </g>
                );
              })}
            </svg>
          </div>
        </div>
      </div>
    </section>
  );
}
