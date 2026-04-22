import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SectionEyebrow } from '../components/SectionEyebrow';

gsap.registerPlugin(ScrollTrigger);

const QUADRANTS = [
  { name: 'Library', x: 45, y: 2, color: '#5A7A6A', desc: 'Low M, High Y. The universal attractor for stable social organisms.' },
  { name: 'Praetorian', x: 52, y: 8, color: '#C8943A', desc: 'High M, High Y. Maintenance of order through coercion and rigidity.' },
  { name: 'Fragmentation', x: 12, y: 8, color: '#8A4A4A', desc: 'High M, Low Y. Unraveling and systemic collapse.' },
  { name: 'Latent', x: 15, y: 2, color: '#8A9499', desc: 'Low M, Low Y. Reorganization and dormant potential.' },
];

const TRAJECTORY = [
  [45, 2], [52, 4], [55, 8], [52, 8], [40, 8.5], [25, 7], [12, 8], [8, 5], [10, 2.5], [15, 2], [30, 1.8], [45, 2],
];

const TABLE_ROWS = [
  { holling: 'r (Exploitation)', cams: 'Library', mechanism: 'Capital accumulation under strong symbolic coordination; entropy export balanced.', color: '#5A7A6A' },
  { holling: 'K (Conservation)', cams: 'Praetorian', mechanism: 'Rigidity trap onset; stress rises but mythic coherence is maintained through regulation.', color: '#C8943A' },
  { holling: '\u03a9 (Release)', cams: 'Fragmentation', mechanism: 'Rapid dissolution; mythic-material decoupling.', color: '#8A4A4A' },
  { holling: '\u03b1 (Reorganization)', cams: 'Latent', mechanism: 'Novel configurations tested; potential re-coupling of mythic and material layers.', color: '#8A9499' },
];

export function PhaseSpaceSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const chartRef = useRef<HTMLDivElement>(null);
  const pathRef = useRef<SVGPathElement>(null);
  const tableRef = useRef<HTMLDivElement>(null);
  const conclusionRef = useRef<HTMLParagraphElement>(null);

  const [hoveredQuad, setHoveredQuad] = useState<string | null>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const chart = chartRef.current;
    const path = pathRef.current;
    const table = tableRef.current;
    const conclusion = conclusionRef.current;
    if (!section || !chart || !path || !table || !conclusion) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const points = chart.querySelectorAll('.quad-point');
    const labels = chart.querySelectorAll('.quad-label');
    const rows = table.querySelectorAll('.table-row');

    const pathLength = path.getTotalLength();
    gsap.set(path, { strokeDasharray: pathLength, strokeDashoffset: pathLength });

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: section,
        start: 'top 60%',
        toggleActions: 'play none none none',
      },
    });

    tl.fromTo(chart, { y: 60, opacity: 0 }, { y: 0, opacity: 1, duration: 1.2, ease: 'power3.out' })
      .to(path, { strokeDashoffset: 0, duration: 2, ease: 'power2.out' }, 0.5)
      .fromTo(points, { scale: 0 }, { scale: 1, duration: 0.5, stagger: 0.15, ease: 'back.out(1.7)' }, 1.5)
      .fromTo(labels, { opacity: 0 }, { opacity: 1, duration: 0.5, stagger: 0.2 }, 1.8)
      .fromTo(rows, { x: -20, opacity: 0 }, { x: 0, opacity: 1, duration: 0.6, stagger: 0.1, ease: 'power3.out' }, 2)
      .fromTo(conclusion, { y: 20, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, ease: 'power3.out' }, 2.5);

    return () => { tl.kill(); };
  }, []);

  const chartW = 600;
  const chartH = 320;
  const padL = 60;
  const padR = 20;
  const padT = 20;
  const padB = 50;
  const plotW = chartW - padL - padR;
  const plotH = chartH - padT - padB;

  const toX = (v: number) => padL + (v / 60) * plotW;
  const toY = (v: number) => padT + plotH - (v / 10) * plotH;

  const pathD = TRAJECTORY.map((p, i) => `${i === 0 ? 'M' : 'L'} ${toX(p[0])} ${toY(p[1])}`).join(' ');

  return (
    <section id="phase-space" ref={sectionRef} className="bg-charcoal-light section-padding">
      <div className="content-max-width">
        <SectionEyebrow text="06 / Phase Space" color="text-sage" className="mb-4 block" />
        <h2 className="font-serif text-parchment text-[clamp(1.8rem,3.5vw,2.8rem)] tracking-[-0.03em] mb-16">
          Holling Adaptive Cycle in CAMS Phase Space
        </h2>

        {/* Phase Space Chart */}
        <div ref={chartRef} className="relative w-full overflow-x-auto">
          <svg viewBox={`0 0 ${chartW} ${chartH}`} className="w-full min-w-[500px]">
            {/* Grid */}
            {[0, 2, 4, 6, 8, 10].map(tick => (
              <line key={`h-${tick}`} x1={padL} y1={toY(tick)} x2={chartW - padR} y2={toY(tick)} stroke="#3A4148" strokeWidth="0.5" />
            ))}
            {[0, 10, 20, 30, 40, 50, 60].map(tick => (
              <line key={`v-${tick}`} x1={toX(tick)} y1={padT} x2={toX(tick)} y2={chartH - padB} stroke="#3A4148" strokeWidth="0.5" />
            ))}

            {/* Axes */}
            <line x1={padL} y1={chartH - padB} x2={chartW - padR} y2={chartH - padB} stroke="#E8E4DF" strokeWidth="1" />
            <line x1={padL} y1={padT} x2={padL} y2={chartH - padB} stroke="#E8E4DF" strokeWidth="1" />

            {/* Y-axis labels */}
            {[0, 2, 4, 6, 8, 10].map(tick => (
              <text key={`yl-${tick}`} x={padL - 8} y={toY(tick) + 4} fill="#8A9499" fontSize="10" fontFamily="JetBrains Mono" textAnchor="end">{tick}</text>
            ))}
            {/* X-axis labels */}
            {[0, 10, 20, 30, 40, 50, 60].map(tick => (
              <text key={`xl-${tick}`} x={toX(tick)} y={chartH - padB + 20} fill="#8A9499" fontSize="10" fontFamily="JetBrains Mono" textAnchor="middle">{tick}</text>
            ))}

            {/* Axis titles */}
            <text x={chartW / 2 + padL / 2} y={chartH - 8} fill="#8A9499" fontSize="11" fontFamily="Inter" textAnchor="middle">Mythic Integration Y(t)</text>
            <text x={14} y={chartH / 2} fill="#8A9499" fontSize="11" fontFamily="Inter" textAnchor="middle" transform={`rotate(-90, 14, ${chartH / 2})`}>Metabolic Load M(t)</text>

            {/* Trajectory path */}
            <path ref={pathRef} d={pathD} fill="none" stroke="#5A7A6A" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" opacity="0.8" />

            {/* Direction arrows along path */}
            {[3, 7].map(i => (
              <circle key={`arrow-${i}`} cx={toX(TRAJECTORY[i][0])} cy={toY(TRAJECTORY[i][1])} r="3" fill="#5A7A6A" />
            ))}

            {/* Quadrant points and labels */}
            {QUADRANTS.map((q) => (
              <g key={q.name}>
                <circle
                  className="quad-point"
                  cx={toX(q.x)}
                  cy={toY(q.y)}
                  r="6"
                  fill={q.color}
                  opacity={hoveredQuad === q.name ? 1 : hoveredQuad ? 0.4 : 0.9}
                  style={{ cursor: 'pointer', transition: 'opacity 0.2s' }}
                  onMouseEnter={() => setHoveredQuad(q.name)}
                  onMouseLeave={() => setHoveredQuad(null)}
                />
                <text
                  className="quad-label"
                  x={toX(q.x)}
                  y={toY(q.y) - 14}
                  fill={q.color}
                  fontSize="12"
                  fontFamily="Inter"
                  fontWeight="500"
                  textAnchor="middle"
                >
                  {q.name}
                </text>
              </g>
            ))}
          </svg>

          {/* Tooltip */}
          {hoveredQuad && (
            <div className="absolute bg-charcoal-surface border border-divider px-4 py-3 rounded max-w-[280px] pointer-events-none" style={{ top: '10%', right: '5%' }}>
              <p className="font-sans font-medium text-parchment text-[0.85rem]">{hoveredQuad}</p>
              <p className="font-sans font-light text-steel text-[0.75rem] mt-1">
                {QUADRANTS.find(q => q.name === hoveredQuad)?.desc}
              </p>
            </div>
          )}
        </div>

        {/* Isomorphism Table */}
        <div ref={tableRef} className="mt-20">
          <h3 className="font-serif text-parchment text-[1.4rem] mb-6">The Panarchy-CAMS Isomorphism</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-charcoal">
                  <th className="font-sans font-medium text-steel text-[0.75rem] uppercase tracking-[0.08em] text-left px-6 py-4 border-b border-divider">Holling Ecological Phase</th>
                  <th className="font-sans font-medium text-steel text-[0.75rem] uppercase tracking-[0.08em] text-left px-6 py-4 border-b border-divider">CAMS Quadrant</th>
                  <th className="font-sans font-medium text-steel text-[0.75rem] uppercase tracking-[0.08em] text-left px-6 py-4 border-b border-divider">Thermodynamic Mechanism</th>
                </tr>
              </thead>
              <tbody>
                {TABLE_ROWS.map((row) => (
                  <tr key={row.holling} className="table-row border-b border-divider hover:bg-charcoal-surface/50 transition-colors">
                    <td className="font-mono text-parchment text-[0.85rem] px-6 py-4 font-light">{row.holling}</td>
                    <td className="font-sans text-[0.85rem] px-6 py-4 font-medium" style={{ color: row.color }}>{row.cams}</td>
                    <td className="font-sans font-light text-parchment text-[0.85rem] px-6 py-4 leading-[1.6]">{row.mechanism}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Conclusion */}
        <p ref={conclusionRef} className="font-serif italic text-sage text-[1.2rem] text-center mt-12">
          The Holling cycle is simply the trajectory of a dissipative structure through its own free-energy landscape.
        </p>
      </div>
    </section>
  );
}
