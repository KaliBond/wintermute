import { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import { SectionEyebrow } from '../components/SectionEyebrow';

gsap.registerPlugin(ScrollTrigger);

const SCISSORS_DATA = [
  { year: 1900, scissors: 8, affect: 1.2 },
  { year: 1910, scissors: 12, affect: -0.5 },
  { year: 1920, scissors: 5, affect: -2.1 },
  { year: 1930, scissors: 18, affect: -1.8 },
  { year: 1940, scissors: -5, affect: -3.5 },
  { year: 1950, scissors: 22, affect: 0.5 },
  { year: 1960, scissors: 28, affect: 2.8 },
  { year: 1970, scissors: 30, affect: 3.2 },
  { year: 1980, scissors: 30, affect: 3.5 },
  { year: 1990, scissors: 28, affect: 3.0 },
  { year: 2000, scissors: 25, affect: 2.5 },
  { year: 2010, scissors: 18, affect: 1.8 },
  { year: 2020, scissors: 12, affect: 0.5 },
];

export function ScissorsOperatorSection() {
  const sectionRef = useRef<HTMLElement>(null);
  const formulaRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<SVGSVGElement>(null);
  const descRef = useRef<HTMLParagraphElement>(null);

  const [hoveredPoint, setHoveredPoint] = useState<number | null>(null);
  const [mouseX, setMouseX] = useState<number | null>(null);

  useEffect(() => {
    const section = sectionRef.current;
    const formula = formulaRef.current;
    const chart = chartRef.current;
    const desc = descRef.current;
    if (!section || !formula || !chart || !desc) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const scissorsPath = chart.querySelector('.scissors-path');
    const affectPath = chart.querySelector('.affect-path');
    const labels = chart.querySelectorAll('.chart-label');

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: section,
        start: 'top 70%',
        toggleActions: 'play none none none',
      },
    });

    tl.fromTo(formula, { y: 30, opacity: 0 }, { y: 0, opacity: 1, duration: 1, ease: 'power3.out' });

    if (scissorsPath) {
      const len = (scissorsPath as SVGPathElement).getTotalLength();
      gsap.set(scissorsPath, { strokeDasharray: len, strokeDashoffset: len });
      tl.to(scissorsPath, { strokeDashoffset: 0, duration: 2, ease: 'power2.out' }, 0.5);
    }
    if (affectPath) {
      const len = (affectPath as SVGPathElement).getTotalLength();
      gsap.set(affectPath, { strokeDasharray: len, strokeDashoffset: len });
      tl.to(affectPath, { strokeDashoffset: 0, duration: 2, ease: 'power2.out' }, 0.7);
    }

    tl.fromTo(labels, { opacity: 0 }, { opacity: 1, duration: 0.4, stagger: 0.1 }, 2)
      .fromTo(desc, { y: 20, opacity: 0 }, { y: 0, opacity: 1, duration: 0.8, ease: 'power3.out' }, 1.5);

    return () => { tl.kill(); };
  }, []);

  const chartW = 700;
  const chartH = 320;
  const padL = 50;
  const padR = 50;
  const padT = 30;
  const padB = 40;
  const plotW = chartW - padL - padR;
  const plotH = chartH - padT - padB;

  const toX = (year: number) => padL + ((year - 1900) / 120) * plotW;
  const toY1 = (v: number) => padT + plotH - ((v + 10) / 50) * plotH;
  const toY2 = (v: number) => padT + plotH - ((v + 5) / 12) * plotH;

  const scissorsLineD = SCISSORS_DATA.map((d, i) => `${i === 0 ? 'M' : 'L'} ${toX(d.year)} ${toY1(d.scissors)}`).join(' ');
  const affectLineD = SCISSORS_DATA.map((d, i) => `${i === 0 ? 'M' : 'L'} ${toX(d.year)} ${toY2(d.affect)}`).join(' ');

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * chartW;
    setMouseX(x);

    // Find closest data point
    let closest = 0;
    let minDist = Infinity;
    SCISSORS_DATA.forEach((d, i) => {
      const dist = Math.abs(toX(d.year) - x);
      if (dist < minDist) {
        minDist = dist;
        closest = i;
      }
    });
    if (minDist < 30) {
      setHoveredPoint(closest);
    } else {
      setHoveredPoint(null);
    }
  };

  return (
    <section ref={sectionRef} className="bg-charcoal section-padding">
      <div className="content-max-width">
        <div className="text-center">
          <SectionEyebrow text="09 / The Precursor Signal" color="text-crimson" className="mb-4 block" />
          <h2 className="font-serif text-parchment text-[clamp(1.8rem,3vw,2.5rem)] tracking-[-0.03em]">
            The Scissors Operator
          </h2>

          <div ref={formulaRef} className="mt-8">
            <p className="font-mono text-parchment text-[clamp(1.4rem,2.5vw,2.2rem)] font-light">
              <var>{'\u03a6'}</var>(<var>t</var>) = <var>M</var>&#775;(<var>t</var>) - <var>Y</var>&#775;(<var>t</var>)
            </p>
            <div className="flex justify-center gap-8 mt-4">
              <span className="font-sans font-light text-amber text-[0.8rem]">
                <var>M</var>&#775;(<var>t</var>): Rate of change in Metabolic Load (ascending)
              </span>
              <span className="font-sans font-light text-sage text-[0.8rem]">
                <var>Y</var>&#775;(<var>t</var>): Rate of change in Mythic Integration (descending)
              </span>
            </div>
          </div>

          <p ref={descRef} className="font-sans font-light text-steel text-[1rem] max-w-[600px] mx-auto mt-8 leading-[1.7]">
            The universal precursor signal for the K → {'\u03a9'} transition. Sustained positive <var>{'\u03a6'}</var>(<var>t</var>) visually and mathematically demonstrates a system's stress accelerating faster than its coordination capacity can adapt.
          </p>

          {/* Chart */}
          <div className="mt-16 overflow-x-auto">
            <svg
              ref={chartRef}
              viewBox={`0 0 ${chartW} ${chartH}`}
              className="w-full min-w-[500px]"
              onMouseMove={handleMouseMove}
              onMouseLeave={() => { setHoveredPoint(null); setMouseX(null); }}
            >
              {/* Grid lines */}
              {[0, 10, 20, 30].map(tick => (
                <line key={`h-${tick}`} x1={padL} y1={toY1(tick)} x2={chartW - padR} y2={toY1(tick)} stroke="#3A4148" strokeWidth="0.5" />
              ))}
              {[1900, 1920, 1940, 1960, 1980, 2000, 2020].map(year => (
                <line key={`v-${year}`} x1={toX(year)} y1={padT} x2={toX(year)} y2={chartH - padB} stroke="#3A4148" strokeWidth="0.5" />
              ))}

              {/* Scissors area fill */}
              <path d={`${scissorsLineD} L ${toX(2020)} ${toY1(0)} L ${toX(1900)} ${toY1(0)} Z`} fill="#C8943A" opacity="0.08" />

              {/* Scissors line */}
              <path className="scissors-path" d={scissorsLineD} fill="none" stroke="#C8943A" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />

              {/* Affect line */}
              <path className="affect-path" d={affectLineD} fill="none" stroke="#8A4A4A" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="4 4" />

              {/* X-axis labels */}
              {[1900, 1920, 1940, 1960, 1980, 2000, 2020].map(year => (
                <text key={`xl-${year}`} x={toX(year)} y={chartH - padB + 20} fill="#8A9499" fontSize="10" fontFamily="JetBrains Mono" textAnchor="middle" className="chart-label">{year}</text>
              ))}

              {/* Y-axis labels (left - scissors) */}
              {[0, 10, 20, 30].map(tick => (
                <text key={`yl-${tick}`} x={padL - 8} y={toY1(tick) + 4} fill="#C8943A" fontSize="9" fontFamily="JetBrains Mono" textAnchor="end" className="chart-label">{tick}</text>
              ))}

              {/* Y-axis labels (right - affect) */}
              {[-4, -2, 0, 2, 4].map(tick => (
                <text key={`yr-${tick}`} x={chartW - padR + 8} y={toY2(tick) + 4} fill="#8A4A4A" fontSize="9" fontFamily="JetBrains Mono" textAnchor="start" className="chart-label">{tick}</text>
              ))}

              {/* Axis titles */}
              <text x={padL - 35} y={chartH / 2} fill="#C8943A" fontSize="10" fontFamily="Inter" textAnchor="middle" transform={`rotate(-90, ${padL - 35}, ${chartH / 2})`} className="chart-label">Scissors {'\u03a6'}(t)</text>
              <text x={chartW - padR + 35} y={chartH / 2} fill="#8A4A4A" fontSize="10" fontFamily="Inter" textAnchor="middle" transform={`rotate(90, ${chartW - padR + 35}, ${chartH / 2})`} className="chart-label">System Affect {'\u03c3'}(t)</text>

              {/* Hover cursor line */}
              {mouseX !== null && (
                <line x1={mouseX} y1={padT} x2={mouseX} y2={chartH - padB} stroke="#E8E4DF" strokeWidth="0.5" opacity="0.3" />
              )}

              {/* Hovered point tooltip */}
              {hoveredPoint !== null && (
                <g>
                  <circle cx={toX(SCISSORS_DATA[hoveredPoint].year)} cy={toY1(SCISSORS_DATA[hoveredPoint].scissors)} r="5" fill="#C8943A" />
                  <circle cx={toX(SCISSORS_DATA[hoveredPoint].year)} cy={toY2(SCISSORS_DATA[hoveredPoint].affect)} r="4" fill="#8A4A4A" />
                  <rect x={toX(SCISSORS_DATA[hoveredPoint].year) + 10} y={toY1(SCISSORS_DATA[hoveredPoint].scissors) - 30} width="100" height="44" fill="#2A2F35" stroke="#3A4148" rx="2" />
                  <text x={toX(SCISSORS_DATA[hoveredPoint].year) + 16} y={toY1(SCISSORS_DATA[hoveredPoint].scissors) - 14} fill="#C8943A" fontSize="9" fontFamily="JetBrains Mono">{'\u03a6'} = {SCISSORS_DATA[hoveredPoint].scissors}</text>
                  <text x={toX(SCISSORS_DATA[hoveredPoint].year) + 16} y={toY1(SCISSORS_DATA[hoveredPoint].scissors) - 2} fill="#8A4A4A" fontSize="9" fontFamily="JetBrains Mono">{'\u03c3'} = {SCISSORS_DATA[hoveredPoint].affect}</text>
                </g>
              )}
            </svg>
          </div>
        </div>
      </div>
    </section>
  );
}
