import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

export function Footer() {
  const footerRef = useRef<HTMLElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);
  const linksRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const footer = footerRef.current;
    const cta = ctaRef.current;
    const links = linksRef.current;
    if (!footer || !cta || !links) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) return;

    const linkEls = links.querySelectorAll('.footer-link');

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: footer,
        start: 'top 90%',
        toggleActions: 'play none none none',
      },
    });

    tl.fromTo(cta, { y: 40, opacity: 0 }, { y: 0, opacity: 1, duration: 1, ease: 'power3.out' })
      .fromTo(linkEls, { opacity: 0 }, { opacity: 1, duration: 0.4, stagger: 0.1 }, 0.5);

    return () => { tl.kill(); };
  }, []);

  return (
    <footer id="footer" ref={footerRef} className="bg-charcoal border-t border-divider">
      {/* Top Row - CTA */}
      <div ref={ctaRef} className="pt-[120px] pb-20 px-6 sm:px-12 lg:px-20">
        <div className="content-max-width text-center">
          <h2 className="font-serif text-parchment text-[clamp(2rem,4vw,3.5rem)] tracking-[-0.03em]">
            Read the Full Framework
          </h2>
          <p className="font-sans font-light text-steel text-[1rem] mt-4">
            The complete CAMS v3.2-R-ESCH specification with all derivations, proofs, and case studies.
          </p>
          <button className="mt-10 font-sans font-medium text-[0.8rem] uppercase tracking-[0.12em] text-charcoal bg-sage px-9 py-3.5 rounded-[2px] hover:bg-[#6A8A7A] transition-colors duration-300 cursor-pointer">
            Download the Paper
          </button>
        </div>
      </div>

      {/* Bottom Row */}
      <div ref={linksRef} className="border-t border-divider py-10 px-6 sm:px-12 lg:px-20">
        <div className="content-max-width flex flex-col md:flex-row items-center justify-between gap-4">
          <span className="font-sans font-medium text-[0.8rem] text-parchment tracking-[0.1em]">
            Neural Nations
          </span>
          <span className="font-sans font-light text-[0.7rem] text-steel">
            © 2025 Neural Nations Research. All frameworks released under open science principles.
          </span>
          <div className="flex gap-6">
            {['GitHub', 'Contact', 'Cite'].map((link) => (
              <a
                key={link}
                href="#"
                className="footer-link font-sans text-[0.7rem] text-steel hover:text-parchment transition-colors duration-200"
                onClick={(e) => e.preventDefault()}
              >
                {link}
              </a>
            ))}
          </div>
        </div>
      </div>
    </footer>
  );
}
