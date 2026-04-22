import { useState, useEffect } from 'react';

const NAV_LINKS = [
  { label: 'Framework', href: '#manifesto' },
  { label: 'Diagnostics', href: '#diagnostics' },
  { label: 'Phase Space', href: '#phase-space' },
];

export function Navigation() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, href: string) => {
    e.preventDefault();
    setMobileOpen(false);
    const el = document.querySelector(href);
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <nav
      className={`fixed top-0 left-0 right-0 z-50 h-16 flex items-center transition-all duration-300 ${
        scrolled ? 'bg-charcoal/85 backdrop-blur-[12px]' : 'bg-transparent'
      }`}
    >
      <div className="content-max-width w-full flex items-center justify-between px-6 sm:px-12 lg:px-20">
        {/* Brand */}
        <a
          href="#"
          className="font-sans font-medium text-[0.8rem] uppercase tracking-[0.12em] text-parchment hover:text-sage transition-colors duration-300"
          onClick={(e) => {
            e.preventDefault();
            window.scrollTo({ top: 0, behavior: 'smooth' });
          }}
        >
          Neural Nations
        </a>

        {/* Desktop Nav */}
        <div className="hidden md:flex items-center gap-8">
          {NAV_LINKS.map((link) => (
            <a
              key={link.label}
              href={link.href}
              onClick={(e) => handleNavClick(e, link.href)}
              className="nav-link-underline font-sans text-[0.75rem] uppercase tracking-[0.08em] text-steel hover:text-parchment transition-colors duration-300"
            >
              {link.label}
            </a>
          ))}
          <a
            href="#footer"
            onClick={(e) => handleNavClick(e, '#footer')}
            className="font-sans font-medium text-[0.7rem] uppercase tracking-[0.1em] text-sage border border-sage px-5 py-2 rounded-[2px] hover:bg-sage hover:text-charcoal transition-all duration-300"
          >
            Read the Paper
          </a>
        </div>

        {/* Mobile Hamburger */}
        <button
          className="md:hidden flex flex-col gap-[5px] p-2"
          onClick={() => setMobileOpen(!mobileOpen)}
          aria-label="Toggle menu"
        >
          <span
            className={`block w-5 h-[1.5px] bg-parchment transition-all duration-300 ${
              mobileOpen ? 'rotate-45 translate-y-[6.5px]' : ''
            }`}
          />
          <span
            className={`block w-5 h-[1.5px] bg-parchment transition-all duration-300 ${
              mobileOpen ? 'opacity-0' : ''
            }`}
          />
          <span
            className={`block w-5 h-[1.5px] bg-parchment transition-all duration-300 ${
              mobileOpen ? '-rotate-45 -translate-y-[6.5px]' : ''
            }`}
          />
        </button>
      </div>

      {/* Mobile Menu */}
      {mobileOpen && (
        <div className="absolute top-16 left-0 right-0 bg-charcoal/95 backdrop-blur-[12px] border-t border-divider md:hidden">
          <div className="flex flex-col py-6 px-6 gap-4">
            {NAV_LINKS.map((link) => (
              <a
                key={link.label}
                href={link.href}
                onClick={(e) => handleNavClick(e, link.href)}
                className="font-sans text-[0.85rem] uppercase tracking-[0.08em] text-steel hover:text-parchment transition-colors duration-300 py-2"
              >
                {link.label}
              </a>
            ))}
            <a
              href="#footer"
              onClick={(e) => handleNavClick(e, '#footer')}
              className="font-sans font-medium text-[0.75rem] uppercase tracking-[0.1em] text-sage border border-sage px-5 py-3 rounded-[2px] text-center hover:bg-sage hover:text-charcoal transition-all duration-300 mt-2"
            >
              Read the Paper
            </a>
          </div>
        </div>
      )}
    </nav>
  );
}
