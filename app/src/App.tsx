import { useEffect, useRef } from 'react';
import Lenis from 'lenis';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

import { Navigation } from './sections/Navigation';
import { HeroSection } from './sections/HeroSection';
import { ManifestoSection } from './sections/ManifestoSection';
import { CoreAxiomsSection } from './sections/CoreAxiomsSection';
import { ThermodynamicBaselineSection } from './sections/ThermodynamicBaselineSection';
import { DiagnosticPathologySection } from './sections/DiagnosticPathologySection';
import { ActivationIndexSection } from './sections/ActivationIndexSection';
import { PhaseSpaceSection } from './sections/PhaseSpaceSection';
import { DivergenceFieldSection } from './sections/DivergenceFieldSection';
import { ScaleFreeValidationSection } from './sections/ScaleFreeValidationSection';
import { ScissorsOperatorSection } from './sections/ScissorsOperatorSection';
import { KuramotoBridgeSection } from './sections/KuramotoBridgeSection';
import { SystemDynamicsSection } from './sections/SystemDynamicsSection';
import { Footer } from './sections/Footer';

gsap.registerPlugin(ScrollTrigger);

function App() {
  const lenisRef = useRef<Lenis | null>(null);

  useEffect(() => {
    const lenis = new Lenis({
      lerp: 0.08,
      duration: 1.2,
    });
    lenisRef.current = lenis;

    // Sync Lenis with GSAP ScrollTrigger
    lenis.on('scroll', ScrollTrigger.update);

    gsap.ticker.add((time) => {
      lenis.raf(time * 1000);
    });

    gsap.ticker.lagSmoothing(0);

    return () => {
      lenis.destroy();
      gsap.ticker.remove(lenis.raf as unknown as gsap.TickerCallback);
    };
  }, []);

  return (
    <div className="bg-charcoal min-h-screen">
      <Navigation />
      <HeroSection />
      <ManifestoSection />
      <CoreAxiomsSection />
      <ThermodynamicBaselineSection />
      <DiagnosticPathologySection />
      <ActivationIndexSection />
      <PhaseSpaceSection />
      <DivergenceFieldSection />
      <ScaleFreeValidationSection />
      <ScissorsOperatorSection />
      <KuramotoBridgeSection />
      <SystemDynamicsSection />
      <Footer />
    </div>
  );
}

export default App;
