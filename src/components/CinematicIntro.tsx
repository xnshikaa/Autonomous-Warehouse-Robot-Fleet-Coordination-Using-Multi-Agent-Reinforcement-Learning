import React, { useEffect, useState } from 'react';
import gsap from 'gsap';

interface CinematicIntroProps {
  onComplete: () => void;
}

export const CinematicIntro: React.FC<CinematicIntroProps> = ({ onComplete }) => {
  const [step, setStep] = useState(0);

  useEffect(() => {
    const t1 = setTimeout(() => setStep(1), 300);
    const t2 = setTimeout(() => setStep(2), 1400);
    const t3 = setTimeout(() => setStep(3), 2600);
    const t4 = setTimeout(() => {
      onComplete();
    }, 3800);

    return () => {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      clearTimeout(t4);
    };
  }, []);

  return (
    <div className="fixed inset-0 z-50 bg-industrial-900 flex flex-col items-center justify-center font-mono pointer-events-auto transition-opacity duration-1000">
      <div className="text-center space-y-6 max-w-xl p-8">
        {step >= 1 && (
          <div className="text-sm uppercase tracking-widest text-slate-400 font-semibold animate-fade-in">
            AUTONOMOUS WAREHOUSE SIMULATION
          </div>
        )}

        {step >= 2 && (
          <div className="text-xl md:text-2xl font-bold text-slate-100 tracking-wide leading-snug">
            MULTI-AGENT REINFORCEMENT LEARNING
          </div>
        )}

        {step >= 3 && (
          <div className="inline-block px-6 py-2 rounded-xl bg-industrial-accent/20 border border-industrial-accent text-industrial-accent font-bold text-lg shadow-2xl shadow-industrial-accent/30 tracking-wider animate-pulse">
            ALGORITHM: QMIX (CTDE)
          </div>
        )}

        <div className="pt-8 text-[11px] text-slate-500 tracking-widest uppercase">
          Initializing 3D Telemetry Environment...
        </div>
      </div>
    </div>
  );
};
