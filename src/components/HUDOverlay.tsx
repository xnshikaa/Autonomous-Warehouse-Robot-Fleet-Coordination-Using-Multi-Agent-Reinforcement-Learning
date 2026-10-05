import React from 'react';
import { 
  WarehouseState, 
  AlgorithmType 
} from '../types/warehouse';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  ShieldAlert, 
  Bot, 
  CheckCircle2, 
  Maximize2, 
  Zap, 
  AlertTriangle,
  Radio
} from 'lucide-react';

interface HUDOverlayProps {
  state: WarehouseState;
  onFleetSizeChange: (size: number) => void;
  onPauseToggle: () => void;
  onReset: () => void;
  onSpeedChange: (speed: number) => void;
  onPresentationModeToggle: () => void;
  isPaused: boolean;
  currentSpeed: number;
  isPresentationMode: boolean;
}

export const HUDOverlay: React.FC<HUDOverlayProps> = ({
  state,
  onFleetSizeChange,
  onPauseToggle,
  onReset,
  onSpeedChange,
  onPresentationModeToggle,
  isPaused,
  currentSpeed,
  isPresentationMode
}) => {
  const { metrics, isConnectedToBackend, safetyOverrideActive } = state;

  return (
    <div className="absolute inset-0 pointer-events-none z-30 flex flex-col justify-between p-4 font-mono text-xs">
      {/* Top Header Bar */}
      <div className="flex items-start justify-between w-full pointer-events-auto">
        <div className="glass-panel rounded-2xl p-4 border border-slate-700/60 shadow-2xl flex items-center space-x-5">
          {/* Title Branding */}
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-industrial-accent/15 border border-industrial-accent/40 flex items-center justify-center text-industrial-accent shadow-inner">
              <Bot className="w-6 h-6" />
            </div>
            <div>
              <h1 className="text-base font-bold tracking-wider text-slate-100 flex items-center gap-2">
                AUTONOMOUS WAREHOUSE
              </h1>
              <p className="text-[11px] text-slate-400 flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
                SIMULATION & MARL INTERFACE
              </p>
            </div>
          </div>

          <div className="h-8 w-[1px] bg-slate-800" />

          {/* Simulation Status Badges */}
          <div className="flex items-center space-x-2 text-[11px]">
            <span className="px-2.5 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-bold">
              MODE: SIMULATION
            </span>

            <span className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-700/80 text-slate-300">
              POLICY: SIMULATION CONTROLLER
            </span>

            <span className="px-2.5 py-1 rounded-lg bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 font-bold">
              TARGET: QMIX READY
            </span>
          </div>

          <div className="h-8 w-[1px] bg-slate-800" />

          {/* Fleet Size Selection */}
          <div className="flex items-center space-x-1.5 bg-slate-900/80 p-1.5 rounded-xl border border-slate-800 text-slate-400">
            <span className="px-1.5 text-[10px] uppercase text-slate-500 font-bold">Fleet:</span>
            {[5, 10, 20].map(size => (
              <button
                key={size}
                onClick={() => onFleetSizeChange(size)}
                className={`px-2.5 py-1 rounded-lg transition-all ${
                  metrics.totalRobots === size
                    ? 'bg-industrial-accent text-slate-950 font-bold shadow-md shadow-industrial-accent/20'
                    : 'hover:text-white'
                }`}
              >
                N={size}
              </button>
            ))}
          </div>

          {/* Connection Telemetry Tag */}
          <div className="flex items-center space-x-2">
            {isConnectedToBackend ? (
              <span className="px-3 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-bold flex items-center gap-1.5">
                <Radio className="w-3.5 h-3.5 animate-pulse" /> LIVE BACKEND
              </span>
            ) : (
              <span className="px-3 py-1 rounded-lg bg-slate-900 border border-slate-800 text-slate-400 flex items-center gap-1.5">
                <Zap className="w-3.5 h-3.5 text-industrial-accent" /> LOCAL SIM
              </span>
            )}
          </div>
        </div>

        {/* Presentation Toggle */}
        <div className="pointer-events-auto flex items-center space-x-2">
          <button
            onClick={onPresentationModeToggle}
            className={`flex items-center space-x-2 px-4 py-2.5 rounded-2xl transition-all glass-panel border border-slate-700/60 shadow-2xl ${
              isPresentationMode
                ? 'bg-industrial-accent text-slate-950 font-bold shadow-lg shadow-industrial-accent/20'
                : 'text-slate-300 hover:text-white'
            }`}
          >
            <Maximize2 className="w-4 h-4" />
            <span>PRESENTATION</span>
          </button>
        </div>
      </div>

      {/* Safety Action Override Warning Banner */}
      {safetyOverrideActive && (
        <div className="w-full max-w-xl mx-auto pointer-events-auto glass-panel border border-amber-500/50 rounded-xl p-2.5 bg-amber-500/10 flex items-center justify-between text-amber-400 text-xs shadow-2xl animate-pulse">
          <div className="flex items-center space-x-2.5">
            <AlertTriangle className="w-4 h-4 text-amber-400" />
            <span className="font-bold">SAFETY OVERRIDE TRIGGERED → WAIT</span>
          </div>
          <span className="text-[10px] text-amber-300">DETERMINISTIC CONFLICT RESOLUTION</span>
        </div>
      )}

      {/* Bottom Telemetry Metrics Grid & Simulation Controls */}
      {!isPresentationMode && (
        <div className="w-full pointer-events-auto flex items-end justify-between">
          {/* Engineering Telemetry Dashboard */}
          <div className="glass-panel rounded-2xl p-3.5 border border-slate-700/60 shadow-2xl grid grid-cols-7 gap-2.5 text-xs">
            <div className="px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-slate-500 block text-[10px] uppercase">Active AMRs</span>
              <span className="text-industrial-accent font-bold text-base flex items-center gap-1">
                <Bot className="w-4 h-4" /> {metrics.activeRobots} / {metrics.totalRobots}
              </span>
            </div>

            <div className="px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-slate-500 block text-[10px] uppercase">Completed Tasks</span>
              <span className="text-emerald-400 font-bold text-base flex items-center gap-1">
                <CheckCircle2 className="w-4 h-4" /> {metrics.completedTasks}
              </span>
            </div>

            <div className="px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-slate-500 block text-[10px] uppercase">Move Steps</span>
              <span className="text-slate-200 font-bold text-base">
                {metrics.movementSteps}
              </span>
            </div>

            <div className="px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-slate-500 block text-[10px] uppercase">Idle Steps</span>
              <span className="text-slate-400 font-bold text-base">
                {metrics.idleRobotSteps}
              </span>
            </div>

            <div className="px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-slate-500 block text-[10px] uppercase">Conflicts</span>
              <span className="text-rose-400 font-bold text-base">
                {metrics.totalCollisions}
              </span>
            </div>

            <div className="px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-slate-500 block text-[10px] uppercase">Overrides</span>
              <span className="text-amber-400 font-bold text-base flex items-center gap-1">
                <ShieldAlert className="w-4 h-4" /> {metrics.overrideCount}
              </span>
            </div>

            <div className="px-3 py-2 rounded-xl bg-slate-900/60 border border-slate-800">
              <span className="text-slate-500 block text-[10px] uppercase">Timestep / Ep</span>
              <span className="text-slate-200 font-bold text-base">
                #{metrics.timestep} <span className="text-[10px] text-slate-500">(Ep {metrics.episode})</span>
              </span>
            </div>
          </div>

          {/* Simulation & Replay Controls */}
          <div className="glass-panel rounded-2xl p-3 border border-slate-700/60 shadow-2xl flex items-center space-x-3 text-xs">
            <button
              onClick={onPauseToggle}
              className="p-2.5 rounded-xl bg-industrial-accent/20 border border-industrial-accent/40 text-industrial-accent hover:bg-industrial-accent hover:text-slate-950 transition-all flex items-center space-x-1.5 font-bold"
              title={isPaused ? 'Resume Simulation' : 'Pause Simulation'}
            >
              {isPaused ? (
                <>
                  <Play className="w-4 h-4 fill-current" />
                  <span>RUN</span>
                </>
              ) : (
                <>
                  <Pause className="w-4 h-4 fill-current" />
                  <span>PAUSE</span>
                </>
              )}
            </button>

            <button
              onClick={onReset}
              className="p-2.5 rounded-xl bg-slate-800 border border-slate-700 text-slate-300 hover:text-white transition-all flex items-center space-x-1.5"
              title="Reset Simulation Episode"
            >
              <RotateCcw className="w-4 h-4" />
              <span>RESET</span>
            </button>

            {/* Speed Multipliers */}
            <div className="flex items-center space-x-1 bg-slate-900 p-1 rounded-xl border border-slate-800">
              {[0.5, 1, 2, 5].map(spd => (
                <button
                  key={spd}
                  onClick={() => onSpeedChange(spd)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] transition-all ${
                    currentSpeed === spd
                      ? 'bg-industrial-accent text-slate-950 font-bold'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {spd}×
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
