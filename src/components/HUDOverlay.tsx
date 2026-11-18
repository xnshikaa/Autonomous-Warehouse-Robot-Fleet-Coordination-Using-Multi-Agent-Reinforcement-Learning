import React from 'react';
import { 
  WarehouseState
} from '../types/warehouse';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  ShieldAlert, 
  Bot, 
  CheckCircle2, 
  Maximize2,
  Minimize2,
  Layers,
  Zap, 
  AlertTriangle,
  Radio,
  Route
} from 'lucide-react';
import { PathVisibilityMode } from '../scene/PathSystem';

interface HUDOverlayProps {
  state: WarehouseState;
  onFleetSizeChange: (size: number) => void;
  onPauseToggle: () => void;
  onReset: () => void;
  onSpeedChange: (speed: number) => void;
  onPresentationModeToggle: () => void;
  onWarehouseFullscreenToggle: () => void;
  isPaused: boolean;
  currentSpeed: number;
  isPresentationMode: boolean;
  isWarehouseFullscreen: boolean;
  pathMode: PathVisibilityMode;
  onPathModeToggle: () => void;
}

export const HUDOverlay: React.FC<HUDOverlayProps> = ({
  state,
  onFleetSizeChange,
  onPauseToggle,
  onReset,
  onSpeedChange,
  onPresentationModeToggle,
  onWarehouseFullscreenToggle,
  isPaused,
  currentSpeed,
  isPresentationMode,
  isWarehouseFullscreen,
  pathMode,
  onPathModeToggle
}) => {
  const { metrics, isConnectedToBackend, safetyOverrideActive } = state;
  const [showLegend, setShowLegend] = React.useState(true);
  const isLiveBackend = isConnectedToBackend && !state.isDemoMode && state.controllerType === 'RL_BACKEND';
  const policyLabel = isLiveBackend
    ? `${state.policy?.source ?? state.algorithm} / ${state.algorithm}`
    : 'DEMO CONTROLLER / LOCAL SIM';
  const targetLabel = isLiveBackend
    ? `${state.controllerType} / ${state.algorithm}`
    : 'DEMO CONTROLLER / LOCAL SIM';
  const latestSafetyEvent = [...state.recentCollisions].sort((a, b) => b.timestamp - a.timestamp)[0];
  const activeSafetyRobotIds = state.robots
    .filter(robot => robot.safetyAlert || robot.status === 'COLLISION')
    .map(robot => robot.id);
  const safetyRobotIds = latestSafetyEvent?.robots?.length
    ? latestSafetyEvent.robots
    : activeSafetyRobotIds;

  return (
    <div className="absolute inset-0 pointer-events-none z-30 flex flex-col p-3 sm:p-4 font-mono text-xs">
      {/* Top Header Bar */}
      <div className="flex items-start justify-between gap-3 w-full pointer-events-auto flex-wrap">
        <div className="glass-panel rounded-2xl p-3 sm:p-4 border border-slate-700/60 shadow-2xl flex flex-1 min-w-0 items-center gap-3 sm:gap-5 flex-wrap">
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
          <div className="flex items-center gap-2 text-[10px] sm:text-[11px] flex-wrap">
            <span className="px-2.5 py-1 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-bold">
              MODE: SIMULATION
            </span>

            <span className="px-2.5 py-1 rounded-lg bg-slate-900 border border-slate-700/80 text-slate-300">
              POLICY: {policyLabel}
            </span>

            <span className="px-2.5 py-1 rounded-lg bg-indigo-500/10 border border-indigo-500/30 text-indigo-300 font-bold">
              TARGET: {targetLabel}
            </span>
          </div>

          <div className="h-8 w-[1px] bg-slate-800" />

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

      </div>

      {/* Row 2: all simulation controls stay together above the warehouse. */}
      <div className="w-full pointer-events-auto flex items-center justify-between gap-2 sm:gap-3 mt-2 sm:mt-3 flex-wrap">
        <div className="glass-panel rounded-2xl p-2 border border-slate-700/60 shadow-2xl flex flex-1 min-w-0 items-center gap-2 sm:gap-3 flex-wrap">
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

          <button
            onClick={onPauseToggle}
            className="px-2.5 py-2 rounded-xl bg-industrial-accent/20 border border-industrial-accent/40 text-industrial-accent hover:bg-industrial-accent hover:text-slate-950 transition-all flex items-center gap-1.5 font-bold"
            title={isPaused ? 'Resume Simulation' : 'Pause Simulation'}
          >
            {isPaused ? <Play className="w-4 h-4 fill-current" /> : <Pause className="w-4 h-4 fill-current" />}
            <span>{isPaused ? 'RUN' : 'PAUSE'}</span>
          </button>

          <button
            onClick={onReset}
            className="px-2.5 py-2 rounded-xl bg-slate-800 border border-slate-700 text-slate-300 hover:text-white transition-all flex items-center gap-1.5"
            title="Reset Simulation Episode"
          >
            <RotateCcw className="w-4 h-4" />
            <span>RESET</span>
          </button>

          <div className="flex items-center space-x-1 bg-slate-900 p-1 rounded-xl border border-slate-800">
            <span className="px-1 text-[10px] uppercase text-slate-500">Speed</span>
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

          <button
            onClick={() => setShowLegend(!showLegend)}
            className={`px-2.5 py-2 rounded-xl border transition-all flex items-center gap-1.5 ${
              showLegend
                ? 'border-industrial-accent/40 bg-industrial-accent/15 text-industrial-accent'
                : 'border-slate-700 bg-slate-800 text-slate-300'
            }`}
            title="Toggle warehouse legend"
          >
            <Layers className="w-4 h-4" />
            <span>LEGEND</span>
          </button>

          <button
            onClick={onPathModeToggle}
            className={`px-2.5 py-2 rounded-xl border transition-all flex items-center gap-1.5 ${
              pathMode !== 'OFF'
                ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-400 font-bold'
                : 'border-slate-700 bg-slate-800 text-slate-300'
            }`}
            title="Toggle 3D Robot Route Paths (ALL / SELECTED / OFF)"
          >
            <Route className="w-4 h-4" />
            <span>PATHS: {pathMode}</span>
          </button>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={onPresentationModeToggle}
            className={`px-3 py-2.5 rounded-2xl transition-all glass-panel border border-slate-700/60 shadow-2xl text-slate-300 hover:text-white ${
              isPresentationMode ? 'bg-industrial-accent text-slate-950 font-bold' : ''
            }`}
          >
            PRESENTATION
          </button>
          <button
            onClick={onWarehouseFullscreenToggle}
            className="flex items-center gap-2 px-3 py-2.5 rounded-2xl transition-all glass-panel border border-slate-700/60 shadow-2xl text-slate-300 hover:text-white"
            title={isWarehouseFullscreen ? 'Exit warehouse fullscreen' : 'Open warehouse fullscreen'}
          >
            {isWarehouseFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
            <span>{isWarehouseFullscreen ? 'EXIT FULLSCREEN' : 'WAREHOUSE FULLSCREEN'}</span>
          </button>
        </div>
      </div>

      {/* Bottom Telemetry Metrics Grid */}
      {!isPresentationMode && (
        <div className="w-full pointer-events-auto flex items-end justify-between gap-3 mt-auto flex-col lg:flex-row">
          {/* Engineering Telemetry Dashboard */}
          <div className="glass-panel rounded-2xl p-2.5 sm:p-3.5 border border-slate-700/60 shadow-2xl grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-7 gap-2 text-xs flex-1 min-w-0 w-full">
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

          <div className={`glass-panel rounded-2xl p-3.5 border shadow-2xl w-full lg:w-auto min-w-0 lg:min-w-[250px] max-w-none lg:max-w-[320px] ${
            safetyOverrideActive || latestSafetyEvent
              ? 'border-amber-500/50 bg-amber-500/10'
              : 'border-slate-700/60'
          }`}>
            <div className="flex items-center gap-2 text-[10px] uppercase tracking-wider font-bold text-slate-400">
              <AlertTriangle className={`w-4 h-4 ${safetyOverrideActive || latestSafetyEvent ? 'text-amber-400' : 'text-emerald-400'}`} />
              Safety status
            </div>
            <div className={`mt-1 font-bold ${safetyOverrideActive ? 'text-amber-300' : 'text-slate-200'}`}>
              {safetyOverrideActive ? 'OVERRIDE ACTIVE — WAIT' : latestSafetyEvent ? 'RECENT CONFLICT' : 'CLEAR'}
            </div>
            <div className="mt-1 text-[10px] text-slate-400 truncate">
              {latestSafetyEvent
                ? `${latestSafetyEvent.type.replace(/_/g, ' ')} · ${safetyRobotIds.join(', ') || 'IDs unavailable'}`
                : 'No recent safety events'}
            </div>
          </div>

        </div>
      )}

      {/* Legend stays outside the telemetry grid so the grid remains readable. */}
      {showLegend && (
        <div className="absolute top-[18rem] sm:top-36 right-4 pointer-events-auto glass-panel rounded-xl p-3 border border-slate-700/60 shadow-2xl min-w-[190px]">
          <div className="text-[10px] uppercase tracking-wider text-slate-400 font-bold mb-2">Warehouse Legend</div>
          <div className="space-y-1.5 text-[11px]">
            <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />Robot / active AMR</div>
            <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-sm bg-amber-400" />Shelf / storage</div>
            <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-sm bg-emerald-400" />Task delivery zone</div>
            <div className="flex items-center gap-2"><span className="w-2.5 h-2.5 rounded-full bg-rose-400" />Safety warning / conflict</div>
          </div>
        </div>
      )}
    </div>
  );
};
