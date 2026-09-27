import React from 'react';
import { WarehouseState, Action, ExecutedAction } from '../types/warehouse';
import { DemoStateProvider } from '../state/DemoStateProvider';
import { StepForward, RotateCcw, AlertTriangle, ShieldAlert, X } from 'lucide-react';

interface TestRunnerPanelProps {
  provider: DemoStateProvider;
  state: WarehouseState;
  onClose: () => void;
}

const ACTION_NAMES: Record<Action, string> = {
  [Action.UP]: 'UP',
  [Action.DOWN]: 'DOWN',
  [Action.LEFT]: 'LEFT',
  [Action.RIGHT]: 'RIGHT'
};

const formatExecutedAction = (exec?: ExecutedAction): string => {
  if (exec === undefined) return 'NO_OP';
  if (exec === 'NO_OP') return 'NO_OP';
  return ACTION_NAMES[exec] || 'NO_OP';
};

export const TestRunnerPanel: React.FC<TestRunnerPanelProps> = ({ provider, state, onClose }) => {
  const testController = provider.testController;
  const currentTest = testController.getCurrentTest();
  const stepIdx = testController.getStepIdx();

  const handleTestSelect = (testId: string) => {
    provider.selectTestSuite(testId);
  };

  const handleStepOnce = () => {
    provider.stepOnce();
  };

  const handleResetTest = () => {
    provider.selectTestSuite(currentTest.id);
  };

  return (
    <div className="absolute top-20 left-4 z-40 w-96 glass-panel rounded-2xl p-4 border border-slate-700/80 shadow-2xl font-mono text-xs text-slate-200 pointer-events-auto">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <ShieldAlert className="w-4 h-4 text-industrial-accent" />
          <h2 className="font-bold text-slate-100 tracking-wide text-sm">DEVELOPER VALIDATION SUITE</h2>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-all"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Test Selector Dropdown */}
      <div className="mt-3">
        <label className="text-[10px] uppercase text-slate-400 block mb-1">Select Validation Scenario:</label>
        <select
          value={currentTest.id}
          onChange={(e) => handleTestSelect(e.target.value)}
          className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-industrial-accent"
        >
          {Object.values(testController.testSuites).map(ts => (
            <option key={ts.id} value={ts.id}>
              {ts.name}
            </option>
          ))}
        </select>
      </div>

      {/* Test Card Description */}
      <div className="mt-3 p-3 rounded-xl bg-slate-900/80 border border-slate-800/80">
        <p className="text-[11px] text-slate-300 leading-relaxed">{currentTest.description}</p>
        <div className="mt-2 flex items-center justify-between text-[10px] text-slate-400">
          <span>Step Progress: <strong className="text-industrial-accent">{stepIdx}</strong> / {currentTest.expectedSteps}</span>
          <span className="px-2 py-0.5 rounded bg-slate-800 text-slate-300">Timestep #{state.timestep}</span>
        </div>
      </div>

      {/* Action Trace & Safety Overrides Table */}
      <div className="mt-3">
        <span className="text-[10px] uppercase text-slate-400 block mb-1">Last Action Traces (Req vs Exec):</span>
        <div className="max-h-36 overflow-y-auto space-y-1.5 pr-1">
          {state.robots.map(r => (
            <div
              key={r.id}
              className={`p-2 rounded-lg border text-[11px] flex items-center justify-between ${
                r.actionOverridden
                  ? 'bg-amber-500/10 border-amber-500/40 text-amber-300'
                  : 'bg-slate-900/60 border-slate-800 text-slate-300'
              }`}
            >
              <div className="flex items-center space-x-2">
                <span className="font-bold text-slate-100">{r.id}</span>
                <span className="text-[10px] text-slate-500">[{r.position[0]},{r.position[1]}]</span>
              </div>
              <div className="flex items-center space-x-2">
                <span>Req: <strong className="text-slate-200">{r.lastRequestedAction !== undefined ? ACTION_NAMES[r.lastRequestedAction] : 'UP'}</strong></span>
                <span>$\rightarrow$</span>
                <span>Exec: <strong className={r.actionOverridden ? 'text-amber-400' : 'text-emerald-400'}>{formatExecutedAction(r.lastExecutedAction)}</strong></span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Structured Conflict Log */}
      {state.recentCollisions.length > 0 && (
        <div className="mt-3 p-2.5 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-[11px]">
          <div className="flex items-center space-x-1.5 mb-1 font-bold text-rose-400">
            <AlertTriangle className="w-3.5 h-3.5" />
            <span>Active Conflicts Logged ({state.recentCollisions.length}):</span>
          </div>
          {state.recentCollisions.map(c => (
            <div key={c.id} className="text-[10px] text-rose-200">
              • [{c.type}] Robots ({c.robots.join(', ')}) at cell ({c.position.join(',')})
            </div>
          ))}
        </div>
      )}

      {/* Step Controls */}
      <div className="mt-4 flex items-center space-x-2">
        <button
          onClick={handleStepOnce}
          className="flex-1 py-2 px-3 rounded-xl bg-industrial-accent text-slate-950 font-bold flex items-center justify-center space-x-2 shadow-lg shadow-industrial-accent/20 hover:brightness-110 transition-all"
        >
          <StepForward className="w-4 h-4" />
          <span>STEP TEST (+1)</span>
        </button>

        <button
          onClick={handleResetTest}
          className="p-2 rounded-xl bg-slate-800 border border-slate-700 text-slate-300 hover:text-white transition-all"
          title="Reset Test Case"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
};
