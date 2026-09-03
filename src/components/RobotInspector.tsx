import React from 'react';
import { RobotState } from '../types/warehouse';
import { Bot, BatteryCharging, Navigation, ShieldCheck, ShieldAlert, Package, X, Camera } from 'lucide-react';

interface RobotInspectorProps {
  robot: RobotState | null;
  onClose: () => void;
  onFollow: (robotId: string) => void;
}

export const RobotInspector: React.FC<RobotInspectorProps> = ({ robot, onClose, onFollow }) => {
  if (!robot) return null;

  return (
    <div className="absolute top-20 left-4 z-40 w-80 glass-panel-accent rounded-2xl p-5 border border-industrial-accent/40 shadow-2xl font-mono text-xs text-slate-200 pointer-events-auto">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-700/60 pb-3 mb-4">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-industrial-accent/20 border border-industrial-accent/40 flex items-center justify-center text-industrial-accent">
            <Bot className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-bold text-sm text-white">{robot.id}</h3>
            <span className="text-[10px] text-slate-400">AUTONOMOUS MOBILE ROBOT</span>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-all"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Details Grid */}
      <div className="space-y-3">
        <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400">Status:</span>
          <span className="px-2 py-0.5 rounded-md font-bold bg-industrial-accent/20 border border-industrial-accent/40 text-industrial-accent">
            {robot.status}
          </span>
        </div>

        <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 flex items-center gap-1.5">
            <Navigation className="w-3.5 h-3.5 text-industrial-accent" /> Grid Position:
          </span>
          <span className="text-white font-bold">
            ({robot.position[0]}, {robot.position[1]})
          </span>
        </div>

        <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400">Assigned Task:</span>
          <span className="text-amber-400 font-bold">{robot.taskId || 'NONE'}</span>
        </div>

        <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 flex items-center gap-1.5">
            <Package className="w-3.5 h-3.5 text-amber-400" /> Cargo Payload:
          </span>
          <span className={`font-bold ${robot.hasCargo ? 'text-emerald-400' : 'text-slate-500'}`}>
            {robot.hasCargo ? 'LOADED' : 'EMPTY'}
          </span>
        </div>

        <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 flex items-center gap-1.5">
            <BatteryCharging className="w-3.5 h-3.5 text-emerald-400" /> Battery Telemetry:
          </span>
          <span className="text-emerald-400 font-bold">{robot.battery}%</span>
        </div>

        <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400">Planned Route Path:</span>
          <span className="text-slate-200 font-bold">{robot.path.length} cells</span>
        </div>

        <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400">Safety Status:</span>
          {robot.safetyAlert ? (
            <span className="text-amber-400 font-bold flex items-center gap-1">
              <ShieldAlert className="w-3.5 h-3.5" /> ZONE BUFFER
            </span>
          ) : (
            <span className="text-emerald-400 font-bold flex items-center gap-1">
              <ShieldCheck className="w-3.5 h-3.5" /> CLEAR
            </span>
          )}
        </div>
      </div>

      {/* Action Button */}
      <button
        onClick={() => onFollow(robot.id)}
        className="w-full mt-4 py-2.5 rounded-xl bg-industrial-accent text-slate-950 font-bold flex items-center justify-center space-x-2 shadow-lg shadow-industrial-accent/20 hover:bg-sky-300 transition-all"
      >
        <Camera className="w-4 h-4" />
        <span>TRACK ROBOT CAMERA</span>
      </button>
    </div>
  );
};
