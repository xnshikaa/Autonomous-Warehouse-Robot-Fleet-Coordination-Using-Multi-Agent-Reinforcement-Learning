import React from 'react';
import { TaskState } from '../types/warehouse';
import { Package, MapPin, Bot, CheckCircle, X } from 'lucide-react';

interface TaskInspectorProps {
  task: TaskState | null;
  onClose: () => void;
}

export const TaskInspector: React.FC<TaskInspectorProps> = ({ task, onClose }) => {
  if (!task) return null;

  return (
    <div className="absolute top-20 right-4 z-40 w-80 glass-panel-accent rounded-2xl p-5 border border-industrial-accent/40 shadow-2xl font-mono text-xs text-slate-200 pointer-events-auto">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-700/60 pb-3 mb-4">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-amber-500/20 border border-amber-500/40 flex items-center justify-center text-amber-400">
            <Package className="w-5 h-5" />
          </div>
          <div>
            <h3 className="font-bold text-sm text-white">{task.id}</h3>
            <span className="text-[10px] text-slate-400">LOGISTICS TASK DISPATCH</span>
          </div>
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-all"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Details */}
      <div className="space-y-3">
        <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400">Task Status:</span>
          <span className="px-2 py-0.5 rounded-md font-bold bg-amber-500/20 border border-amber-500/40 text-amber-400">
            {task.status}
          </span>
        </div>

        <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 flex items-center gap-1.5">
            <MapPin className="w-3.5 h-3.5 text-amber-400" /> Pickup Location:
          </span>
          <span className="text-white font-bold">
            ({task.pickupPos[0]}, {task.pickupPos[1]}) <span className="text-slate-400">[{task.pickupShelfId || 'S01'}]</span>
          </span>
        </div>

        <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 flex items-center gap-1.5">
            <CheckCircle className="w-3.5 h-3.5 text-emerald-400" /> Destination Bay:
          </span>
          <span className="text-white font-bold">
            ({task.deliveryPos[0]}, {task.deliveryPos[1]}) <span className="text-emerald-400">[{task.deliveryZoneId || 'D01'}]</span>
          </span>
        </div>

        <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400 flex items-center gap-1.5">
            <Bot className="w-3.5 h-3.5 text-industrial-accent" /> Assigned AMR:
          </span>
          <span className="text-industrial-accent font-bold">
            {task.assignedRobotId || 'UNASSIGNED'}
          </span>
        </div>
      </div>
    </div>
  );
};
