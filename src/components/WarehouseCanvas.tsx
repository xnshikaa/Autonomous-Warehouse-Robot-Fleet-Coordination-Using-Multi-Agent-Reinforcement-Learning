import React, { useEffect, useRef, useState } from 'react';
import { WarehouseSceneController } from '../scene/WarehouseSceneController';
import { CameraMode, WarehouseState } from '../types/warehouse';
import { PathVisibilityMode } from '../scene/PathSystem';
import { Camera, Eye, Orbit, Route } from 'lucide-react';

interface WarehouseCanvasProps {
  state: WarehouseState;
  onRobotSelect: (robotId: string | null) => void;
  onControllerReady?: (controller: WarehouseSceneController) => void;
  onFpsUpdate?: (fps: number) => void;
}

export const WarehouseCanvas: React.FC<WarehouseCanvasProps> = ({ 
  state, 
  onRobotSelect, 
  onControllerReady, 
  onFpsUpdate 
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const controllerRef = useRef<WarehouseSceneController | null>(null);
  const [currentCameraMode, setCurrentCameraMode] = useState<CameraMode>('OVERVIEW');
  const [pathMode, setPathMode] = useState<PathVisibilityMode>('ALL');
  const [fps, setFps] = useState<number>(60);

  useEffect(() => {
    if (!containerRef.current) return;

    const controller = new WarehouseSceneController(containerRef.current);
    controllerRef.current = controller;

    controller.onRobotSelected = (robotId) => {
      onRobotSelect(robotId);
    };

    if (onControllerReady) {
      onControllerReady(controller);
    }

    const fpsInterval = setInterval(() => {
      if (controllerRef.current) {
        const currentFps = controllerRef.current.getFPS();
        setFps(currentFps);
        if (onFpsUpdate) onFpsUpdate(currentFps);
      }
    }, 500);

    return () => {
      clearInterval(fpsInterval);
      if (controllerRef.current) {
        controllerRef.current.dispose();
        controllerRef.current = null;
      }
    };
  }, []);

  // Sync scene with authoritative WarehouseState
  useEffect(() => {
    if (controllerRef.current) {
      controllerRef.current.updateState(state);
    }
  }, [state]);

  const handleCameraChange = (mode: CameraMode) => {
    if (controllerRef.current) {
      controllerRef.current.cameraManager.setCameraMode(mode);
      setCurrentCameraMode(mode);
    }
  };

  const handlePathModeToggle = () => {
    const nextMode: Record<PathVisibilityMode, PathVisibilityMode> = {
      ALL: 'SELECTED_ONLY',
      SELECTED_ONLY: 'OFF',
      OFF: 'ALL'
    };
    const newMode = nextMode[pathMode];
    setPathMode(newMode);
    if (controllerRef.current) {
      controllerRef.current.setPathVisibilityMode(newMode);
    }
  };

  return (
    <div className="w-full h-full relative overflow-hidden bg-industrial-900 select-none">
      {/* 3D Canvas Mount Point */}
      <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Floating Toolbar */}
      <div className="absolute top-4 right-4 z-20 flex items-center space-x-2 pointer-events-auto">
        <button
          onClick={handlePathModeToggle}
          className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-xl text-xs font-mono transition-all glass-panel border border-slate-700/60 shadow-xl ${
            pathMode !== 'OFF'
              ? 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10 font-bold'
              : 'text-slate-400 hover:text-white'
          }`}
          title="Toggle 3D Robot Route Paths (ALL / SELECTED / OFF)"
        >
          <Route className="w-3.5 h-3.5" />
          <span>PATHS: {pathMode}</span>
        </button>

        <div className="flex items-center space-x-1.5 glass-panel p-1.5 rounded-xl border border-slate-700/60 shadow-xl">
          <button
            onClick={() => handleCameraChange('OVERVIEW')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
              currentCameraMode === 'OVERVIEW'
                ? 'bg-industrial-accent text-slate-950 font-bold shadow-lg shadow-industrial-accent/20'
                : 'text-slate-300 hover:bg-slate-800/80 hover:text-white'
            }`}
          >
            <Camera className="w-3.5 h-3.5" />
            <span>OVERVIEW</span>
          </button>

          <button
            onClick={() => handleCameraChange('TOP_DOWN')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
              currentCameraMode === 'TOP_DOWN'
                ? 'bg-industrial-accent text-slate-950 font-bold shadow-lg shadow-industrial-accent/20'
                : 'text-slate-300 hover:bg-slate-800/80 hover:text-white'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>TOP-DOWN</span>
          </button>

          <button
            onClick={() => handleCameraChange('FREE_ORBIT')}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
              currentCameraMode === 'FREE_ORBIT'
                ? 'bg-industrial-accent text-slate-950 font-bold shadow-lg shadow-industrial-accent/20'
                : 'text-slate-300 hover:bg-slate-800/80 hover:text-white'
            }`}
          >
            <Orbit className="w-3.5 h-3.5" />
            <span>FREE ORBIT</span>
          </button>
        </div>
      </div>

      {/* Real-time FPS Telemetry Widget */}
      <div className="absolute bottom-4 right-4 z-20 glass-panel px-3 py-1.5 rounded-lg border border-slate-800 text-[11px] font-mono flex items-center space-x-2 text-slate-400">
        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
        <span>RENDER ENGINE</span>
        <span className="text-emerald-400 font-bold">{fps} FPS</span>
      </div>
    </div>
  );
};
