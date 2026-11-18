import React, { useEffect, useRef, useState } from 'react';
import { WarehouseSceneController } from '../scene/WarehouseSceneController';
import { WarehouseState } from '../types/warehouse';
import { PathVisibilityMode } from '../scene/PathSystem';

interface WarehouseCanvasProps {
  state: WarehouseState;
  pathMode: PathVisibilityMode;
  isWarehouseFullscreen: boolean;
  onRobotSelect: (robotId: string | null) => void;
  onControllerReady?: (controller: WarehouseSceneController) => void;
  onFpsUpdate?: (fps: number) => void;
  onPathModeChange?: (mode: PathVisibilityMode) => void;
}

export const WarehouseCanvas: React.FC<WarehouseCanvasProps> = ({ 
  state, 
  pathMode,
  isWarehouseFullscreen,
  onRobotSelect, 
  onControllerReady, 
  onFpsUpdate,
  onPathModeChange
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const controllerRef = useRef<WarehouseSceneController | null>(null);
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

  useEffect(() => {
    controllerRef.current?.setPathVisibilityMode(pathMode);
    onPathModeChange?.(pathMode);
  }, [pathMode, onPathModeChange]);

  return (
    <div className={`absolute inset-x-0 overflow-hidden bg-industrial-900 select-none ${
      isWarehouseFullscreen
        ? 'inset-y-0'
        : 'top-[18rem] bottom-[18rem] sm:top-[14rem] sm:bottom-[14rem] lg:top-[10rem] lg:bottom-[9rem]'
    }`}>
      {/* 3D Canvas Mount Point */}
      <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Real-time FPS Telemetry Widget */}
      <div className={`absolute right-4 z-20 glass-panel px-3 py-1.5 rounded-lg border border-slate-800 text-[11px] font-mono flex items-center space-x-2 text-slate-400 ${
        isWarehouseFullscreen ? 'bottom-4' : 'top-2'
      }`}>
        <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
        <span>RENDER ENGINE</span>
        <span className="text-emerald-400 font-bold">{fps} FPS</span>
      </div>
    </div>
  );
};
