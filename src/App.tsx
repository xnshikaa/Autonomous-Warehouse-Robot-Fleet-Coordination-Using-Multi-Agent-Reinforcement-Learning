import React, { useEffect, useRef, useState } from 'react';
import { WarehouseCanvas } from './components/WarehouseCanvas';
import { HUDOverlay } from './components/HUDOverlay';
import { RobotInspector } from './components/RobotInspector';
import { TaskInspector } from './components/TaskInspector';
import { TestRunnerPanel } from './components/TestRunnerPanel';
import { CinematicIntro } from './components/CinematicIntro';
import { PythonWebSocketProvider } from './api/PythonWebSocketProvider';
import { WarehouseSceneController } from './scene/WarehouseSceneController';
import { WarehouseState } from './types/warehouse';
import { PathVisibilityMode } from './scene/PathSystem';

export default function App() {
  const providerRef = useRef<PythonWebSocketProvider | null>(null);
  const controllerRef = useRef<WarehouseSceneController | null>(null);

  const [state, setState] = useState<WarehouseState | null>(null);
  const [selectedRobotId, setSelectedRobotId] = useState<string | null>(null);
  const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
  
  const [isPaused, setIsPaused] = useState<boolean>(false);
  const [speedMultiplier, setSpeedMultiplier] = useState<number>(1.0);
  const [isPresentationMode, setIsPresentationMode] = useState<boolean>(false);
  const [isWarehouseFullscreen, setIsWarehouseFullscreen] = useState<boolean>(false);
  const [pathMode, setPathMode] = useState<PathVisibilityMode>('ALL');
  const [showIntro, setShowIntro] = useState<boolean>(true);
  const warehouseRootRef = useRef<HTMLDivElement>(null);

  // Developer validation flag via ?dev=true URL parameter
  const isDevMode = new URLSearchParams(window.location.search).get('dev') === 'true';
  const [isTestRunnerOpen, setIsTestRunnerOpen] = useState<boolean>(isDevMode);

  useEffect(() => {
    const provider = new PythonWebSocketProvider();
    providerRef.current = provider;

    const unsubscribe = provider.subscribe(newState => {
      setState({ ...newState });
    });

    return () => {
      unsubscribe();
      provider.dispose();
    };
  }, []);

  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsWarehouseFullscreen(document.fullscreenElement === warehouseRootRef.current);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  const getFallbackProvider = () => {
    return providerRef.current ? (providerRef.current as any).fallbackProvider : null;
  };

  const handleFleetSizeChange = (size: number) => {
    providerRef.current?.setFleetSize(size);
  };

  const handlePauseToggle = () => {
    if (!providerRef.current) return;
    if (isPaused) {
      providerRef.current.resume();
      setIsPaused(false);
    } else {
      providerRef.current.pause();
      setIsPaused(true);
    }
  };

  const handleReset = () => {
    if (providerRef.current) {
      providerRef.current.reset();
      setIsPaused(false);
    }
  };

  const handleSpeedChange = (speed: number) => {
    setSpeedMultiplier(speed);
    if (providerRef.current) {
      providerRef.current.setSpeed(speed);
    }
  };

  const handleRobotFollow = (robotId: string) => {
    if (controllerRef.current) {
      controllerRef.current.focusRobot(robotId);
    }
  };

  const handleWarehouseFullscreenToggle = () => {
    if (document.fullscreenElement === warehouseRootRef.current) {
      void document.exitFullscreen();
      return;
    }

    if (warehouseRootRef.current?.requestFullscreen) {
      void warehouseRootRef.current.requestFullscreen();
    }
  };

  if (!state) {
    return (
      <div className="w-screen h-screen bg-industrial-900 flex items-center justify-center font-mono text-slate-300 text-xs">
        Loading Autonomous Warehouse Environment...
      </div>
    );
  }

  const selectedRobot = state.robots.find(r => r.id === selectedRobotId) || null;
  const selectedTask = state.tasks.find(t => t.id === selectedTaskId) || null;
  const fallbackProvider = getFallbackProvider();

  return (
    <div ref={warehouseRootRef} className="w-screen h-screen relative bg-industrial-900 text-slate-100 overflow-hidden select-none">
      {/* Cinematic Intro Overlay */}
      {showIntro && (
        <CinematicIntro onComplete={() => setShowIntro(false)} />
      )}

      {/* 3D WebGL Canvas Layer */}
      <WarehouseCanvas
        state={state}
        pathMode={pathMode}
        isWarehouseFullscreen={isWarehouseFullscreen}
        onRobotSelect={setSelectedRobotId}
        onPathModeChange={setPathMode}
        onControllerReady={ctrl => {
          controllerRef.current = ctrl;
        }}
      />

      {/* Industrial Engineering Control Dashboard HUD */}
      <HUDOverlay
        state={state}
        onFleetSizeChange={handleFleetSizeChange}
        onPauseToggle={handlePauseToggle}
        onReset={handleReset}
        onSpeedChange={handleSpeedChange}
        onPresentationModeToggle={() => setIsPresentationMode(!isPresentationMode)}
        onWarehouseFullscreenToggle={handleWarehouseFullscreenToggle}
        isPaused={isPaused}
        currentSpeed={speedMultiplier}
        isPresentationMode={isPresentationMode}
        isWarehouseFullscreen={isWarehouseFullscreen}
        pathMode={pathMode}
        onPathModeToggle={() => {
          const nextMode: Record<PathVisibilityMode, PathVisibilityMode> = {
            ALL: 'SELECTED_ONLY',
            SELECTED_ONLY: 'OFF',
            OFF: 'ALL'
          };
          setPathMode(nextMode[pathMode]);
        }}
      />

      {/* Developer-Only Validation Panel (accessible via ?dev=true) */}
      {!isPresentationMode && isTestRunnerOpen && fallbackProvider && (
        <TestRunnerPanel
          provider={fallbackProvider}
          state={state}
          onClose={() => setIsTestRunnerOpen(false)}
        />
      )}

      {/* Robot Telemetry Inspector Panel */}
      {!isPresentationMode && (
        <RobotInspector
          robot={selectedRobot}
          onClose={() => setSelectedRobotId(null)}
          onFollow={handleRobotFollow}
        />
      )}

      {/* Task Logistics Inspector Panel */}
      {!isPresentationMode && (
        <TaskInspector
          task={selectedTask}
          onClose={() => setSelectedTaskId(null)}
        />
      )}
    </div>
  );
}
