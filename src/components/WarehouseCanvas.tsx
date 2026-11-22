import { useEffect, useRef, useState } from 'react';
import { WarehouseSceneController } from '../scene/WarehouseSceneController';
import { WarehouseState } from '../types/warehouse';
import { PathVisibilityMode } from '../scene/PathSystem';

interface WarehouseCanvasProps {
  state: WarehouseState;
  pathMode: PathVisibilityMode;
  selectedRobot: string | null;
  onRobotSelect: (robotId: string | null) => void;
  onControllerReady: (controller: WarehouseSceneController | null) => void;
}

export function WarehouseCanvas(props: WarehouseCanvasProps) {
  const mount = useRef<HTMLDivElement>(null);
  const controller = useRef<WarehouseSceneController | null>(null);
  const callbacks = useRef(props);
  callbacks.current = props;
  const [renderError, setRenderError] = useState(false);

  useEffect(() => {
    if (!mount.current) return;
    let scene: WarehouseSceneController;
    try {
      scene = new WarehouseSceneController(mount.current);
    } catch {
      setRenderError(true);
      return;
    }
    controller.current = scene;
    scene.onRobotSelected = id => callbacks.current.onRobotSelect(id);
    callbacks.current.onControllerReady(scene);
    return () => {
      scene.dispose();
      controller.current = null;
      callbacks.current.onControllerReady(null);
    };
  }, []);

  useEffect(() => {
    controller.current?.setPathVisibilityMode(props.pathMode);
    controller.current?.pathSystem.setSelectedRobot(props.selectedRobot);
    controller.current?.updateState(props.state);
  }, [props.state, props.pathMode, props.selectedRobot]);

  return <>
    <div ref={mount} className="absolute inset-0 cursor-grab active:cursor-grabbing" />
    {renderError && <div className="stale-overlay" role="alert">The browser could not initialize WebGL. Live telemetry is available in the control panel.</div>}
  </>;
}
