import { useEffect, useRef, useState } from 'react';
import { Camera, Pause, Play, RotateCcw } from 'lucide-react';
import { WarehouseCanvas } from './components/WarehouseCanvas';
import { LiveAlgorithm, PythonWebSocketProvider, WarehouseFeed } from './api/PythonWebSocketProvider';
import { WarehouseSceneController } from './scene/WarehouseSceneController';
import { PathVisibilityMode } from './scene/PathSystem';
import './warehouse-dashboard.css';

const algorithms: LiveAlgorithm[] = ['QMIX', 'IQL', 'Rule-Based'];
function format(value: number | null | undefined, decimals = 0) {
  return value == null ? '—' : value.toFixed(decimals);
}

export default function App() {
  const provider = useRef<PythonWebSocketProvider | null>(null);
  const scene = useRef<WarehouseSceneController | null>(null);
  const [feed, setFeed] = useState<WarehouseFeed>({ state: null, connection: 'connecting', error: null, pending: false });
  const [selected, setSelected] = useState<string | null>(null);
  const [pathMode, setPathMode] = useState<PathVisibilityMode>('ALL');

  useEffect(() => {
    let mounted = true;
    queueMicrotask(() => {
      if (mounted) provider.current = new PythonWebSocketProvider(setFeed);
    });
    return () => {
      mounted = false;
      provider.current?.dispose();
      provider.current = null;
    };
  }, []);

  const state = feed.state;
  const selectedRobot = state?.robots.find(robot => robot.id === selected);
  const live = feed.connection === 'live';
  const interactive = live && !feed.pending;
  const status = !state ? 'Awaiting telemetry' : state.simulationFinished ? 'Episode complete'
    : state.simulationPaused ? 'Paused' : 'Running';
  const connection = { live: 'Live backend', stale: 'Stale telemetry', connecting: 'Connecting', disconnected: 'Disconnected' }[feed.connection];
  const metrics = state?.metrics;
  const checkpoint = state?.policy?.checkpointPath;
  const checkpointName = checkpoint?.split(/[\\/]/).slice(-2).join(' / ');
  const kpis = [
    ['Tasks completed', metrics ? format(metrics.completedTasks) : '—'],
    ['Pickups', format(metrics?.pickups)],
    ['Completion', metrics?.completionRate == null ? '—' : format(metrics.completionRate * 100, 1) + '%'],
    ['Avg. delivery', metrics?.averageDeliveryTimeSteps == null ? '—' : format(metrics.averageDeliveryTimeSteps, 1) + ' steps'],
    ['Tasks / 100 steps', format(metrics?.throughputPer100Steps, 1)],
    ['Physical collisions', format(metrics?.physicalCollisions)],
    ['Safety conflicts', format(metrics?.safetyConflictEvents)],
    ['Blocked move groups', format(metrics?.blockedConflictGroups)],
    ['Safety overrides', format(metrics?.safetyOverrides)],
    ['Blocked moves', format(metrics?.blockedMoves)],
    ['Idle robots', format(metrics?.idleRobots)],
  ];

  return (
    <main className="control-room">
      <header className="control-header">
        <h1>Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning</h1>
        <div className="header-signals">
          <span className={live ? 'live-signal' : 'warning-signal'}>{connection}</span>
          {live && state && <span>{state.algorithm} · {state.robots.length} {state.robots.length === 1 ? 'robot' : 'robots'} · {status}</span>}
        </div>
      </header>
      <div className="control-layout">
        <section className="warehouse-stage" aria-label="3D warehouse">
          <header className="panel-heading">
            <h2>Warehouse</h2>
            <div className="button-row">
              <button type="button" onClick={() => scene.current?.cameraManager.setCameraMode('OVERVIEW')}>
                <Camera size={16} /> Reset camera
              </button>
              <button type="button" onClick={() => scene.current?.cameraManager.setCameraMode('TOP_DOWN')}>Overhead</button>
              <select aria-label="Executed route visibility" value={pathMode} onChange={e => setPathMode(e.target.value as PathVisibilityMode)}>
                <option value="ALL">All routes</option><option value="SELECTED_ONLY">Selected route</option><option value="OFF">Routes off</option>
              </select>
            </div>
          </header>
          <div className="warehouse-viewport">
            {state && <WarehouseCanvas state={state} pathMode={pathMode} selectedRobot={selectedRobot?.id ?? null}
              onRobotSelect={setSelected} onControllerReady={controller => { scene.current = controller; }} />}
            {!live && <div className="stale-overlay" role="status">{state ? 'Connection lost · showing last received frame' : connection + ' · waiting for warehouse state'}</div>}
          </div>
          <div className="warehouse-legend">
            <span><i className="swatch pickup" /> Pickup</span><span><i className="swatch delivery" /> Delivery</span>
            <span><i className="swatch corridor" /> Worker exclusion</span><span>Robot rings show movement and safety state</span>
            <span>Lines show executed routes</span>
            <span>Drag to orbit · scroll to zoom</span>
          </div>
        </section>

        <aside className="control-sidebar">
          <section className="side-panel">
            <h2>Simulation controls</h2>
            <div className="control-fields">
              <label>Fleet size<select disabled={!interactive} value={state?.robots.length ?? ''} onChange={e => provider.current?.send({ type: 'set_fleet_size', size: Number(e.target.value) })}>
                {!state && <option value="">Waiting</option>}
                {[1, 5, 10, 20].map(size => <option key={size} value={size}>{size === 1 ? '1 robot' : `${size} robots`}</option>)}
              </select></label>
              <label>Algorithm<select disabled={!interactive || state?.robots.length === 1} value={state?.algorithm ?? ''} onChange={e => provider.current?.send({ type: 'set_algorithm', algorithm: e.target.value as LiveAlgorithm })}>
                {!state && <option value="">Waiting</option>}
                {algorithms.map(algorithm => <option key={algorithm}>{algorithm}</option>)}
              </select></label>
              <label>Speed<select disabled={!interactive} value={state?.config.speedMultiplier ?? 1} onChange={e => provider.current?.send({ type: 'set_speed', speed: Number(e.target.value) })}>
                {[0.25, 0.5, 1, 2, 4].map(speed => <option key={speed} value={speed}>{speed}×</option>)}
              </select></label>
            </div>
            <div className="button-row transport">
              <button disabled={!interactive || state?.simulationFinished} onClick={() => provider.current?.send({ type: state?.simulationPaused ? (state.timestep ? 'resume' : 'start') : 'pause' })}>
                {state?.simulationPaused ? <Play size={16} /> : <Pause size={16} />}
                {state?.simulationPaused ? (state.timestep ? 'Resume' : 'Start') : 'Pause'}
              </button>
              <button disabled={!interactive} onClick={() => provider.current?.send({ type: 'reset' })}><RotateCcw size={16} /> {state?.missionSeed && state.simulationFinished ? 'Next mission' : 'Reset'}</button>
            </div>
            {state && <div className="model-status" title={checkpoint ?? undefined}>
              {state.policy?.checkpointLoaded ? 'Compatible checkpoint' : 'Rule-based controller'}{checkpointName && <strong>{checkpointName}</strong>}
            </div>}
            {feed.pending && <p role="status">Applying command…</p>}
            {(feed.error || state?.startupWarning) && <p role="alert" className="warning-signal">{feed.error || state?.startupWarning}</p>}
          </section>

          <section className="side-panel">
            <h2>Episode telemetry</h2>
            {state && <p className="episode-line">Episode {state.episode} · Step {state.timestep}{!live && ' · Last received'}</p>}
            {state?.missionSeed && <p className="episode-line">Single-robot mission · Seed {state.missionSeed}</p>}
            <dl className="kpi-grid">{kpis.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
          </section>

          <section className="side-panel">
            <h2>Robot inspection</h2>
            <select aria-label="Select robot" value={selectedRobot?.id ?? ''} onChange={e => setSelected(e.target.value || null)}>
              <option value="">Select a robot</option>
              {state?.robots.map(robot => <option key={robot.id}>{robot.id}</option>)}
            </select>
            {selectedRobot && <dl className="robot-details">
              <div><dt>Position</dt><dd>{selectedRobot.position.join(', ')}</dd></div>
              <div><dt>State</dt><dd>{selectedRobot.status}</dd></div>
              <div><dt>Task</dt><dd>{selectedRobot.taskId ?? 'Unassigned'}</dd></div>
              <div><dt>Cargo</dt><dd>{selectedRobot.hasCargo ? 'Carrying' : 'Empty'}</dd></div>
              <div><dt>Target</dt><dd>{selectedRobot.targetPosition?.join(', ') ?? '—'}</dd></div>
              <div><dt>Last successful move</dt><dd>{selectedRobot.lastSuccessfulMove ? `${selectedRobot.lastSuccessfulMove.from.join(', ')} → ${selectedRobot.lastSuccessfulMove.to.join(', ')}` : '—'}</dd></div>
              <div><dt>Navigation</dt><dd>{selectedRobot.blockedReason?.replace(/_/g, ' ') ?? (selectedRobot.replanning ? 'Replanning' : selectedRobot.oscillating ? 'Repeated movement' : '—')}</dd></div>
              <div><dt>Remaining route</dt><dd>{selectedRobot.plannedPath?.length ? `${selectedRobot.plannedPath.length - 1} cells` : '—'}</dd></div>
            </dl>}
            {!!selectedRobot?.plannedPath?.length && <details>
              <summary>Planned cells</summary>
              <p>{selectedRobot.plannedPath.map(cell => cell.join(', ')).join(' → ')}</p>
            </details>}
            {selectedRobot && <div className="button-row">
              <button type="button" onClick={() => scene.current?.focusRobot(selectedRobot.id)}>Follow selected robot</button>
              <button type="button" onClick={() => scene.current?.stopFollowing()}>Stop following</button>
            </div>}
          </section>
        </aside>
      </div>
    </main>
  );
}
