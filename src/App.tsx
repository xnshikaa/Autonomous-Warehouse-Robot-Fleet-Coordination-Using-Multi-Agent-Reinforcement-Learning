import { useEffect, useRef, useState } from 'react';
import { Camera, Pause, Play, RotateCcw, Wifi, WifiOff } from 'lucide-react';
import { WarehouseCanvas } from './components/WarehouseCanvas';
import { LiveAlgorithm, PythonWebSocketProvider, WarehouseFeed } from './api/PythonWebSocketProvider';
import { DemoStateProvider } from './state/DemoStateProvider';
import { WarehouseSceneController } from './scene/WarehouseSceneController';
import { PathVisibilityMode } from './scene/PathSystem';
import { WarehouseState } from './types/warehouse';
import './warehouse-dashboard.css';

const algorithms: LiveAlgorithm[] = ['QMIX', 'IQL', 'Rule-Based'];

function format(value: number | null | undefined, decimals = 0) {
  return value == null ? '—' : value.toFixed(decimals);
}

// How long to wait for the WebSocket backend before giving up and staying in demo mode
const WS_CONNECT_TIMEOUT_MS = 5000;

type Mode = 'demo' | 'live';

export default function App() {
  const wsProvider = useRef<PythonWebSocketProvider | null>(null);
  const demoProvider = useRef<DemoStateProvider | null>(null);
  const scene = useRef<WarehouseSceneController | null>(null);

  // Live feed from WebSocket (null = not connected / not available)
  const [wsFeed, setWsFeed] = useState<WarehouseFeed | null>(null);
  // Demo state from JS simulation engine
  const [demoState, setDemoState] = useState<WarehouseState | null>(null);
  // Current mode — starts as 'demo', switches to 'live' when WS connects
  const [mode, setMode] = useState<Mode>('demo');
  // Whether we gave up waiting for WS (stopped trying to connect)
  const [wsGaveUp, setWsGaveUp] = useState(false);

  const [selected, setSelected] = useState<string | null>(null);
  const [following, setFollowing] = useState<string | null>(null);
  const [pathMode, setPathMode] = useState<PathVisibilityMode>('ALL');
  const [demoPaused, setDemoPaused] = useState(false);
  const [demoSpeed, setDemoSpeed] = useState(1);
  const [demoFleetSize, setDemoFleetSize] = useState(5);

  // Start the demo simulation immediately — always works without a server
  useEffect(() => {
    const provider = new DemoStateProvider(demoFleetSize);
    demoProvider.current = provider;
    const unsub = provider.subscribe(state => setDemoState(state));
    return () => {
      unsub();
      provider.dispose();
      demoProvider.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Attempt WebSocket backend connection; give up after WS_CONNECT_TIMEOUT_MS
  useEffect(() => {
    const wsUrl = import.meta.env.VITE_WAREHOUSE_WS_URL || 'ws://127.0.0.1:8000/ws';
    let mounted = true;
    let giveUpTimer: ReturnType<typeof setTimeout>;

    const provider = new PythonWebSocketProvider(feed => {
      if (!mounted) return;
      setWsFeed(feed);
      if (feed.connection === 'live') {
        clearTimeout(giveUpTimer);
        setMode('live');
        setWsGaveUp(false);
      }
      if (feed.connection === 'disconnected' && mode !== 'live') {
        // Stay in demo — backend not available
      }
    }, wsUrl);

    wsProvider.current = provider;

    // If not live within timeout, give up and stay in demo
    giveUpTimer = setTimeout(() => {
      if (mounted && mode !== 'live') {
        setWsGaveUp(true);
        provider.dispose();
        wsProvider.current = null;
      }
    }, WS_CONNECT_TIMEOUT_MS);

    return () => {
      mounted = false;
      clearTimeout(giveUpTimer);
      provider.dispose();
      wsProvider.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // If backend goes away after being live, fall back to demo
  useEffect(() => {
    if (wsFeed?.connection === 'disconnected' && mode === 'live') {
      setMode('demo');
      setWsGaveUp(true);
    }
  }, [wsFeed, mode]);

  // ── Active state (live or demo) ──────────────────────────────────────────
  const state = mode === 'live' ? wsFeed?.state ?? demoState : demoState;
  const feed = wsFeed ?? { state: null, connection: 'disconnected' as const, error: null, pending: false };
  const live = mode === 'live' && feed.connection === 'live';
  const interactive = live && !feed.pending;

  const selectedRobot = state?.robots.find(r => r.id === selected);
  const selectedTask = state?.tasks.find(t => t.id === selectedRobot?.taskId)
    ?? state?.tasks.find(t => selectedRobot && t.assignedRobotId === selectedRobot.id);

  const status = !state ? 'Starting simulation…'
    : state.simulationFinished ? 'Episode complete'
    : (state.simulationPaused || demoPaused) ? 'Paused'
    : 'Running';

  const connectionLabel = mode === 'live'
    ? { live: 'Live backend', stale: 'Stale telemetry', connecting: 'Connecting', disconnected: 'Reconnecting' }[feed.connection]
    : wsGaveUp ? 'Demo mode (no backend)' : 'Demo mode · connecting…';

  const metrics = state?.metrics;
  const checkpoint = state?.policy?.checkpointPath;
  const checkpointName = checkpoint?.split(/[\\\/]/).slice(-2).join(' / ');
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

  // ── Demo mode control helpers ────────────────────────────────────────────
  const handleDemoPause = () => {
    if (demoPaused) demoProvider.current?.resume();
    else demoProvider.current?.pause();
    setDemoPaused(p => !p);
  };

  const handleDemoReset = () => { demoProvider.current?.reset(); };

  const handleDemoSpeed = (speed: number) => {
    setDemoSpeed(speed);
    demoProvider.current?.setSpeed(speed);
  };

  const handleDemoFleet = (size: number) => {
    setDemoFleetSize(size);
    demoProvider.current?.setFleetSize(size);
  };

  // ── Live mode control helpers ────────────────────────────────────────────
  const handleLivePause = () => {
    wsProvider.current?.send({ type: state?.simulationPaused ? (state.timestep ? 'resume' : 'start') : 'pause' });
  };
  const handleLiveReset = () => { wsProvider.current?.send({ type: 'reset' }); };

  return (
    <main className="control-room">
      <header className="control-header">
        <h1>Autonomous Warehouse Robot Fleet Coordination Using Multi-Agent Reinforcement Learning</h1>
        <div className="header-signals">
          <span className={live ? 'live-signal' : 'warning-signal'} title={mode === 'demo' ? 'Running fully in-browser — no server required' : undefined}>
            {live ? <Wifi size={12} style={{ display: 'inline', marginRight: 4 }} /> : <WifiOff size={12} style={{ display: 'inline', marginRight: 4 }} />}
            {connectionLabel}
          </span>
          {state && <span>{state.algorithm} · {state.robots.length} {state.robots.length === 1 ? 'robot' : 'robots'} · {status}</span>}
        </div>
      </header>

      {/* Demo mode banner */}
      {mode === 'demo' && (
        <div className="demo-banner" role="status">
          🤖 <strong>Demo Mode</strong> — Full in-browser MARL simulation running locally.
          {!wsGaveUp && ' Trying to connect to live backend…'}
          {wsGaveUp && ' Connect a live backend via VITE_WAREHOUSE_WS_URL to switch to live mode.'}
        </div>
      )}

      <div className="control-layout">
        <section className="warehouse-stage" aria-label="3D warehouse">
          <header className="panel-heading">
            <h2>Warehouse</h2>
            <div className="button-row">
              <button type="button" onClick={() => scene.current?.resetCamera()}>
                <Camera size={16} /> Reset camera
              </button>
              <button type="button" onClick={() => scene.current?.showOverhead()}>Overhead</button>
              <select aria-label="Executed route visibility" value={pathMode} onChange={e => setPathMode(e.target.value as PathVisibilityMode)}>
                <option value="ALL">All routes</option>
                <option value="SELECTED_ONLY">Selected route</option>
                <option value="OFF">Routes off</option>
              </select>
            </div>
          </header>
          <div className="warehouse-viewport">
            {state && <WarehouseCanvas state={state} pathMode={pathMode} selectedRobot={selectedRobot?.id ?? null}
              onRobotSelect={setSelected} onControllerReady={controller => {
                scene.current = controller;
                if (controller) controller.onFollowChanged = setFollowing;
              }} />}
            {!state && <div className="stale-overlay" role="status">Starting simulation…</div>}
            {state && mode !== 'live' && (
              <div className="stale-overlay demo-mode-overlay" role="status" style={{ background: 'rgba(0,40,80,0.55)', fontSize: '0.75rem', pointerEvents: 'none' }}>
                Demo mode · In-browser simulation
              </div>
            )}
          </div>
          <div className="warehouse-legend">
            <span><i className="swatch pickup" /> Pickup</span>
            <span><i className="swatch delivery" /> Delivery</span>
            <span><i className="swatch corridor" /> Worker exclusion</span>
            <span>Robot rings show movement and safety state</span>
            <span>Lines show executed routes</span>
            <span>Drag to orbit · scroll to zoom</span>
          </div>
        </section>

        <aside className="control-sidebar">
          <section className="side-panel">
            <h2>Simulation controls</h2>
            <div className="control-fields">
              <label>Fleet size
                <select
                  disabled={mode === 'live' && !interactive}
                  value={mode === 'live' ? (state?.robots.length ?? '') : demoFleetSize}
                  onChange={e => {
                    const size = Number(e.target.value);
                    if (mode === 'live') wsProvider.current?.send({ type: 'set_fleet_size', size });
                    else handleDemoFleet(size);
                  }}>
                  {mode === 'live' && !state && <option value="">Waiting</option>}
                  {[1, 5, 10, 20].map(size => <option key={size} value={size}>{size === 1 ? '1 robot' : `${size} robots`}</option>)}
                </select>
              </label>
              <label>Algorithm
                <select
                  disabled={mode === 'live' ? (!interactive || state?.robots.length === 1) : false}
                  value={state?.algorithm ?? ''}
                  onChange={e => {
                    if (mode === 'live') wsProvider.current?.send({ type: 'set_algorithm', algorithm: e.target.value as LiveAlgorithm });
                  }}>
                  {mode === 'live' && !state && <option value="">Waiting</option>}
                  {algorithms.map(a => <option key={a}>{a}</option>)}
                  {mode === 'demo' && <option value="DEMO">Rule-Based (Demo)</option>}
                </select>
              </label>
              <label>Speed
                <select
                  value={mode === 'live' ? (state?.config.speedMultiplier ?? 1) : demoSpeed}
                  onChange={e => {
                    const speed = Number(e.target.value);
                    if (mode === 'live') wsProvider.current?.send({ type: 'set_speed', speed });
                    else handleDemoSpeed(speed);
                  }}>
                  {[0.25, 0.5, 1, 2, 4].map(s => <option key={s} value={s}>{s}×</option>)}
                </select>
              </label>
            </div>
            <div className="button-row transport">
              <button
                disabled={mode === 'live' && !interactive}
                onClick={mode === 'live' ? handleLivePause : handleDemoPause}>
                {(mode === 'demo' ? demoPaused : state?.simulationPaused)
                  ? <><Play size={16} /> Resume</>
                  : <><Pause size={16} /> Pause</>}
              </button>
              <button
                disabled={mode === 'live' && !interactive}
                onClick={mode === 'live' ? handleLiveReset : handleDemoReset}>
                <RotateCcw size={16} /> {state?.simulationFinished ? 'Next mission' : 'Reset'}
              </button>
            </div>
            {state && (
              <div className="model-status" title={checkpoint ?? undefined}>
                {mode === 'demo'
                  ? 'Demo mode · BFS rule-based controller'
                  : state.policy?.checkpointLoaded ? 'Compatible checkpoint' : 'Rule-based controller'}
                {checkpointName && <strong>{checkpointName}</strong>}
              </div>
            )}
            {feed.pending && <p role="status">Applying command…</p>}
            {(feed.error || state?.startupWarning) && <p role="alert" className="warning-signal">{feed.error || state?.startupWarning}</p>}
          </section>

          <section className="side-panel">
            <h2>Episode telemetry</h2>
            {state && <p className="episode-line">Episode {state.episode} · Step {state.timestep}{mode === 'demo' ? ' · Demo' : !live ? ' · Last received' : ''}</p>}
            {state?.missionSeed && <p className="episode-line">Single-robot mission · Seed {state.missionSeed}</p>}
            {state?.referenceSeed && <p className="episode-line">Reference scenario · Seed {state.referenceSeed}</p>}
            <dl className="kpi-grid">{kpis.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
          </section>

          <section className="side-panel">
            <h2>Robot inspection</h2>
            {following && <p className="episode-line" role="status">Following {following}</p>}
            <select aria-label="Select robot" value={selectedRobot?.id ?? ''} onChange={e => setSelected(e.target.value || null)}>
              <option value="">Select a robot</option>
              {state?.robots.map(r => <option key={r.id}>{r.id}</option>)}
            </select>
            {selectedRobot && <dl className="robot-details">
              <div><dt>Position</dt><dd>{selectedRobot.position.join(', ')}</dd></div>
              <div><dt>State</dt><dd>{selectedRobot.status}</dd></div>
              <div><dt>Task</dt><dd>{selectedTask ? `${selectedTask.id} · ${selectedTask.status.replace(/_/g, ' ').toLowerCase()}` : 'Unassigned'}</dd></div>
              {selectedTask && <>
                <div><dt>Pickup cell</dt><dd>{selectedTask.pickupPos.join(', ')}{selectedTask.pickupTimestep != null ? ` · step ${selectedTask.pickupTimestep}` : ''}</dd></div>
                <div><dt>Delivery cell</dt><dd>{selectedTask.deliveryPos.join(', ')}{selectedTask.completedTimestep != null ? ` · step ${selectedTask.completedTimestep}` : ''}</dd></div>
              </>}
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
