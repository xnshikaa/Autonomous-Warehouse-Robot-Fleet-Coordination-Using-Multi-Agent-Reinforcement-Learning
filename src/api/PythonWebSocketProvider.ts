import { WarehouseState, StateProvider } from '../types/warehouse';
import { DemoStateProvider } from '../state/DemoStateProvider';

export class PythonWebSocketProvider implements StateProvider {
  private ws: WebSocket | null = null;
  private subscribers: Set<(state: WarehouseState) => void> = new Set();
  private fallbackProvider: DemoStateProvider;
  private fallbackUnsubscribe: (() => void) | null = null;
  private activeState: WarehouseState;
  private url: string;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private disposed = false;

  constructor(url: string = 'ws://127.0.0.1:8000/ws') {
    this.url = url;
    this.fallbackProvider = new DemoStateProvider();
    this.activeState = this.fallbackProvider.getState();

    // Subscribe to fallback provider as default
    this.fallbackUnsubscribe = this.fallbackProvider.subscribe(state => {
      if (!this.activeState.isConnectedToBackend) {
        this.activeState = state;
        this.notifySubscribers();
      }
    });

    this.connect();
  }

  public getState(): WarehouseState {
    return this.activeState;
  }

  public subscribe(callback: (state: WarehouseState) => void): () => void {
    this.subscribers.add(callback);
    callback(this.activeState);
    return () => {
      this.subscribers.delete(callback);
    };
  }

  public setSpeed(multiplier: number): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'set_speed', speed: multiplier }));
    } else {
      this.fallbackProvider.setSpeed(multiplier);
    }
  }

  public pause(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'pause' }));
    } else {
      this.fallbackProvider.pause();
    }
  }

  public resume(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'resume' }));
    } else {
      this.fallbackProvider.resume();
    }
  }

  public reset(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'reset' }));
    } else {
      this.fallbackProvider.reset();
    }
  }

  public setFleetSize(size: number): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'set_fleet_size', size }));
    } else {
      this.fallbackProvider.setFleetSize(size);
    }
  }
  public selectRobot(robotId: string | null): void {
    this.fallbackProvider.selectRobot(robotId);
  }

  public selectTask(taskId: string | null): void {
    this.fallbackProvider.selectTask(taskId);
  }

  public dispose(): void {
    this.disposed = true;
    if (this.reconnectTimer !== null) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }
    this.fallbackUnsubscribe?.();
    this.fallbackUnsubscribe = null;
    this.fallbackProvider.dispose();
    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }
  }

  private connect(): void {
    if (this.disposed) return;
    try {
      this.ws = new WebSocket(this.url);

      this.ws.onopen = () => {
        console.log(`[WebSocket] Connected to Python RL Backend at ${this.url}`);
        this.fallbackProvider.pause();
      };

      this.ws.onmessage = event => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.type === 'state_update' || payload.robots) {
            this.activeState = {
              ...payload,
              isConnectedToBackend: true,
              isDemoMode: false
            };
            this.notifySubscribers();
          }
        } catch (e) {
          console.error('[WebSocket] Failed to parse message:', e);
        }
      };

      this.ws.onerror = err => {
        console.warn('[WebSocket] Backend connection error. Using local DemoMode fallback.');
        this.activeState.isConnectedToBackend = false;
        this.activeState.isDemoMode = true;
        this.fallbackProvider.resume();
        this.notifySubscribers();
      };

      this.ws.onclose = () => {
        if (this.disposed) return;
        console.warn('[WebSocket] Backend disconnected. Reconnecting in 5s...');
        this.activeState.isConnectedToBackend = false;
        this.activeState.isDemoMode = true;
        this.fallbackProvider.resume();
        this.notifySubscribers();
        this.reconnectTimer = setTimeout(() => {
          this.reconnectTimer = null;
          this.connect();
        }, 5000);
      };
    } catch (e) {
      console.warn('[WebSocket] Unable to instantiate WebSocket. Using DemoMode.');
    }
  }

  private notifySubscribers(): void {
    this.subscribers.forEach(cb => cb(this.activeState));
  }
}
