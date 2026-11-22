import { WarehouseState } from '../types/warehouse';

type Connection = 'connecting' | 'live' | 'stale' | 'disconnected';
export type LiveAlgorithm = 'QMIX' | 'IQL' | 'Rule-Based';
export interface WarehouseFeed {
  state: WarehouseState | null;
  connection: Connection;
  error: string | null;
  pending: boolean;
}
type Command =
  | { type: 'start' | 'pause' | 'resume' | 'reset' }
  | { type: 'set_speed'; speed: number }
  | { type: 'set_fleet_size'; size: number }
  | { type: 'set_algorithm'; algorithm: LiveAlgorithm };

export class PythonWebSocketProvider {
  private socket: WebSocket | null = null;
  private feed: WarehouseFeed = { state: null, connection: 'connecting', error: null, pending: false };
  private retry: ReturnType<typeof setTimeout> | undefined;
  private watchdog: ReturnType<typeof setTimeout> | undefined;
  private disposed = false;

  constructor(private notify: (feed: WarehouseFeed) => void,
              private url = import.meta.env.VITE_WAREHOUSE_WS_URL || 'ws://127.0.0.1:8000/ws') {
    this.connect();
  }

  private update(changes: Partial<WarehouseFeed>) {
    this.feed = { ...this.feed, ...changes };
    this.notify(this.feed);
  }

  send(command: Command) {
    if (this.feed.connection !== 'live' || this.feed.pending || this.socket?.readyState !== WebSocket.OPEN) return;
    this.socket.send(JSON.stringify(command));
    this.update({ pending: true, error: null });
  }

  private connect() {
    if (this.disposed) return;
    this.update({ connection: this.feed.state ? 'stale' : 'connecting' });
    try {
      const socket = new WebSocket(this.url);
      this.socket = socket;
      const refreshTimeout = () => {
        clearTimeout(this.watchdog);
        this.watchdog = setTimeout(() => socket.close(), 6000);
      };
      refreshTimeout();
      socket.onmessage = event => {
        try {
          const payload = JSON.parse(event.data);
          if (payload.type === 'command_error') {
            this.update({ pending: false, error: payload.error.message });
          } else if (payload.type === 'state_update' && Array.isArray(payload.robots)) {
            refreshTimeout();
            this.update({ state: payload, connection: 'live', pending: false });
          }
        } catch {
          this.update({ error: 'The backend sent unreadable telemetry.' });
        }
      };
      socket.onerror = () => socket.close();
      socket.onclose = () => this.reconnect();
    } catch {
      this.reconnect();
    }
  }

  private reconnect() {
    clearTimeout(this.watchdog);
    if (this.disposed) return;
    this.update({ connection: this.feed.state ? 'stale' : 'disconnected', pending: false });
    this.retry = setTimeout(() => this.connect(), 1500);
  }

  dispose() {
    this.disposed = true;
    clearTimeout(this.retry);
    clearTimeout(this.watchdog);
    if (this.socket) {
      this.socket.onclose = this.socket.onmessage = this.socket.onerror = null;
      this.socket.close();
    }
  }
}
