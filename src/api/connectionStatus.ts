export type BackendConnectionLabel = 'LIVE BACKEND' | 'DEMO CONTROLLER' | 'DISCONNECTED';

export function markBackendConnected<T extends { isDemoMode: boolean }>(payload: T): T & {
  isConnectedToBackend: true;
} {
  return { ...payload, isConnectedToBackend: true };
}

export function markLocalFallback<T extends object>(state: T): T & {
  isConnectedToBackend: false;
  isDemoMode: true;
} {
  return { ...state, isConnectedToBackend: false, isDemoMode: true };
}

/** Keep transport connectivity separate from the source of simulation state. */
export function getBackendConnectionLabel(
  isConnectedToBackend: boolean,
  isDemoMode: boolean,
): BackendConnectionLabel {
  if (!isConnectedToBackend) return 'DISCONNECTED';
  return isDemoMode ? 'DEMO CONTROLLER' : 'LIVE BACKEND';
}
