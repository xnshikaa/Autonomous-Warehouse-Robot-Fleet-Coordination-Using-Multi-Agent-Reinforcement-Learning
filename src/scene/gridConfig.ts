import * as THREE from 'three';

export const GRID_SIZE = 20; // Default 20x20 logical grid
export const CELL_SIZE = 2.0; // 2 meters per grid cell (40m x 40m total warehouse)
export const WAREHOUSE_WIDTH = GRID_SIZE * CELL_SIZE; // 40.0 world units
export const WAREHOUSE_LENGTH = GRID_SIZE * CELL_SIZE; // 40.0 world units

/**
 * Convert logical grid coordinates [gridX, gridY] to World 3D coordinates Vector3
 */
export function gridToWorld(
  gridX: number, 
  gridY: number, 
  heightY: number = 0, 
  gridWidth: number = GRID_SIZE, 
  gridHeight: number = GRID_SIZE
): THREE.Vector3 {
  const worldX = (gridX - (gridWidth / 2) + 0.5) * CELL_SIZE;
  const worldZ = (gridY - (gridHeight / 2) + 0.5) * CELL_SIZE;
  return new THREE.Vector3(worldX, heightY, worldZ);
}

/**
 * Convert World 3D position to closest logical grid cell coordinates [gridX, gridY]
 */
export function worldToGrid(
  worldPos: THREE.Vector3, 
  gridWidth: number = GRID_SIZE, 
  gridHeight: number = GRID_SIZE
): [number, number] {
  const gridX = Math.round((worldPos.x / CELL_SIZE) + (gridWidth / 2) - 0.5);
  const gridY = Math.round((worldPos.z / CELL_SIZE) + (gridHeight / 2) - 0.5);
  const clampedX = Math.max(0, Math.min(gridWidth - 1, gridX));
  const clampedY = Math.max(0, Math.min(gridHeight - 1, gridY));
  return [clampedX, clampedY];
}

