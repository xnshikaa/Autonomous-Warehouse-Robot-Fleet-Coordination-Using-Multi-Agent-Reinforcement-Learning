"""CodeCarbon sustainability and energy consumption analysis for Warehouse MARL."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from codecarbon import OfflineEmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False


# Industrial AMR physical specifications (literature-based standard 100kg payload AMR)
MOTOR_POWER_WATTS = 200.0  # 200W driving motor power consumption
IDLE_POWER_WATTS = 40.0    # 40W baseline computer and sensor standby power
ROBOT_SPEED_MPS = 1.0      # 1.0 meter/second nominal velocity
CELL_SIZE_METERS = 1.0     # 1 grid step = 1 meter


class SustainabilityTracker:
    """Tracks training compute emissions and calculates simulated robot physical energy."""

    def __init__(
        self,
        algorithm: str = "QMIX",
        fleet_size: int = 5,
        output_dir: str | Path = "results/monitoring",
        country_iso_code: str = "USA",
    ):
        self.algorithm = algorithm
        self.fleet_size = fleet_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.country_iso_code = country_iso_code

        self.start_time: Optional[float] = None
        self.tracker = None
        self.emissions_kg: float = 0.0
        self.energy_kwh: float = 0.0

    def start_tracking(self) -> None:
        """Start tracking energy footprint."""
        self.start_time = time.time()
        if HAS_CODECARBON:
            try:
                self.tracker = OfflineEmissionsTracker(
                    country_iso_code=self.country_iso_code,
                    output_dir=str(self.output_dir),
                    log_level="warning",
                )
                self.tracker.start()
            except Exception as e:
                logger.warning(f"Could not initialize OfflineEmissionsTracker: {e}")
                self.tracker = None

    def stop_tracking(
        self,
        total_steps: int = 1000,
        completed_deliveries: int = 50,
        idle_steps: int = 100,
    ) -> Dict[str, Any]:
        """Stop tracking and compute comprehensive compute & AMR physical energy metrics."""
        duration_sec = (time.time() - self.start_time) if self.start_time else 1.0

        if self.tracker is not None:
            try:
                emissions = self.tracker.stop()
                self.emissions_kg = float(emissions) if emissions is not None else 0.0001
            except Exception:
                self.emissions_kg = duration_sec * (150.0 / 3600.0 / 1000.0) * 0.4  # fallback 150W TDP
        else:
            # Theoretical fallback based on 150W compute TDP
            compute_kwh = duration_sec * (150.0 / 3600.0 / 1000.0)
            self.energy_kwh = compute_kwh
            self.emissions_kg = compute_kwh * 0.42  # 0.42 kg CO2 / kWh average grid

        # 2. Simulated Robot Physical Energy Calculations
        # 1 step = 1 meter / 1.0 mps = 1 second
        active_steps = max(0, total_steps - idle_steps)
        moving_energy_joules = active_steps * (MOTOR_POWER_WATTS + IDLE_POWER_WATTS) * (CELL_SIZE_METERS / ROBOT_SPEED_MPS)
        idle_energy_joules = idle_steps * IDLE_POWER_WATTS * (CELL_SIZE_METERS / ROBOT_SPEED_MPS)
        total_robot_joules = moving_energy_joules + idle_energy_joules
        total_robot_kwh = total_robot_joules / 3.6e6

        energy_per_delivery_kwh = (
            float(total_robot_kwh / max(1, completed_deliveries))
            if completed_deliveries > 0 else 0.0
        )
        energy_per_delivery_joules = (
            float(total_robot_joules / max(1, completed_deliveries))
            if completed_deliveries > 0 else 0.0
        )

        report = {
            "algorithm": self.algorithm,
            "fleet_size": self.fleet_size,
            "compute_training": {
                "duration_seconds": round(duration_sec, 2),
                "compute_energy_kwh": round(self.energy_kwh, 6),
                "carbon_emissions_kg_co2": round(self.emissions_kg, 6),
                "measurement_method": "CodeCarbon OfflineEmissionsTracker" if self.tracker else "Theoretical 150W TDP Model",
            },
            "simulated_robot_physical_energy": {
                "total_movement_steps": total_steps,
                "idle_steps": idle_steps,
                "completed_deliveries": completed_deliveries,
                "total_travel_distance_meters": total_steps * CELL_SIZE_METERS,
                "motor_power_rating_watts": MOTOR_POWER_WATTS,
                "idle_power_rating_watts": IDLE_POWER_WATTS,
                "total_physical_energy_joules": round(total_robot_joules, 2),
                "total_physical_energy_kwh": round(total_robot_kwh, 6),
                "energy_per_delivery_joules": round(energy_per_delivery_joules, 2),
                "energy_per_delivery_kwh": round(energy_per_delivery_kwh, 6),
            },
        }

        report_file = self.output_dir / f"sustainability_{self.algorithm.lower()}_n{self.fleet_size}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        return report
