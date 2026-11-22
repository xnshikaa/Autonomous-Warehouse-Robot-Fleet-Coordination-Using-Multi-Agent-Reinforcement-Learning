# Warehouse application integration

## Live startup

Run `./Start-Warehouse.ps1` from PowerShell at the repository root. The
script verifies all six selected checkpoint file sets before binding FastAPI
to port 8000 and Vite to port 5174. It prints the application URL and the two
process IDs to stop. Pass `-Python` with a virtual-environment executable
when needed, and `-BackendPort` or `-FrontendPort` for an occupied port.
Startup logs are under `.pytest_cache/warehouse-startup`.

Open [the local warehouse](http://127.0.0.1:5174/). Choose a fleet and
algorithm, then press **Start**. **Pause** freezes the current episode.
**Resume** continues it, and **Reset** starts a fresh paused episode.
The service exposes `/health`, `/readiness`, `/state`, and `/ws`.

## Scene and telemetry

The original perspective camera from the parent of commit `90338b7` is
back. The floor, racks, dispatch area, AMR meshes and the task, path, and
safety rendering systems remain in use. Shelf coordinates and worker
exclusion cells come from backend configuration. Pickup and delivery
markers use episode task coordinates. Robot turns follow executed motion
and the route lines retain the most recent 32 visited cells. The optional
overhead camera helps inspect congestion.

The dashboard uses the backend-confirmed controller name, fleet size,
checkpoint and simulation status. QMIX and IQL each load their own
fleet-specific checkpoint. Rule-Based uses the benchmark baseline
controller. A failed request returns an explicit error and leaves the
confirmed state intact. The browser never creates local robot movement.
The last frame is marked stale during a disconnect and controls stay
disabled until fresh backend state arrives.

Each local episode generates one assigned task per robot. Delivery time is
measured in simulation steps from episode start. Throughput is deliveries
per 100 steps. Collision events use committed robot positions, while
safety conflicts count robot-conflict interventions and blocked groups
count connected sets of contested intentions.

## Live verification

Headless Chrome rendered the actual application at desktop size using
software WebGL. The check exercised QMIX, IQL, Rule-Based, all fleet sizes,
start, pause, resume, reset, robot inspection, camera presets, refresh,
and a dropped and restored WebSocket. It reported zero console errors
and zero uncaught exceptions.

Run the separate live API check against the started server:

```powershell
python -m python_infra.verify_live_warehouse
```

The live API check passed all nine controller/fleet combinations and
confirmed that an invalid algorithm request left the prior policy active.

[Warehouse overview](results/project25_live/screenshots/01-warehouse-overview.png) ·
[QMIX, five robots](results/project25_live/screenshots/02-qmix-five-running.png) ·
[Robot details and telemetry](results/project25_live/screenshots/03-robot-details-and-telemetry.png) ·
[IQL, ten robots](results/project25_live/screenshots/04-iql-ten-running.png) ·
[Rule-Based, twenty robots](results/project25_live/screenshots/05-rule-based-twenty-running.png) ·
[QMIX, twenty robots](results/project25_live/screenshots/06-qmix-twenty-running.png) ·
[Robot camera](results/project25_live/screenshots/07-robot-camera.png) ·
[Overhead inspection](results/project25_live/screenshots/08-overhead-inspection.png)

## Docker verification procedure

Docker Desktop Linux engine verification is pending. In this session `docker
version` found the client in context `desktop-linux` but returned
`open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`.
`docker desktop start --timeout 45` returned
`Docker Desktop is still starting: context deadline exceeded`.
On a machine with a running Docker Linux engine, run these commands from the
repository root to load the verified N=5 checkpoint and test actual inference.

```powershell
docker version
docker build --tag project25-inference .
docker run --detach --name project25-inference-check --publish 127.0.0.1:8001:8001 `
    --env QMIX_CHECKPOINT=/app/artifacts/training/qmix_n5_v1/snapshots/episode_00600/checkpoint_state.pt `
    project25-inference
try {
    $deadline = (Get-Date).AddMinutes(3)
    do {
        $status = docker inspect --format '{{.State.Health.Status}}' project25-inference-check
        if ($status -eq 'healthy') { break }
        if ($status -eq 'unhealthy' -or (Get-Date) -gt $deadline) {
            docker logs project25-inference-check
            throw "Container health status is $status"
        }
        Start-Sleep -Seconds 3
    } while ($true)
    $base = 'http://127.0.0.1:8001'
    $health = Invoke-RestMethod "$base/health"
    $ready = Invoke-RestMethod "$base/readiness"
    $model = Invoke-RestMethod "$base/model/info"
    if (!$health.ok -or !$ready.ok -or !$model.trained -or $model.fleet_size -ne 5 -or $model.obs_dim -ne 456) {
        throw 'Expected trained N=5 checkpoint with 456 observations'
    }
    $obs = [double[]]::new(456)
    $request = @{fleet_size=5; observations=@($obs,$obs,$obs,$obs,$obs); epsilon=0} | ConvertTo-Json -Compress -Depth 4
    $decision = Invoke-RestMethod "$base/decide" -Method Post -ContentType 'application/json' -Body $request
    if (!$decision.trained -or $decision.actions.Count -ne 5 -or @($decision.actions | Where-Object { $_ -lt 0 -or $_ -gt 3 }).Count) {
        throw 'Invalid inference response'
    }
    $model
    $decision
} finally {
    docker stop project25-inference-check
    docker rm project25-inference-check
}
```
