# Runtime Validation Report

**Generated**: 2026-10-14T16:35:00+05:30  
**Target**: `C:\Users\hetan\OneDrive\Desktop\plm\warehouse-final-integration\repository`

## Summary

| Step | Status | Exit Code | Details |
|------|--------|-----------|---------|
| Startup | PASS | 0 | Vite started from the integration repository at `http://127.0.0.1:3000`; HTTP readiness check returned 200 |
| TypeScript validation | PASS | 0 | `npm run lint` completed successfully |
| Production build | PASS | 0 | `npm run build` completed successfully |
| Python tests | PASS | 0 | 152 tests passed, 0 failed |
| TypeScript acceptance tests | PASS | 0 | 23 assertions passed, 0 failed |
| Browser rendering smoke test | PASS | 0 | Playwright screenshot captured the rendered 3D warehouse UI |
| Browser interaction test | PASS | 0 | Browser helper launched after installing revision 1200 and successfully clicked N=5, N=10, and N=20 |

## Acceptance Coverage

The existing TypeScript acceptance runner verified:

- Four-direction movement semantics
- Boundary and shelf rejection
- Same-cell, swap, and simultaneous conflict resolution
- NO_OP execution
- Observation shape, types, and one-hot occupancy encoding
- Task-bearing and robot-state flags
- Environment reset
- Fleet sizes `N=5`, `N=10`, and `N=20`
- Timestep metric updates
- Logical position ownership
- Active AMR count synchronization

## Environment

```text
docker: UNAVAILABLE — Docker command exists, but the daemon did not respond to `docker info`
node: AVAILABLE — v22.22.3
playwright: AVAILABLE — Chromium installed through `npx playwright install chromium`
infra-tier: FALLBACK(embedded/local simulation) — this frontend demo does not require Docker
browser-tier: PRIMARY(Playwright) — rendering and fleet-size interaction flow passed
```

## Issues and Gaps

| # | Severity | Description |
|---|----------|-------------|
| 1 | Informational | Docker was unavailable, but no Docker-managed dependency is required for the local frontend simulation path. |

## Verdict

```text
startup: PASS — Vite, HTTP 200, exit_code: 0
integration: PASS — exit_code: 0, passed: 152, failed: 0, skipped: 0, Python scope
e2e: PASS — exit_code: 0, rendered UI and clicked N=5, N=10, and N=20
overall: PASS — local startup, build, Python integration, simulation acceptance, rendering, and fleet-size interaction all passed
```
