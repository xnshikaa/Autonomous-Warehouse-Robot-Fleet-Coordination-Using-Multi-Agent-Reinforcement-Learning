param(
    [ValidateRange(1024, 65535)][int]$BackendPort = 8000,
    [ValidateRange(1024, 65535)][int]$FrontendPort = 5174,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$backendUrl = "http://127.0.0.1:$BackendPort"
$frontendUrl = "http://127.0.0.1:$FrontendPort"
$pythonExe = (Get-Command $Python -CommandType Application | Select-Object -First 1).Source
$nodeExe = (Get-Command node -CommandType Application | Select-Object -First 1).Source
if ($BackendPort -eq $FrontendPort) { throw 'Frontend and backend ports must differ.' }
foreach ($port in $BackendPort, $FrontendPort) {
    $probe = [Net.Sockets.TcpClient]::new()
    try {
        if ($probe.ConnectAsync('127.0.0.1', $port).Wait(500) -and $probe.Connected) {
            throw "Port $port is in use. Stop its owner or choose a different port."
        }
    } catch [AggregateException] {
        # A refused connection means the dev server can bind the port.
        if ($_.Exception.GetBaseException().SocketErrorCode -ne 'ConnectionRefused') { throw }
    } finally {
        $probe.Dispose()
    }
}

$vite = Join-Path $root 'node_modules/vite/bin/vite.js'
if (!(Test-Path -LiteralPath $vite)) { throw 'Install frontend dependencies with npm ci.' }
& $pythonExe -c 'import fastapi, uvicorn, torch, websockets'
if ($LASTEXITCODE) { throw 'Install the backend dependencies from requirements.txt in the selected Python environment.' }

$logDir = Join-Path $root '.pytest_cache/warehouse-startup'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$backend = $null
$frontend = $null
Push-Location -LiteralPath $root
try {
    $oldEnvironment = @{}
    $configuration = @{
        QMIX_CHECKPOINT = $null
        OMP_NUM_THREADS = '1'
        MKL_NUM_THREADS = '1'
        VITE_WAREHOUSE_WS_URL = "ws://127.0.0.1:$BackendPort/ws"
    }
    foreach ($algorithm in 'qmix','iql') {
        foreach ($n in 5,10,20) {
            $selection = Get-Content -Raw "results/project25_validation/${algorithm}_n$n.json" | ConvertFrom-Json
            foreach ($file in $selection.results[0].checkpoint_sha256.PSObject.Properties) {
                if ((Get-FileHash -LiteralPath $file.Name -Algorithm SHA256).Hash -ne $file.Value) {
                    throw "Checkpoint checksum mismatch: $($file.Name)"
                }
            }
            $configuration["$($algorithm.ToUpper())_CHECKPOINT_N$n"] = (Resolve-Path -LiteralPath $selection.selected_checkpoint).Path
        }
    }
    try {
        foreach ($name in $configuration.Keys) {
            $oldEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
            [Environment]::SetEnvironmentVariable($name, $configuration[$name], 'Process')
        }
        $backend = Start-Process $pythonExe -ArgumentList '-m','uvicorn','src.api.warehouse_backend:create_app','--factory','--host','127.0.0.1','--port',"$BackendPort" -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'backend.out') -RedirectStandardError (Join-Path $logDir 'backend.err')
        $frontend = Start-Process $nodeExe -ArgumentList 'node_modules/vite/bin/vite.js','--host','127.0.0.1','--port',"$FrontendPort",'--strictPort' -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'frontend.out') -RedirectStandardError (Join-Path $logDir 'frontend.err')
    } finally {
        foreach ($name in $oldEnvironment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $oldEnvironment[$name], 'Process')
        }
    }

    $deadline = (Get-Date).AddSeconds(60)
    do {
        if ($backend.HasExited -or $frontend.HasExited) { throw 'A warehouse server exited during startup.' }
        try {
            $ready = Invoke-RestMethod "$backendUrl/readiness" -TimeoutSec 2
            $page = Invoke-WebRequest "$frontendUrl/" -TimeoutSec 2
            if ($ready.policy.source -eq 'QMIX' -and $ready.policy.trained -and
                $ready.policy.activeFleetSize -eq 5 -and
                $page.Content -match 'Autonomous Warehouse.*(QMIX|Reinforcement Learning)') {
                Write-Output "Warehouse: $frontendUrl/"
                Write-Output "Backend: $backendUrl/state"
                Write-Output "Shutdown: Stop-Process -Id $($backend.Id),$($frontend.Id)"
                return
            }
        } catch {
            if ((Get-Date) -gt $deadline) { throw }
        }
        Start-Sleep -Milliseconds 600
    } while ((Get-Date) -lt $deadline)
    throw 'Warehouse servers did not become ready.'
} catch {
    foreach ($process in $backend,$frontend) {
        if ($process -and !$process.HasExited) { Stop-Process -Id $process.Id }
    }
    Get-Content (Join-Path $logDir 'backend.err'),(Join-Path $logDir 'frontend.err') -Tail 12 -ErrorAction SilentlyContinue
    throw
} finally {
    Pop-Location
}
