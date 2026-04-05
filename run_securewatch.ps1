$projectDir = "D:\DrTew\SecureWatch by QingYing JinXuan\SecureWatch"
$restartSec = 3600

Set-Location $projectDir

$lostProc = $null
$attireProc = $null

function Start-BackendWindow {
    param(
        [string]$Title,
        [string]$Command
    )

    Start-Process powershell `
        -ArgumentList @(
            "-NoExit",
            "-ExecutionPolicy", "Bypass",
            "-Command", @"
`$Host.UI.RawUI.WindowTitle = '$Title'
cd '$projectDir'
& '.\venv\Scripts\Activate.ps1'
$Command
"@
        ) `
        -PassThru
}

function Stop-BackendProcessTree {
    param(
        [System.Diagnostics.Process]$Proc
    )

    if ($Proc -and -not $Proc.HasExited) {
        try {
            taskkill /PID $Proc.Id /T /F | Out-Null
        } catch {}
    }
}

try {
    while ($true) {
        Write-Host "=========================================="
        Write-Host "Starting SecureWatch backends..."
        Write-Host (Get-Date)
        Write-Host "=========================================="

        # Lost & Found backend window
        $lostProc = Start-BackendWindow `
            -Title "Lost & Found Backend :8000" `
            -Command '$env:UVICORN_RELOAD=""; python -m uvicorn backend.backend:app --host 0.0.0.0 --port 8000 --log-level warning --no-access-log'

        # Attire backend window
        $attireProc = Start-BackendWindow `
            -Title "Attire Backend :8001" `
            -Command '$env:UVICORN_RELOAD=""; python -m uvicorn server:app --host 0.0.0.0 --port 8001 --log-level warning --no-access-log'

        Write-Host "Lost & Found backend window PID: $($lostProc.Id)"
        Write-Host "Attire backend window PID: $($attireProc.Id)"
        Write-Host "Waiting $restartSec seconds before restart..."
        Write-Host "Press CTRL+C once to stop everything."

        $startTime = Get-Date

        while ($true) {
            Start-Sleep -Seconds 1

            $lostExited = $true
            $attireExited = $true

            try {
                if ($lostProc) { $lostExited = $lostProc.HasExited }
            } catch {
                $lostExited = $true
            }

            try {
                if ($attireProc) { $attireExited = $attireProc.HasExited }
            } catch {
                $attireExited = $true
            }

            if ($lostExited -or $attireExited) {
                Write-Host "One backend exited by itself."
                Stop-BackendProcessTree -Proc $lostProc
                Stop-BackendProcessTree -Proc $attireProc
                break
            }

            $elapsed = (Get-Date) - $startTime
            if ($elapsed.TotalSeconds -ge $restartSec) {
                Write-Host "Restart interval reached. Stopping both backends..."
                Stop-BackendProcessTree -Proc $lostProc
                Stop-BackendProcessTree -Proc $attireProc
                break
            }
        }

        Write-Host "Restarting in 3 seconds..."
        Start-Sleep -Seconds 3
    }
}
catch {
    Write-Host "`nCTRL+C detected. Stopping launcher..."
}
finally {
    Stop-BackendProcessTree -Proc $lostProc
    Stop-BackendProcessTree -Proc $attireProc
    Write-Host "Launcher stopped."
}