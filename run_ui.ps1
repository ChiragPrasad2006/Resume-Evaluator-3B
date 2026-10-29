# ==============================================================================
# Production Web UI Launcher for Local AI Resume Reviewer
# Runs on http://127.0.0.1:7860 with 8-bit INT8 inference (< 4.5 GB VRAM)
# Loads LoRA Adapter from Hugging Face Hub: Kami0867/Qwen-2.5-3B-Resume-Evaluator
# ==============================================================================

$ErrorActionPreference = "Stop"

# 1. Resolve Project Root dynamically
$ProjectRoot = Split-Path -Parent -Path $MyInvocation.MyCommand.Definition
if (-not $ProjectRoot) { $ProjectRoot = Get-Location }
Set-Location $ProjectRoot

# 2. Force unbuffered output
$env:PYTHONUNBUFFERED = "1"

# 3. Detect and Activate Virtual Environment
$VenvCandidateList = @(
    "$ProjectRoot\.venv",
    "$ProjectRoot\..\.venv",
    "D:\MLops\.venv"
)

$VenvDir = $null
foreach ($cand in $VenvCandidateList) {
    if (Test-Path "$cand\Scripts\python.exe") {
        $VenvDir = (Resolve-Path $cand).Path
        break
    }
}

if ($VenvDir) {
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "Activating Python Virtual Environment: $VenvDir" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
    & "$VenvDir\Scripts\Activate.ps1"
    $PythonExe = "$VenvDir\Scripts\python.exe"
} else {
    Write-Host "[Notice] No dedicated .venv folder found; utilizing system Python." -ForegroundColor Yellow
    $PythonExe = "python"
}

# 4. Display Pre-Flight Diagnostic Banner
Write-Host "`n[Pre-Flight Server Check]" -ForegroundColor Green
Write-Host "  Project Root        : $ProjectRoot"
Write-Host "  Base Model          : Qwen/Qwen2.5-3B-Instruct (8-bit INT8)"
Write-Host "  Trained LoRA Adapter: Kami0867/Qwen-2.5-3B-Resume-Evaluator (Hugging Face Hub)"
Write-Host "  Database Store      : $ProjectRoot\data\sessions.db"
Write-Host "  Inference VRAM Cap  : < 4.50 GB Allocation Budget"
Write-Host "  Local Web URL       : http://127.0.0.1:7860"
Write-Host "  Browser Auto-Launch : Enabled (opens automatically when server is ready)`n"

# 5. Start FastAPI Server via Python module entrypoint
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Starting FastAPI Server on http://127.0.0.1:7860 (Press Ctrl+C to stop)..." -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

try {
    & $PythonExe -m web.app
} catch {
    # Suppress PowerShell termination on user interrupt
}
