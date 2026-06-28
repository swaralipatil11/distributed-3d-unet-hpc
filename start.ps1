# start.ps1
# Unified Windows PowerShell launcher for the 3D UNet Pipeline

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "   STARTING 3D MRI PIPELINE LAUNCHER SERVICE   " -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan

# 1. Install/Verify dependencies
Write-Host "[1/4] Checking python dependencies..." -ForegroundColor Yellow
cd "$PSScriptRoot/backend"
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error installing requirements.txt" -ForegroundColor Red
    Exit $LASTEXITCODE
}

# 2. Re-verify mock data arrays
Write-Host "[2/4] Verifying mock dataset arrays..." -ForegroundColor Yellow
if (!(Test-Path "../data/mock_Task01_BrainTumour/imagesTr/BRATS_001.nii.gz") -or !(Test-Path "../data/mock_dicom_volume.zip")) {
    Write-Host "Mock files missing or deleted. Generating new mocks..." -ForegroundColor Yellow
    python ../tests/create_mock_data.py
    python ../tests/create_mock_dicom_zip.py
}

# 3. Check JIT compiled trace binary
Write-Host "[3/4] Checking TorchScript compiled trace checkpoint..." -ForegroundColor Yellow
if (!(Test-Path "deploy/model_trace.pt")) {
    Write-Host "Model trace not found. JIT-compiling model..." -ForegroundColor Yellow
    python -m src.export
}

# 4. Start FastAPI server
Write-Host "[4/4] Starting FastAPI Worker Engine on http://localhost:8000 ..." -ForegroundColor Green
python -m src.main
