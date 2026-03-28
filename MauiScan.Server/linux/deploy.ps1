# ============================================================
# MauiScan Server Deploy Script
# 完整部署（构建 + 配置 + 代码 + 重启）
# ============================================================

$ErrorActionPreference = "Stop"

# 服务器配置
$SERVER = "origami@downf.cn"
$PROJECT_ROOT = "D:\Programing\C#\MauiScan"

function Write-Step {
    param([string]$Message)
    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "=> $Message" -ForegroundColor Yellow
    Write-Host "========================================" -ForegroundColor Yellow
}

function Write-Success {
    param([string]$Message)
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-ErrorMsg {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor Red
}

try {
    # Step 1: Build Server
    Write-Step "Step 1/6: Building MauiScan Server"
    Set-Location $PROJECT_ROOT
    dotnet publish MauiScan.Server/MauiScan.Server.csproj -c Release -o MauiScan.Server/bin/Release/net10.0/publish --runtime linux-x64 --self-contained false
    if ($LASTEXITCODE -ne 0) { throw "Server build failed" }
    Write-Success "Server build completed"

    # Step 2: Upload nginx config to temp and install
    Write-Step "Step 2/6: Installing nginx config"
    scp MauiScan.Server/linux/mauiscan.origami7023.net.cn.conf ${SERVER}:/tmp/mauiscan.origami7023.net.cn.conf
    if ($LASTEXITCODE -ne 0) { throw "nginx config upload failed" }
    ssh $SERVER "sudo mv /tmp/mauiscan.origami7023.net.cn.conf /etc/nginx/conf.d/mauiscan.origami7023.net.cn.conf"
    if ($LASTEXITCODE -ne 0) { throw "nginx config install failed" }
    Write-Success "nginx config installed"

    # Step 3: Upload systemd service config to temp and install
    Write-Step "Step 3/6: Installing systemd service config"
    scp MauiScan.Server/linux/mauiscan-server.service ${SERVER}:/tmp/mauiscan-server.service
    if ($LASTEXITCODE -ne 0) { throw "mauiscan-server.service upload failed" }
    ssh $SERVER "sudo mv /tmp/mauiscan-server.service /etc/systemd/system/mauiscan-server.service"
    if ($LASTEXITCODE -ne 0) { throw "mauiscan-server.service install failed" }
    Write-Success "mauiscan-server.service installed"

    # Step 4: Upload Server
    Write-Step "Step 4/7: Uploading Server"
    ssh $SERVER "mkdir -p /var/www/mauiscan-server/data/scans"
    scp -r MauiScan.Server/bin/Release/net10.0/publish/* ${SERVER}:/var/www/mauiscan-server/
    if ($LASTEXITCODE -ne 0) { throw "Server upload failed" }
    Write-Success "Server upload completed"

    # Step 5: Upload appsettings.json
    Write-Step "Step 5/7: Uploading appsettings.json"
    if (Test-Path "MauiScan.Server/appsettings.json") {
        scp MauiScan.Server/appsettings.json ${SERVER}:/var/www/mauiscan-server/appsettings.json
        if ($LASTEXITCODE -ne 0) { throw "appsettings.json upload failed" }
        Write-Success "appsettings.json uploaded (contains API Key)"
    } else {
        Write-Host "[WARN] MauiScan.Server/appsettings.json not found!" -ForegroundColor Yellow
        Write-Host "[WARN] Please create it from appsettings.example.json" -ForegroundColor Yellow
        Write-Host "[WARN] Server will use default configuration (NO API KEY!)" -ForegroundColor Yellow
    }

    # Step 6: Set directory ownership
    Write-Step "Step 6/7: Setting directory ownership"
    ssh $SERVER "sudo chown -R origami:origami /var/www/mauiscan-server"
    if ($LASTEXITCODE -ne 0) { throw "Directory ownership setup failed" }
    Write-Success "Directory ownership set to origami:origami"

    # Step 7: Test and reload nginx
    Write-Step "Step 7/7: Testing and reloading nginx and restarting service"
    $reloadCommand = 'sudo nginx -t && sudo systemctl reload nginx && sudo systemctl daemon-reload && sudo systemctl enable mauiscan-server && sudo systemctl restart mauiscan-server && sleep 2 && sudo systemctl status mauiscan-server'
    ssh $SERVER $reloadCommand
    if ($LASTEXITCODE -ne 0) { throw "Service reload failed" }
    Write-Success "Nginx and service reloaded"

    # Done
    Write-Step "Full Deployment Completed!"
    Write-Success "Server API: https://mauiscan.origami7023.net.cn"
    Write-Success "SignalR Hub: wss://mauiscan.origami7023.net.cn/hubs/scan"
    Write-Host "`nView logs:" -ForegroundColor Cyan
    Write-Host "  ssh $SERVER 'journalctl -u mauiscan-server -f'" -ForegroundColor Gray
    Write-Host "`nTest API:" -ForegroundColor Cyan
    Write-Host "  curl https://mauiscan.origami7023.net.cn/api/scans/recent" -ForegroundColor Gray

} catch {
    Write-ErrorMsg "Deployment failed: $_"
    Write-Host "`nDeployment aborted" -ForegroundColor Red
    exit 1
}
