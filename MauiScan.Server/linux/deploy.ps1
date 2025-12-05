# ============================================================
# MauiScan Server Deploy Script
# 完整部署（构建 + 配置 + 代码 + 重启）
# ============================================================

$ErrorActionPreference = "Stop"

# 服务器配置
$SERVER = "root@origami7023.net.cn"
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

    # Step 2: Upload nginx config
    Write-Step "Step 2/6: Uploading nginx config"
    scp MauiScan.Server/linux/mauiscan.origami7023.net.cn.conf ${SERVER}:/etc/nginx/conf.d/mauiscan.origami7023.net.cn.conf
    if ($LASTEXITCODE -ne 0) { throw "nginx config upload failed" }
    Write-Success "nginx config uploaded"

    # Step 3: Upload systemd service config
    Write-Step "Step 3/6: Uploading systemd service config"
    scp MauiScan.Server/linux/mauiscan-server.service ${SERVER}:/etc/systemd/system/mauiscan-server.service
    if ($LASTEXITCODE -ne 0) { throw "mauiscan-server.service upload failed" }
    Write-Success "mauiscan-server.service uploaded"

    # Step 4: Upload Server
    Write-Step "Step 4/6: Uploading Server"
    ssh $SERVER "mkdir -p /var/www/mauiscan-server/data/scans"
    scp -r MauiScan.Server/bin/Release/net10.0/publish/* ${SERVER}:/var/www/mauiscan-server/
    if ($LASTEXITCODE -ne 0) { throw "Server upload failed" }
    Write-Success "Server upload completed"

    # Step 5: Test and reload nginx
    Write-Step "Step 5/6: Testing and reloading nginx"
    $nginxCommand = 'sudo nginx -t && sudo systemctl reload nginx'
    ssh $SERVER $nginxCommand
    if ($LASTEXITCODE -ne 0) { throw "nginx reload failed" }
    Write-Success "nginx reloaded"

    # Step 6: Reload systemd and restart service
    Write-Step "Step 6/6: Restarting service"
    $restartCommand = "sudo systemctl daemon-reload && sudo systemctl enable mauiscan-server && sudo systemctl restart mauiscan-server && sleep 2 && sudo systemctl status mauiscan-server"
    ssh $SERVER $restartCommand
    if ($LASTEXITCODE -ne 0) { throw "Service restart failed" }
    Write-Success "Service restarted"

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
