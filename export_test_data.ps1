# 从Android设备导出测试数据的脚本

param(
    [string]$OutputDir = ".\test_data"
)

Write-Host "=== 从Android设备导出测试数据 ===" -ForegroundColor Cyan

# 创建输出目录
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

# 导出图片（假设保存在 /sdcard/DCIM/MauiScan/）
Write-Host "`n1. 导出测试图片..." -ForegroundColor Yellow
adb shell "ls /sdcard/DCIM/MauiScan/*.jpg" 2>$null | ForEach-Object {
    $file = $_.Trim()
    if ($file) {
        $filename = Split-Path -Leaf $file
        Write-Host "  导出: $filename"
        adb pull $file "$OutputDir\$filename" 2>&1 | Out-Null
    }
}

# 导出ML角点数据（假设保存在应用内部存储）
Write-Host "`n2. 导出ML角点数据..." -ForegroundColor Yellow
adb shell "ls /sdcard/Android/data/ORIGAMI.MauiScan/files/*.json" 2>$null | ForEach-Object {
    $file = $_.Trim()
    if ($file) {
        $filename = Split-Path -Leaf $file
        Write-Host "  导出: $filename"
        adb pull $file "$OutputDir\$filename" 2>&1 | Out-Null
    }
}

Write-Host "`n=== 导出完成 ===" -ForegroundColor Green
Write-Host "数据保存在: $OutputDir"
Write-Host "`n使用方法："
Write-Host "  python cv_debug_visualizer.py --image $OutputDir\test.jpg --corners $OutputDir\test_corners.json --output cv_debug_output"
