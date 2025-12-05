# MauiScan 项目说明

## 项目结构

- `MauiScan/` - .NET MAUI 主应用
- `MauiScan.Server/` - ASP.NET Core 服务器（SignalR + API）
- `Native.OpenCV/` - C++ OpenCV 原生库

## 原生库编译

### Android

**⚠️ 重要：必须使用 Developer Command Prompt for VS 2022**

```cmd
cd D:\Programing\C#\MauiScan\Native.OpenCV\android
build-android.bat
```

详细说明：[Native.OpenCV/android/README.md](../Native.OpenCV/android/README.md)

### iOS

需要在 Mac 上编译：
```bash
cd D:\Programing\C#\MauiScan\Native.OpenCV/ios
./build-ios.sh
```

## 服务器部署

```powershell
cd D:\Programing\C#\MauiScan
.\MauiScan.Server\linux\deploy.ps1
```

详细说明：[MauiScan.Server/linux/README.md](../MauiScan.Server/linux/README.md)

## 关键技术栈

- **.NET 10 MAUI** - 跨平台移动应用
- **OpenCV C++** - 文档扫描和透视变换
- **ASP.NET Core + SignalR** - 实时同步
- **Android NDK r27c** - Android 原生库构建
