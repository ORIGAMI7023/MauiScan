# MauiScan Server 部署说明

## 部署步骤

1. 在项目根目录执行部署脚本：
```powershell
cd D:\Programing\C#\MauiScan
.\MauiScan.Server\linux\deploy.ps1
```

2. 部署完成后，访问以下地址测试：
- API: https://mauiscan.origami7023.net.cn/api/scans/recent
- Swagger: https://mauiscan.origami7023.net.cn/swagger

## 查看服务状态

```bash
# 查看服务状态
ssh root@origami7023.net.cn 'systemctl status mauiscan-server'

# 查看实时日志
ssh root@origami7023.net.cn 'journalctl -u mauiscan-server -f'

# 重启服务
ssh root@origami7023.net.cn 'systemctl restart mauiscan-server'
```

## 文件位置

- 服务器代码: `/var/www/mauiscan-server/`
- 扫描图片: `/var/www/mauiscan-server/data/scans/`
- nginx 配置: `/etc/nginx/conf.d/mauiscan.origami7023.net.cn.conf`
- systemd 服务: `/etc/systemd/system/mauiscan-server.service`

## 配置说明

- 监听端口: 5012 (内网)
- 外部域名: https://mauiscan.origami7023.net.cn
- 文件大小限制: 20MB
- 存储目录: /var/www/mauiscan-server/data/scans
