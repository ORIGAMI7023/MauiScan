using Microsoft.AspNetCore.SignalR;

namespace MauiScan.Server.Hubs;

public class ScanHub : Hub
{
    private readonly ILogger<ScanHub> _logger;

    public ScanHub(ILogger<ScanHub> logger)
    {
        _logger = logger;
    }

    public override async Task OnConnectedAsync()
    {
        // 所有设备加入同一个全局组
        await Groups.AddToGroupAsync(Context.ConnectionId, "AllDevices");

        _logger.LogInformation($"设备已连接: {Context.ConnectionId}");

        await base.OnConnectedAsync();
    }

    public override async Task OnDisconnectedAsync(Exception? exception)
    {
        await Groups.RemoveFromGroupAsync(Context.ConnectionId, "AllDevices");

        _logger.LogInformation($"设备已断开: {Context.ConnectionId}");

        await base.OnDisconnectedAsync(exception);
    }
}
