namespace MauiScan.Models;

public class SyncConfig
{
    public string ServerUrl { get; set; } = "https://mauiscan.origami7023.net.cn";
    public string ApiKey { get; set; } = string.Empty;
    public int ConnectionTimeoutSeconds { get; set; } = 120;
}
