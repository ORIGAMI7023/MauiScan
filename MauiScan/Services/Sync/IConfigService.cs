using MauiScan.Models;

namespace MauiScan.Services.Sync;

public interface IConfigService
{
    Task<SyncConfig> LoadConfigAsync();
    Task SaveConfigAsync(SyncConfig config);
    Task<SyncConfig> GetOrCreateDefaultConfigAsync();
}
