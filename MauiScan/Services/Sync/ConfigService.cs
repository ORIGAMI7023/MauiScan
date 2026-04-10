using System.Text.Json;
using MauiScan.Models;

namespace MauiScan.Services.Sync;

public class ConfigService : IConfigService
{
    private readonly string _appDataConfigPath;
    private readonly string _projectRootConfigPath;

    public ConfigService()
    {
        var appDataDir = FileSystem.AppDataDirectory;
        _appDataConfigPath = Path.Combine(appDataDir, "sync_config.json");

        // 尝试查找项目根目录的配置文件（开发环境）
        // 在 MAUI 中，AppDomain.CurrentDomain.BaseDirectory 指向应用的可执行文件目录
        // 我们需要向上查找项目根目录
        var baseDir = AppDomain.CurrentDomain.BaseDirectory;
        _projectRootConfigPath = FindProjectRootConfigPath(baseDir);
    }

    private string FindProjectRootConfigPath(string startDir)
    {
        try
        {
            // 尝试向上查找项目根目录（包含 .csproj 或 .sln 文件的目录）
            var currentDir = new DirectoryInfo(startDir);
            while (currentDir != null && currentDir.Parent != null)
            {
                var configPath = Path.Combine(currentDir.FullName, "sync_config.json");
                if (File.Exists(configPath))
                {
                    Console.WriteLine($"✅ 找到项目配置文件: {configPath}");
                    return configPath;
                }

                // 检查是否到达项目根目录（包含 .csproj 或 .sln 文件）
                if (currentDir.GetFiles("*.csproj").Length > 0 ||
                    currentDir.GetFiles("*.sln").Length > 0)
                {
                    break;
                }

                currentDir = currentDir.Parent;
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"⚠️  查找项目配置文件失败: {ex.Message}");
        }

        return string.Empty;
    }

    public async Task<SyncConfig> LoadConfigAsync()
    {
        try
        {
            // 优先读取项目根目录的配置文件（开发环境）
            if (!string.IsNullOrEmpty(_projectRootConfigPath) && File.Exists(_projectRootConfigPath))
            {
                Console.WriteLine($"📂 使用项目配置文件: {_projectRootConfigPath}");
                var json = await File.ReadAllTextAsync(_projectRootConfigPath);
                var config = JsonSerializer.Deserialize<SyncConfig>(json, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                if (config != null)
                {
                    return config;
                }
            }

            // 其次读取 AppData 目录的配置文件（生产环境）
            if (File.Exists(_appDataConfigPath))
            {
                Console.WriteLine($"📂 使用 AppData 配置文件: {_appDataConfigPath}");
                var json = await File.ReadAllTextAsync(_appDataConfigPath);
                var config = JsonSerializer.Deserialize<SyncConfig>(json, new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                });

                if (config != null)
                {
                    return config;
                }
            }

            // 尝试从嵌入资源读取（生产环境默认配置）
            try
            {
                using var stream = FileSystem.OpenAppPackageFileAsync("sync_config.json").GetAwaiter().GetResult();
                if (stream != null)
                {
                    Console.WriteLine($"📦 使用嵌入资源配置: sync_config.json");
                    using var reader = new StreamReader(stream);
                    var json = await reader.ReadToEndAsync();
                    var config = JsonSerializer.Deserialize<SyncConfig>(json, new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });

                    if (config != null)
                    {
                        // 保存到 AppData，方便后续修改
                        await SaveConfigAsync(config);
                        return config;
                    }
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"⚠️  无法读取嵌入资源配置: {ex.Message}");
            }

            // 如果都不存在，创建默认配置
            return await GetOrCreateDefaultConfigAsync();
        }
        catch (Exception ex)
        {
            // 加载失败时返回默认配置
            Console.WriteLine($"❌ 加载配置失败: {ex.Message}");
            return await GetOrCreateDefaultConfigAsync();
        }
    }

    public async Task SaveConfigAsync(SyncConfig config)
    {
        try
        {
            var json = JsonSerializer.Serialize(config, new JsonSerializerOptions
            {
                WriteIndented = true
            });

            var directory = Path.GetDirectoryName(_appDataConfigPath);
            if (!string.IsNullOrEmpty(directory) && !Directory.Exists(directory))
            {
                Directory.CreateDirectory(directory);
            }

            await File.WriteAllTextAsync(_appDataConfigPath, json);
            Console.WriteLine($"💾 配置已保存到: {_appDataConfigPath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"❌ 保存配置失败: {ex.Message}");
            throw;
        }
    }

    public async Task<SyncConfig> GetOrCreateDefaultConfigAsync()
    {
        var defaultConfig = new SyncConfig
        {
            ServerUrl = "https://mauiscan.origami7023.net.cn",
            ApiKey = string.Empty,
            ConnectionTimeoutSeconds = 120
        };

        await SaveConfigAsync(defaultConfig);
        Console.WriteLine("📝 已创建默认配置文件");
        return defaultConfig;
    }
}
