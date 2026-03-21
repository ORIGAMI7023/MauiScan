using System.Configuration;

namespace MauiScan.Server.Middleware;

public class ApiKeyMiddleware
{
    private readonly RequestDelegate _next;
    private readonly IConfiguration _configuration;
    private readonly ILogger<ApiKeyMiddleware> _logger;

    public ApiKeyMiddleware(
        RequestDelegate next,
        IConfiguration configuration,
        ILogger<ApiKeyMiddleware> logger)
    {
        _next = next;
        _configuration = configuration;
        _logger = logger;
    }

    public async Task InvokeAsync(HttpContext context)
    {
        // 检查是否启用 API Key 验证
        var apiKeyEnabled = _configuration.GetValue<bool>("ApiKey:Enabled", false);
        if (!apiKeyEnabled)
        {
            await _next(context);
            return;
        }

        // 跳过 Swagger 端点
        var path = context.Request.Path.Value ?? string.Empty;
        if (path.Contains("/swagger", StringComparison.OrdinalIgnoreCase) ||
            path.Contains("/api-docs", StringComparison.OrdinalIgnoreCase))
        {
            await _next(context);
            return;
        }

        // 跳过 SignalR 端点（SignalR 使用其他认证方式）
        if (path.Contains("/hubs/", StringComparison.OrdinalIgnoreCase))
        {
            await _next(context);
            return;
        }

        // 获取配置的 API Key
        var defaultApiKey = _configuration["ApiKey:DefaultApiKey"];
        if (string.IsNullOrEmpty(defaultApiKey))
        {
            _logger.LogWarning("ApiKey:DefaultApiKey 未配置，允许所有请求通过");
            await _next(context);
            return;
        }

        // 获取请求头中的 API Key
        var headerName = _configuration["ApiKey:HeaderName"] ?? "X-API-Key";
        var providedApiKey = context.Request.Headers.TryGetValue(headerName, out var headerValue)
            ? headerValue.ToString()
            : null;

        // 验证 API Key
        if (!string.IsNullOrEmpty(providedApiKey) && providedApiKey == defaultApiKey)
        {
            await _next(context);
            return;
        }

        // API Key 验证失败
        _logger.LogWarning($"API Key 验证失败: Path={path}, Header={headerName}");

        context.Response.StatusCode = StatusCodes.Status401Unauthorized;
        context.Response.ContentType = "application/json";
        await context.Response.WriteAsJsonAsync(new
        {
            error = "未授权",
            message = "API Key 无效或未提供"
        });
    }
}
