using MauiScan.Server.Hubs;
using MauiScan.Server.Services;

var builder = WebApplication.CreateBuilder(args);

// 添加服务
builder.Services.AddControllers();
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

// 注册 SignalR
builder.Services.AddSignalR();

// 注册文件存储服务
builder.Services.AddSingleton<IFileStorageService, FileStorageService>();

// 配置 CORS
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", policy =>
    {
        policy.SetIsOriginAllowed(_ => true)
              .AllowAnyMethod()
              .AllowAnyHeader()
              .AllowCredentials();
    });
});

// 配置文件上传大小限制
builder.Services.Configure<Microsoft.AspNetCore.Http.Features.FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = 20 * 1024 * 1024; // 20MB
});

var app = builder.Build();

// 配置 HTTP 请求管道
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

// 启用 CORS
app.UseCors("AllowAll");

// HTTPS 重定向
app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

// 映射 SignalR Hub
app.MapHub<ScanHub>("/hubs/scan");

app.Run();
