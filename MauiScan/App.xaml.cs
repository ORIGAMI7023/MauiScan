using Microsoft.Extensions.DependencyInjection;

namespace MauiScan
{
    public partial class App : Application
    {
        public App()
        {
            InitializeComponent();
        }

        // 公开方法供其他页面调用
        public static async Task EnsureModelFileCopiedAsync()
        {
            await CopyModelFileStaticAsync();
        }

        private static async Task CopyModelFileStaticAsync()
        {
            try
            {
                var modelPath = Path.Combine(FileSystem.AppDataDirectory, "ppt_corner_detector.onnx");
                System.Diagnostics.Debug.WriteLine($"[ML] AppDataDirectory: {FileSystem.AppDataDirectory}");
                System.Diagnostics.Debug.WriteLine($"[ML] Model path: {modelPath}");

                if (!File.Exists(modelPath))
                {
                    System.Diagnostics.Debug.WriteLine($"[ML] Model file not found, copying...");

                    using var stream = await FileSystem.OpenAppPackageFileAsync("ppt_corner_detector.onnx");
                    using var fileStream = File.Create(modelPath);
                    await stream.CopyToAsync(fileStream);
                    await fileStream.FlushAsync();

                    var fileInfo = new FileInfo(modelPath);
                    System.Diagnostics.Debug.WriteLine($"[ML] Model file copied successfully!");
                    System.Diagnostics.Debug.WriteLine($"[ML] File size: {fileInfo.Length / (1024.0 * 1024.0):F2} MB");
                }
                else
                {
                    var fileInfo = new FileInfo(modelPath);
                    System.Diagnostics.Debug.WriteLine($"[ML] Model file already exists");
                    System.Diagnostics.Debug.WriteLine($"[ML] File size: {fileInfo.Length / (1024.0 * 1024.0):F2} MB");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"[ML] ERROR copying model file: {ex.GetType().Name}");
                System.Diagnostics.Debug.WriteLine($"[ML] ERROR message: {ex.Message}");
                System.Diagnostics.Debug.WriteLine($"[ML] ERROR stack: {ex.StackTrace}");
                throw;
            }
        }

        protected override Window CreateWindow(IActivationState? activationState)
        {
            return new Window(new AppShell());
        }
    }
}