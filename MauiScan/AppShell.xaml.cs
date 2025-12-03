using MauiScan.Views;

namespace MauiScan
{
    public partial class AppShell : Shell
    {
        public AppShell()
        {
            InitializeComponent();

            // 注册相机页面路由
            Routing.RegisterRoute("camera", typeof(CameraPage));
        }
    }
}
