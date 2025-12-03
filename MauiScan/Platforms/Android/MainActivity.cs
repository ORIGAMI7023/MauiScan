using Android.App;
using Android.Content;
using Android.Content.PM;
using Android.OS;
using Android.Views;
using MauiScan.Platforms.Android.Services;

namespace MauiScan
{
    [Activity(Theme = "@style/Maui.SplashTheme", MainLauncher = true, LaunchMode = LaunchMode.SingleTop, ConfigurationChanges = ConfigChanges.ScreenSize | ConfigChanges.Orientation | ConfigChanges.UiMode | ConfigChanges.ScreenLayout | ConfigChanges.SmallestScreenSize | ConfigChanges.Density)]
    public class MainActivity : MauiAppCompatActivity
    {
        public static event EventHandler? VolumeKeyPressed;

        protected override void OnActivityResult(int requestCode, Result resultCode, Intent? data)
        {
            base.OnActivityResult(requestCode, resultCode, data);

            // 处理 ML Kit 扫描结果
            MLKitDocumentScannerService.HandleScanResult(requestCode, resultCode, data);
        }

        public override bool OnKeyDown(Keycode keyCode, KeyEvent? e)
        {
            if (keyCode == Keycode.VolumeUp || keyCode == Keycode.VolumeDown)
            {
                VolumeKeyPressed?.Invoke(this, EventArgs.Empty);
                return true; // 消费事件，不调节音量
            }
            return base.OnKeyDown(keyCode, e);
        }
    }
}
