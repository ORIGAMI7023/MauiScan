using Android.App;
using Android.Content.PM;
using Android.Views;

namespace MauiScan
{
    [Activity(Theme = "@style/Maui.SplashTheme", MainLauncher = true, LaunchMode = LaunchMode.SingleTop, ConfigurationChanges = ConfigChanges.ScreenSize | ConfigChanges.Orientation | ConfigChanges.UiMode | ConfigChanges.ScreenLayout | ConfigChanges.SmallestScreenSize | ConfigChanges.Density)]
    public class MainActivity : MauiAppCompatActivity
    {
        public static event EventHandler? VolumeKeyPressed;

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
