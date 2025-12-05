namespace MauiScan.Models;

public class ScanImageDto
{
    public string FileName { get; set; } = string.Empty;
    public long FileSize { get; set; }
    public int Width { get; set; }
    public int Height { get; set; }
    public DateTime ScannedAt { get; set; }
    public string DownloadUrl { get; set; } = string.Empty;
}
