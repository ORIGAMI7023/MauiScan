using System.Text.Json;

namespace AnnotationTool;

public partial class Form1 : Form
{
    private List<string> imageFiles = new();
    private int currentIndex = -1;
    private Image? currentImage;
    private List<Point> corners = new();
    private int? draggedPointIndex = null;
    private float zoomFactor = 1.0f;
    private Point zoomOffset = Point.Empty;

    private PictureBox pictureBox = new();
    private Label statusLabel = new();
    private Button prevButton = new();
    private Button nextButton = new();
    private Button loadFolderButton = new();
    private Button saveButton = new();
    private Label instructionLabel = new();

    public Form1()
    {
        InitializeComponent();
        InitializeUI();
    }

    private void InitializeUI()
    {
        this.Text = "PPT 四角点标注工具";
        this.Width = 1400;
        this.Height = 900;
        this.KeyPreview = true;
        this.MinimumSize = new Size(800, 600);
        this.DoubleBuffered = true;

        // 禁用 Tab 键焦点切换
        this.KeyDown += (s, e) =>
        {
            if (e.KeyCode == Keys.Tab)
            {
                e.Handled = true;
                e.SuppressKeyPress = true;
            }
        };

        // PictureBox - 显示图片 (使用 Anchor 自适应)
        pictureBox.Location = new Point(10, 50);
        pictureBox.Size = new Size(this.ClientSize.Width - 280, this.ClientSize.Height - 60);
        pictureBox.BorderStyle = BorderStyle.FixedSingle;
        pictureBox.SizeMode = PictureBoxSizeMode.Normal;
        pictureBox.BackColor = Color.DarkGray;
        pictureBox.Image = null;  // 确保不使用 Image 属性
        pictureBox.Anchor = AnchorStyles.Top | AnchorStyles.Bottom | AnchorStyles.Left | AnchorStyles.Right;
        pictureBox.TabStop = false;  // 禁用 Tab 焦点
        pictureBox.MouseDown += PictureBox_MouseDown;
        pictureBox.MouseMove += PictureBox_MouseMove;
        pictureBox.MouseUp += PictureBox_MouseUp;
        pictureBox.Paint += PictureBox_Paint;
        pictureBox.MouseWheel += PictureBox_MouseWheel;
        this.Controls.Add(pictureBox);

        // 右侧控制面板的 X 坐标
        int rightPanelX = this.ClientSize.Width - 260;

        // 加载文件夹按钮
        loadFolderButton.Location = new Point(rightPanelX, 50);
        loadFolderButton.Size = new Size(250, 40);
        loadFolderButton.Text = "加载图片文件夹";
        loadFolderButton.Anchor = AnchorStyles.Top | AnchorStyles.Right;
        loadFolderButton.TabStop = false;
        loadFolderButton.Click += LoadFolderButton_Click;
        this.Controls.Add(loadFolderButton);

        // 说明文字
        instructionLabel.Location = new Point(rightPanelX, 100);
        instructionLabel.Size = new Size(250, 180);
        instructionLabel.Anchor = AnchorStyles.Top | AnchorStyles.Right;
        instructionLabel.Text = "操作说明：\n\n" +
                               "1. 点击选择四个角点\n" +
                               "   (左上→右上→右下→左下)\n\n" +
                               "2. 拖拽调整角点位置\n\n" +
                               "3. 滚轮 - 缩放 / R - 重置\n\n" +
                               "4. Enter - 保存并下一张\n" +
                               "5. Space - 跳过当前\n" +
                               "6. Esc - 清除角点";
        this.Controls.Add(instructionLabel);

        // 上一张按钮
        prevButton.Location = new Point(rightPanelX, 290);
        prevButton.Size = new Size(120, 40);
        prevButton.Text = "上一张 (←)";
        prevButton.Anchor = AnchorStyles.Top | AnchorStyles.Right;
        prevButton.TabStop = false;
        prevButton.Click += (s, e) => NavigateImage(-1);
        this.Controls.Add(prevButton);

        // 下一张按钮
        nextButton.Location = new Point(rightPanelX + 130, 290);
        nextButton.Size = new Size(120, 40);
        nextButton.Text = "下一张 (→)";
        nextButton.Anchor = AnchorStyles.Top | AnchorStyles.Right;
        nextButton.TabStop = false;
        nextButton.Click += (s, e) => NavigateImage(1);
        this.Controls.Add(nextButton);

        // 保存按钮
        saveButton.Location = new Point(rightPanelX, 340);
        saveButton.Size = new Size(250, 50);
        saveButton.Text = "保存标注 (Enter)";
        saveButton.Enabled = false;
        saveButton.Anchor = AnchorStyles.Top | AnchorStyles.Right;
        saveButton.TabStop = false;
        saveButton.Click += SaveButton_Click;
        this.Controls.Add(saveButton);

        // 状态栏
        statusLabel.Location = new Point(10, 10);
        statusLabel.Size = new Size(this.ClientSize.Width - 20, 30);
        statusLabel.Text = "请先加载图片文件夹";
        statusLabel.Font = new Font(statusLabel.Font.FontFamily, 12, FontStyle.Bold);
        statusLabel.Anchor = AnchorStyles.Top | AnchorStyles.Left | AnchorStyles.Right;
        this.Controls.Add(statusLabel);

        // 键盘快捷键
        this.KeyDown += Form1_KeyDown;
    }

    private void LoadFolderButton_Click(object? sender, EventArgs e)
    {
        using var dialog = new FolderBrowserDialog();
        dialog.Description = "选择包含图片的文件夹";

        if (dialog.ShowDialog() == DialogResult.OK)
        {
            LoadImagesFromFolder(dialog.SelectedPath);
        }
    }

    private void LoadImagesFromFolder(string folderPath)
    {
        var extensions = new[] { ".jpg", ".jpeg", ".png", ".bmp" };
        imageFiles = Directory.GetFiles(folderPath)
            .Where(f => extensions.Contains(Path.GetExtension(f).ToLower()))
            .OrderBy(f => f)
            .ToList();

        if (imageFiles.Count == 0)
        {
            MessageBox.Show("文件夹中没有找到图片文件！", "错误", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        currentIndex = 0;
        LoadCurrentImage();
    }

    private void LoadCurrentImage()
    {
        if (currentIndex < 0 || currentIndex >= imageFiles.Count)
            return;

        currentImage?.Dispose();
        corners.Clear();
        zoomFactor = 1.0f;
        zoomOffset = Point.Empty;

        string imagePath = imageFiles[currentIndex];

        try
        {
            // 多种方法尝试加载图片
            currentImage = LoadImageWithFallback(imagePath);

            if (currentImage == null)
            {
                throw new Exception("所有加载方法均失败");
            }

            // 尝试加载已有标注
            LoadExistingAnnotation(imagePath);

            UpdateStatus();
            pictureBox.Invalidate();
        }
        catch (Exception ex)
        {
            var errorMsg = $"无法加载图片: {Path.GetFileName(imagePath)}\n\n" +
                          $"完整路径: {imagePath}\n\n" +
                          $"错误类型: {ex.GetType().Name}\n" +
                          $"错误信息: {ex.Message}\n\n" +
                          $"可能原因:\n" +
                          $"- 文件已损坏\n" +
                          $"- 格式不支持（如 HEIC/WebP）\n" +
                          $"- iOS 转换的 PNG 包含特殊元数据\n" +
                          $"- 文件被其他程序占用\n\n" +
                          $"将跳过此图片。";

            MessageBox.Show(errorMsg, "加载错误", MessageBoxButtons.OK, MessageBoxIcon.Warning);

            // 跳过损坏的图片
            if (currentIndex < imageFiles.Count - 1)
            {
                NavigateImage(1);
            }
            else if (currentIndex > 0)
            {
                NavigateImage(-1);
            }
        }
    }

    private Image? LoadImageWithFallback(string imagePath)
    {
        // 方法 1: 直接转换为标准 32bpp ARGB 格式（最可靠）
        try
        {
            using (var stream = new FileStream(imagePath, FileMode.Open, FileAccess.Read))
            using (var tempImage = Image.FromStream(stream, false, false))
            {
                // 立即转换为标准格式，解决色彩空间和特殊编码问题
                var standardBitmap = new Bitmap(tempImage.Width, tempImage.Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                using (var g = Graphics.FromImage(standardBitmap))
                {
                    g.DrawImage(tempImage, 0, 0, tempImage.Width, tempImage.Height);
                }
                // 验证图片可以正常绘制
                ValidateImage(standardBitmap);
                return standardBitmap;
            }
        }
        catch { }

        // 方法 2: 使用 Bitmap 类加载后转换
        try
        {
            using (var stream = new FileStream(imagePath, FileMode.Open, FileAccess.Read))
            using (var bitmap = new Bitmap(stream))
            {
                var standardBitmap = new Bitmap(bitmap.Width, bitmap.Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                using (var g = Graphics.FromImage(standardBitmap))
                {
                    g.DrawImage(bitmap, 0, 0, bitmap.Width, bitmap.Height);
                }
                ValidateImage(standardBitmap);
                return standardBitmap;
            }
        }
        catch { }

        // 方法 3: 忽略色彩管理后转换
        try
        {
            using (var stream = new FileStream(imagePath, FileMode.Open, FileAccess.Read))
            using (var tempImage = Image.FromStream(stream, true, false))
            {
                var standardBitmap = new Bitmap(tempImage.Width, tempImage.Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                using (var g = Graphics.FromImage(standardBitmap))
                {
                    g.DrawImage(tempImage, 0, 0, tempImage.Width, tempImage.Height);
                }
                ValidateImage(standardBitmap);
                return standardBitmap;
            }
        }
        catch { }

        return null;
    }

    private void ValidateImage(Image image)
    {
        // 尝试绘制到临时画布，验证图片数据完整性
        using (var testBitmap = new Bitmap(1, 1))
        using (var g = Graphics.FromImage(testBitmap))
        {
            g.DrawImage(image, 0, 0, 1, 1);
        }
    }

    private void LoadExistingAnnotation(string imagePath)
    {
        string jsonPath = Path.ChangeExtension(imagePath, ".json");
        if (File.Exists(jsonPath))
        {
            try
            {
                var json = File.ReadAllText(jsonPath);
                var data = JsonSerializer.Deserialize<AnnotationData>(json);
                if (data?.Corners != null && data.Corners.Count == 4)
                {
                    corners = data.Corners.Select(c => new Point(c.X, c.Y)).ToList();
                }
            }
            catch { }
        }
    }

    private void UpdateStatus()
    {
        int annotated = imageFiles
            .Count(f => File.Exists(Path.ChangeExtension(f, ".json")));

        statusLabel.Text = $"进度: {currentIndex + 1}/{imageFiles.Count}  " +
                          $"已标注: {annotated}  " +
                          $"当前: {Path.GetFileName(imageFiles[currentIndex])}  " +
                          $"角点: {corners.Count}/4  " +
                          $"缩放: {zoomFactor:F1}x";

        saveButton.Enabled = corners.Count == 4;
    }

    private void PictureBox_MouseDown(object? sender, MouseEventArgs e)
    {
        if (currentImage == null || e.Button != MouseButtons.Left)
            return;

        // 如果已经有4个点，检查是否点击了某个角点
        if (corners.Count == 4)
        {
            // 检查是否点击了某个角点附近 (20像素范围内)
            for (int i = 0; i < corners.Count; i++)
            {
                var screenPoint = GetScreenCoordinates(corners[i]);
                if (screenPoint.HasValue)
                {
                    var distance = Math.Sqrt(
                        Math.Pow(screenPoint.Value.X - e.X, 2) +
                        Math.Pow(screenPoint.Value.Y - e.Y, 2)
                    );

                    if (distance < 20)
                    {
                        draggedPointIndex = i;
                        pictureBox.Cursor = Cursors.Hand;
                        return;
                    }
                }
            }
            // 如果没有点击角点，清除所有点重新开始
            corners.Clear();
            UpdateStatus();
            pictureBox.Invalidate();
        }

        // 添加新角点
        var imagePoint = GetImageCoordinates(e.Location);
        if (imagePoint.HasValue && corners.Count < 4)
        {
            corners.Add(imagePoint.Value);
            UpdateStatus();
            pictureBox.Invalidate();
        }
    }

    private void PictureBox_MouseMove(object? sender, MouseEventArgs e)
    {
        if (draggedPointIndex.HasValue && currentImage != null)
        {
            var imagePoint = GetImageCoordinates(e.Location);
            if (imagePoint.HasValue)
            {
                corners[draggedPointIndex.Value] = imagePoint.Value;
                pictureBox.Invalidate();
            }
        }
    }

    private void PictureBox_MouseUp(object? sender, MouseEventArgs e)
    {
        if (draggedPointIndex.HasValue)
        {
            draggedPointIndex = null;
            pictureBox.Cursor = Cursors.Default;
        }
    }

    private void PictureBox_MouseWheel(object? sender, MouseEventArgs e)
    {
        if (currentImage == null)
            return;

        float oldZoom = zoomFactor;

        // 滚轮向上放大，向下缩小
        if (e.Delta > 0)
            zoomFactor = Math.Min(zoomFactor * 1.2f, 10.0f);
        else
            zoomFactor = Math.Max(zoomFactor / 1.2f, 0.1f);

        // 计算以鼠标位置为中心的缩放偏移
        // 鼠标在 PictureBox 中的位置
        var mousePos = e.Location;

        // 鼠标在图片坐标系中的位置（缩放前）
        float imageX = (mousePos.X - zoomOffset.X) / oldZoom;
        float imageY = (mousePos.Y - zoomOffset.Y) / oldZoom;

        // 缩放后，保持鼠标指向的图片位置不变
        zoomOffset.X = (int)(mousePos.X - imageX * zoomFactor);
        zoomOffset.Y = (int)(mousePos.Y - imageY * zoomFactor);

        UpdateStatus();
        pictureBox.Invalidate();
    }

    private void PictureBox_Paint(object? sender, PaintEventArgs e)
    {
        if (currentImage == null)
            return;

        var g = e.Graphics;
        g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
        g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;

        // 绘制图片
        var imgWidth = currentImage.Width;
        var imgHeight = currentImage.Height;
        var boxWidth = pictureBox.Width;
        var boxHeight = pictureBox.Height;

        float baseScale = Math.Min((float)boxWidth / imgWidth, (float)boxHeight / imgHeight);
        float scale = baseScale * zoomFactor;

        int displayWidth = (int)(imgWidth * scale);
        int displayHeight = (int)(imgHeight * scale);
        int offsetX = (boxWidth - displayWidth) / 2 + zoomOffset.X;
        int offsetY = (boxHeight - displayHeight) / 2 + zoomOffset.Y;

        try
        {
            g.DrawImage(currentImage, offsetX, offsetY, displayWidth, displayHeight);
        }
        catch
        {
            // 绘制失败时显示错误提示
            g.Clear(Color.DarkGray);
            g.DrawString("图片绘制失败，按 → 跳过", new Font("Arial", 14), Brushes.Red, 10, 10);
            return;
        }

        // 绘制角点
        for (int i = 0; i < corners.Count; i++)
        {
            var screenPoint = GetScreenCoordinates(corners[i]);
            if (screenPoint.HasValue)
            {
                var p = screenPoint.Value;

                // 绘制圆圈
                g.FillEllipse(Brushes.Red, p.X - 8, p.Y - 8, 16, 16);
                g.DrawEllipse(new Pen(Color.White, 2), p.X - 8, p.Y - 8, 16, 16);

                // 绘制编号
                var label = (i + 1).ToString();
                g.DrawString(label, new Font("Arial", 10, FontStyle.Bold),
                           Brushes.White, p.X - 5, p.Y - 6);
            }
        }

        // 绘制连线
        if (corners.Count >= 2)
        {
            var pen = new Pen(Color.Lime, 2);
            for (int i = 0; i < corners.Count; i++)
            {
                var p1 = GetScreenCoordinates(corners[i]);
                var p2 = GetScreenCoordinates(corners[(i + 1) % corners.Count]);

                if (p1.HasValue && p2.HasValue)
                {
                    if (i < corners.Count - 1 || corners.Count == 4)
                        g.DrawLine(pen, p1.Value, p2.Value);
                }
            }
        }
    }

    private Point? GetImageCoordinates(Point screenPoint)
    {
        if (currentImage == null)
            return null;

        // 计算图片在 PictureBox 中的实际显示区域
        var imgWidth = currentImage.Width;
        var imgHeight = currentImage.Height;
        var boxWidth = pictureBox.Width;
        var boxHeight = pictureBox.Height;

        float baseScale = Math.Min((float)boxWidth / imgWidth, (float)boxHeight / imgHeight);
        float scale = baseScale * zoomFactor;

        int displayWidth = (int)(imgWidth * scale);
        int displayHeight = (int)(imgHeight * scale);
        int offsetX = (boxWidth - displayWidth) / 2 + zoomOffset.X;
        int offsetY = (boxHeight - displayHeight) / 2 + zoomOffset.Y;

        // 转换为图片坐标（允许超出边界）
        int imageX = (int)((screenPoint.X - offsetX) / scale);
        int imageY = (int)((screenPoint.Y - offsetY) / scale);

        // 不限制坐标范围，允许负数和超出图片尺寸
        // 这样可以标注画面外的角点
        return new Point(imageX, imageY);
    }

    private Point? GetScreenCoordinates(Point imagePoint)
    {
        if (currentImage == null)
            return null;

        var imgWidth = currentImage.Width;
        var imgHeight = currentImage.Height;
        var boxWidth = pictureBox.Width;
        var boxHeight = pictureBox.Height;

        float baseScale = Math.Min((float)boxWidth / imgWidth, (float)boxHeight / imgHeight);
        float scale = baseScale * zoomFactor;

        int displayWidth = (int)(imgWidth * scale);
        int displayHeight = (int)(imgHeight * scale);
        int offsetX = (boxWidth - displayWidth) / 2 + zoomOffset.X;
        int offsetY = (boxHeight - displayHeight) / 2 + zoomOffset.Y;

        int screenX = (int)(imagePoint.X * scale) + offsetX;
        int screenY = (int)(imagePoint.Y * scale) + offsetY;

        return new Point(screenX, screenY);
    }

    private void SaveButton_Click(object? sender, EventArgs e)
    {
        if (SaveCurrentAnnotation())
        {
            NavigateImage(1);
        }
    }

    private bool SaveCurrentAnnotation()
    {
        if (currentIndex < 0 || corners.Count != 4)
            return false;

        string imagePath = imageFiles[currentIndex];
        string jsonPath = Path.ChangeExtension(imagePath, ".json");

        var data = new AnnotationData
        {
            Corners = corners.Select(p => new CornerPoint { X = p.X, Y = p.Y }).ToList()
        };

        var json = JsonSerializer.Serialize(data, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(jsonPath, json);

        UpdateStatus();
        return true;
    }

    private void NavigateImage(int direction)
    {
        if (imageFiles.Count == 0)
            return;

        currentIndex += direction;

        if (currentIndex < 0)
            currentIndex = 0;
        else if (currentIndex >= imageFiles.Count)
            currentIndex = imageFiles.Count - 1;

        LoadCurrentImage();
    }

    private void Form1_KeyDown(object? sender, KeyEventArgs e)
    {
        switch (e.KeyCode)
        {
            case Keys.Enter:
                if (corners.Count == 4)
                {
                    if (SaveCurrentAnnotation())
                    {
                        NavigateImage(1);
                    }
                }
                e.Handled = true;
                break;

            case Keys.Space:
                NavigateImage(1);
                e.Handled = true;
                break;

            case Keys.Left:
                NavigateImage(-1);
                e.Handled = true;
                break;

            case Keys.Right:
                NavigateImage(1);
                e.Handled = true;
                break;

            case Keys.Back:
            case Keys.Escape:
                corners.Clear();
                UpdateStatus();
                pictureBox.Invalidate();
                e.Handled = true;
                break;

            case Keys.R:
                // 重置缩放
                zoomFactor = 1.0f;
                zoomOffset = Point.Empty;
                UpdateStatus();
                pictureBox.Invalidate();
                e.Handled = true;
                break;
        }
    }
}

// JSON 数据模型
public class AnnotationData
{
    public List<CornerPoint> Corners { get; set; } = new();
}

public class CornerPoint
{
    public int X { get; set; }
    public int Y { get; set; }
}
