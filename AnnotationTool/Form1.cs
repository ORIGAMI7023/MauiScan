using System.Text.Json;

namespace AnnotationTool;

public partial class Form1 : Form
{
    private List<string> imageFiles = new();
    private int currentIndex = -1;
    private Image? currentImage;

    // 双模式角点存储
    private Dictionary<AnnotationMode, List<Point>> allCorners = new()
    {
        [AnnotationMode.PptBorder] = new List<Point>(),
        [AnnotationMode.ScreenEdge] = new List<Point>()
    };

    // 当前选中的模式
    private AnnotationMode currentMode = AnnotationMode.PptBorder;

    // 便捷属性：当前模式的角点（保持向后兼容）
    private List<Point> Corners
    {
        get => allCorners[currentMode];
        set => allCorners[currentMode] = value;
    }

    private int? draggedPointIndex = null;
    private float zoomFactor = 1.0f;
    private Point zoomOffset = Point.Empty;
    private bool isMiddleButtonDragging = false;
    private Point lastMiddleButtonPosition = Point.Empty;

    private PictureBox pictureBox = new();
    private Label statusLabel = new();
    private Button prevButton = new();
    private Button nextButton = new();
    private Button loadFolderButton = new();
    private Button saveButton = new();
    private Label instructionLabel = new();
    private ComboBox modeComboBox = new();  // 模式选择下拉框
    private CheckBox quickZoomCheckBox = new();  // 快速缩放复选框

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

        // 模式选择下拉框
        modeComboBox.Location = new Point(rightPanelX, 95);
        modeComboBox.Size = new Size(250, 30);
        modeComboBox.DropDownStyle = ComboBoxStyle.DropDownList;
        modeComboBox.Anchor = AnchorStyles.Top | AnchorStyles.Right;
        modeComboBox.Items.Add("PPT 边框（机器学习）");
        modeComboBox.Items.Add("幕布边缘（传统算法）");
        modeComboBox.SelectedIndex = 0;
        modeComboBox.TabStop = false;
        modeComboBox.SelectedIndexChanged += ModeComboBox_SelectedIndexChanged;
        this.Controls.Add(modeComboBox);

        // 快速缩放复选框
        quickZoomCheckBox.Location = new Point(rightPanelX, 130);
        quickZoomCheckBox.Size = new Size(250, 25);
        quickZoomCheckBox.Text = "快速缩放";
        quickZoomCheckBox.Anchor = AnchorStyles.Top | AnchorStyles.Right;
        quickZoomCheckBox.TabStop = false;
        quickZoomCheckBox.CheckedChanged += QuickZoomCheckBox_CheckedChanged;
        this.Controls.Add(quickZoomCheckBox);

        // 说明文字
        instructionLabel.Location = new Point(rightPanelX, 160);
        instructionLabel.Size = new Size(250, 220);
        instructionLabel.Anchor = AnchorStyles.Top | AnchorStyles.Right;
        instructionLabel.Text = "操作说明：\n\n" +
                               "1. 选择标注模式\n\n" +
                               "2. 点击选择四个角点\n" +
                               "   (左上→右上→右下→左下)\n\n" +
                               "3. 拖拽调整角点位置\n\n" +
                               "4. Tab - 切换模式\n" +
                               "5. 滚轮 - 缩放/快速缩放\n" +
                               "6. 中键 - 拖动图片\n" +
                               "7. R - 重置视图\n\n" +
                               "8. Enter - 保存并下一张\n" +
                               "9. Space - 撤销上一个点\n" +
                               "10. Esc - 清除当前模式角点";
        this.Controls.Add(instructionLabel);

        // 上一张按钮
        prevButton.Location = new Point(rightPanelX, 420);
        prevButton.Size = new Size(120, 40);
        prevButton.Text = "上一张 (←)";
        prevButton.Anchor = AnchorStyles.Top | AnchorStyles.Right;
        prevButton.TabStop = false;
        prevButton.Click += (s, e) => NavigateImage(-1);
        this.Controls.Add(prevButton);

        // 下一张按钮
        nextButton.Location = new Point(rightPanelX + 130, 420);
        nextButton.Size = new Size(120, 40);
        nextButton.Text = "下一张 (→)";
        nextButton.Anchor = AnchorStyles.Top | AnchorStyles.Right;
        nextButton.TabStop = false;
        nextButton.Click += (s, e) => NavigateImage(1);
        this.Controls.Add(nextButton);

        // 保存按钮
        saveButton.Location = new Point(rightPanelX, 470);
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
        var allImages = Directory.GetFiles(folderPath)
            .Where(f => extensions.Contains(Path.GetExtension(f).ToLower()))
            .OrderBy(f => f)
            .ToList();

        if (allImages.Count == 0)
        {
            MessageBox.Show("文件夹中没有找到图片文件！", "错误", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return;
        }

        // 将已标注的图片放在最后
        var unannotated = allImages.Where(f => !File.Exists(Path.ChangeExtension(f, ".json"))).ToList();
        var annotated = allImages.Where(f => File.Exists(Path.ChangeExtension(f, ".json"))).ToList();
        imageFiles = unannotated.Concat(annotated).ToList();

        currentIndex = 0;
        LoadCurrentImage();
    }

    private void LoadCurrentImage()
    {
        if (currentIndex < 0 || currentIndex >= imageFiles.Count)
            return;

        currentImage?.Dispose();
        allCorners[AnnotationMode.PptBorder].Clear();
        allCorners[AnnotationMode.ScreenEdge].Clear();
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
                // 校正 EXIF 方向
                var correctedImage = CorrectImageOrientation(tempImage);

                // 立即转换为标准格式，解决色彩空间和特殊编码问题
                var standardBitmap = new Bitmap(correctedImage.Width, correctedImage.Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                using (var g = Graphics.FromImage(standardBitmap))
                {
                    g.DrawImage(correctedImage, 0, 0, correctedImage.Width, correctedImage.Height);
                }
                correctedImage.Dispose();

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
                // 校正 EXIF 方向
                var correctedImage = CorrectImageOrientation(bitmap);

                var standardBitmap = new Bitmap(correctedImage.Width, correctedImage.Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                using (var g = Graphics.FromImage(standardBitmap))
                {
                    g.DrawImage(correctedImage, 0, 0, correctedImage.Width, correctedImage.Height);
                }
                correctedImage.Dispose();

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
                // 校正 EXIF 方向
                var correctedImage = CorrectImageOrientation(tempImage);

                var standardBitmap = new Bitmap(correctedImage.Width, correctedImage.Height, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
                using (var g = Graphics.FromImage(standardBitmap))
                {
                    g.DrawImage(correctedImage, 0, 0, correctedImage.Width, correctedImage.Height);
                }
                correctedImage.Dispose();

                ValidateImage(standardBitmap);
                return standardBitmap;
            }
        }
        catch { }

        return null;
    }

    /// <summary>
    /// 根据 EXIF 方向信息校正图片方向
    /// </summary>
    private Image CorrectImageOrientation(Image image)
    {
        // 读取 EXIF 方向标签
        const int PropertyTagOrientation = 0x0112;
        if (!image.PropertyIdList.Contains(PropertyTagOrientation))
        {
            // 没有 EXIF 方向信息，返回原图
            return new Bitmap(image);
        }

        var prop = image.GetPropertyItem(PropertyTagOrientation);
        if (prop == null || prop.Value.Length < 1)
        {
            return new Bitmap(image);
        }

        ushort orientation = prop.Value[0];

        // 根据方向值旋转/翻转图片
        Bitmap rotated;

        switch (orientation)
        {
            case 1: // 无旋转
                return new Bitmap(image);

            case 2: // 水平翻转
                rotated = new Bitmap(image);
                rotated.RotateFlip(RotateFlipType.RotateNoneFlipX);
                return rotated;

            case 3: // 旋转 180 度
                rotated = new Bitmap(image);
                rotated.RotateFlip(RotateFlipType.Rotate180FlipNone);
                return rotated;

            case 4: // 垂直翻转
                rotated = new Bitmap(image);
                rotated.RotateFlip(RotateFlipType.RotateNoneFlipY);
                return rotated;

            case 5: // 水平翻转 + 逆时针 90 度
                rotated = new Bitmap(image);
                rotated.RotateFlip(RotateFlipType.Rotate90FlipX);
                return rotated;

            case 6: // 顺时针 90 度（最常见）
                rotated = new Bitmap(image);
                rotated.RotateFlip(RotateFlipType.Rotate90FlipNone);
                return rotated;

            case 7: // 水平翻转 + 顺时针 90 度
                rotated = new Bitmap(image);
                rotated.RotateFlip(RotateFlipType.Rotate270FlipX);
                return rotated;

            case 8: // 逆时针 90 度
                rotated = new Bitmap(image);
                rotated.RotateFlip(RotateFlipType.Rotate270FlipNone);
                return rotated;

            default:
                return new Bitmap(image);
        }
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

                if (data == null)
                    return;

                // 清空所有模式的角点
                allCorners[AnnotationMode.PptBorder].Clear();
                allCorners[AnnotationMode.ScreenEdge].Clear();

                // 判断是新格式还是旧格式
                if (data.Corners != null && data.PptCorners == null && data.ScreenCorners == null)
                {
                    // 旧格式：Corners 字段存在，识别为 PPT 边框
                    allCorners[AnnotationMode.PptBorder] = data.Corners
                        .Select(c => new Point(c.X, c.Y))
                        .ToList();
                }
                else
                {
                    // 新格式：读取双模式字段
                    if (data.PptCorners != null && data.PptCorners.Count == 4)
                    {
                        allCorners[AnnotationMode.PptBorder] = data.PptCorners
                            .Select(c => new Point(c.X, c.Y))
                            .ToList();
                    }

                    if (data.ScreenCorners != null && data.ScreenCorners.Count == 4)
                    {
                        allCorners[AnnotationMode.ScreenEdge] = data.ScreenCorners
                            .Select(c => new Point(c.X, c.Y))
                            .ToList();
                    }
                }
            }
            catch (Exception ex)
            {
                // 加载失败时保持默认值（空角点）
                MessageBox.Show(
                    $"加载标注数据失败: {ex.Message}\n\n将作为新图片处理。",
                    "加载错误",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Warning
                );
            }
        }
    }

    private void ModeComboBox_SelectedIndexChanged(object? sender, EventArgs e)
    {
        SwitchToMode((AnnotationMode)modeComboBox.SelectedIndex);
    }

    private void QuickZoomCheckBox_CheckedChanged(object? sender, EventArgs e)
    {
        // 开启快速缩放模式时，自动切换到 0.75x 并居中
        if (quickZoomCheckBox.Checked)
        {
            zoomFactor = 0.75f;
            zoomOffset = Point.Empty;  // 居中放置
            UpdateStatus();
            pictureBox.Invalidate();
        }
    }

    private void SwitchToMode(AnnotationMode newMode)
    {
        if (newMode == currentMode)
            return;

        currentMode = newMode;
        UpdateStatus();
        pictureBox.Invalidate(); // 重新绘制（边框颜色会改变）
    }

    private void UpdateStatus()
    {
        int annotated = imageFiles
            .Count(f => File.Exists(Path.ChangeExtension(f, ".json")));

        // 检查是否已加载图片
        if (imageFiles.Count > 0 && currentIndex >= 0 && currentIndex < imageFiles.Count)
        {
            statusLabel.Text = $"进度: {currentIndex + 1}/{imageFiles.Count}  " +
                              $"已标注: {annotated}  " +
                              $"当前: {Path.GetFileName(imageFiles[currentIndex])}  " +
                              $"模式: {currentMode.GetDisplayName()}  " +
                              $"角点: {Corners.Count}/4  " +
                              $"缩放: {zoomFactor:F1}x";
        }
        else
        {
            statusLabel.Text = $"请加载图片文件夹  " +
                              $"模式: {currentMode.GetDisplayName()}  " +
                              $"缩放: {zoomFactor:F1}x";
        }

        // 只有当前模式的角点完整时才允许保存
        saveButton.Enabled = Corners.Count == 4;

        // 同步下拉框选中项（防止循环触发）
        modeComboBox.SelectedIndexChanged -= ModeComboBox_SelectedIndexChanged;
        modeComboBox.SelectedIndex = (int)currentMode;
        modeComboBox.SelectedIndexChanged += ModeComboBox_SelectedIndexChanged;
    }

    private void PictureBox_MouseDown(object? sender, MouseEventArgs e)
    {
        if (currentImage == null)
            return;

        // 中键拖动图片
        if (e.Button == MouseButtons.Middle)
        {
            isMiddleButtonDragging = true;
            lastMiddleButtonPosition = e.Location;
            pictureBox.Cursor = Cursors.SizeAll;
            return;
        }

        if (e.Button != MouseButtons.Left)
            return;

        // 如果已经有4个点，检查是否点击了某个角点
        if (Corners.Count == 4)
        {
            // 检查是否点击了某个角点附近 (20像素范围内)
            for (int i = 0; i < Corners.Count; i++)
            {
                var screenPoint = GetScreenCoordinates(Corners[i]);
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
            Corners.Clear();
            UpdateStatus();
            pictureBox.Invalidate();
        }

        // 添加新角点
        var imagePoint = GetImageCoordinates(e.Location);
        if (imagePoint.HasValue && Corners.Count < 4)
        {
            Corners.Add(imagePoint.Value);
            UpdateStatus();
            pictureBox.Invalidate();
        }
    }

    private void PictureBox_MouseMove(object? sender, MouseEventArgs e)
    {
        if (currentImage == null)
            return;

        // 中键拖动图片
        if (isMiddleButtonDragging)
        {
            int deltaX = e.Location.X - lastMiddleButtonPosition.X;
            int deltaY = e.Location.Y - lastMiddleButtonPosition.Y;

            zoomOffset.X += deltaX;
            zoomOffset.Y += deltaY;

            lastMiddleButtonPosition = e.Location;
            pictureBox.Invalidate();
            return;
        }

        // 拖动角点
        if (draggedPointIndex.HasValue)
        {
            var imagePoint = GetImageCoordinates(e.Location);
            if (imagePoint.HasValue)
            {
                Corners[draggedPointIndex.Value] = imagePoint.Value;
                pictureBox.Invalidate();
            }
        }
    }

    private void PictureBox_MouseUp(object? sender, MouseEventArgs e)
    {
        // 释放中键拖动
        if (e.Button == MouseButtons.Middle && isMiddleButtonDragging)
        {
            isMiddleButtonDragging = false;
            pictureBox.Cursor = Cursors.Default;
        }

        // 释放角点拖动
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

        // 快速缩放模式：0.75, 1, 3.5, 6.5, 10 五个档位
        if (quickZoomCheckBox.Checked)
        {
            // 定义所有档位
            float[] zoomLevels = { 0.75f, 1.0f, 3.5f, 6.5f, 10.0f };

            if (e.Delta > 0)  // 滚轮向上：放大
            {
                // 找到当前档位的索引，切换到下一个档位
                int currentIndex = Array.FindIndex(zoomLevels, z => z >= zoomFactor - 0.01f);
                if (currentIndex >= 0 && currentIndex < zoomLevels.Length - 1)
                    zoomFactor = zoomLevels[currentIndex + 1];
                else
                    zoomFactor = zoomLevels[zoomLevels.Length - 1]; // 已是最大
            }
            else  // 滚轮向下：缩小
            {
                // 找到当前档位的索引，切换到上一个档位
                int currentIndex = Array.FindIndex(zoomLevels, z => z >= zoomFactor - 0.01f);
                if (currentIndex > 0)
                    zoomFactor = zoomLevels[currentIndex - 1];
                else if (currentIndex == 0)
                    zoomFactor = zoomLevels[0]; // 已是最小
                else if (zoomFactor > zoomLevels[zoomLevels.Length - 1])
                    zoomFactor = zoomLevels[zoomLevels.Length - 1]; // 超过最大值，回到最大
                else
                    zoomFactor = zoomLevels[0]; // 小于最小值，回到最小
            }

            // 切换到 0.75 倍率时，居中放置
            if (zoomFactor == 0.75f && oldZoom != 0.75f)
            {
                zoomOffset = Point.Empty;
            }
        }
        else
        {
            // 普通缩放模式
            // 滚轮向上放大，向下缩小
            if (e.Delta > 0)
                zoomFactor = Math.Min(zoomFactor * 1.4f, 10.0f);
            else
                zoomFactor = Math.Max(zoomFactor / 1.4f, 0.1f);
        }

        // 计算基础缩放比例
        var imgWidth = currentImage.Width;
        var imgHeight = currentImage.Height;
        var boxWidth = pictureBox.Width;
        var boxHeight = pictureBox.Height;
        float baseScale = Math.Min((float)boxWidth / imgWidth, (float)boxHeight / imgHeight);

        // 计算缩放前后的实际缩放比例
        float oldScale = baseScale * oldZoom;
        float newScale = baseScale * zoomFactor;

        // 计算缩放前图片的居中偏移
        int oldDisplayWidth = (int)(imgWidth * oldScale);
        int oldDisplayHeight = (int)(imgHeight * oldScale);
        int oldCenterOffsetX = (boxWidth - oldDisplayWidth) / 2;
        int oldCenterOffsetY = (boxHeight - oldDisplayHeight) / 2;

        // 计算缩放后图片的居中偏移
        int newDisplayWidth = (int)(imgWidth * newScale);
        int newDisplayHeight = (int)(imgHeight * newScale);
        int newCenterOffsetX = (boxWidth - newDisplayWidth) / 2;
        int newCenterOffsetY = (boxHeight - newDisplayHeight) / 2;

        // 鼠标在 PictureBox 中的位置
        var mousePos = e.Location;

        // 鼠标相对于图片左上角的位置（缩放前）
        float relX = mousePos.X - oldCenterOffsetX - zoomOffset.X;
        float relY = mousePos.Y - oldCenterOffsetY - zoomOffset.Y;

        // 鼠标在原始图片上的位置
        float imgX = relX / oldScale;
        float imgY = relY / oldScale;

        // 缩放后，计算新的偏移以保持鼠标指向位置不变
        float newRelX = imgX * newScale;
        float newRelY = imgY * newScale;

        zoomOffset.X = (int)(mousePos.X - newCenterOffsetX - newRelX);
        zoomOffset.Y = (int)(mousePos.Y - newCenterOffsetY - newRelY);

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

        // 绘制两种模式的边框
        foreach (var mode in new[] { AnnotationMode.PptBorder, AnnotationMode.ScreenEdge })
        {
            var modeCorners = allCorners[mode];
            if (modeCorners.Count >= 2)
            {
                var borderColor = mode.GetBorderColor();
                using var pen = new Pen(borderColor, 2);

                // 非当前模式使用虚线
                if (mode != currentMode)
                {
                    pen.DashStyle = System.Drawing.Drawing2D.DashStyle.Dash;
                    pen.DashPattern = new float[] { 5, 5 };
                    pen.DashCap = System.Drawing.Drawing2D.DashCap.Flat;
                }

                for (int i = 0; i < modeCorners.Count; i++)
                {
                    var p1 = GetScreenCoordinates(modeCorners[i]);
                    var p2 = GetScreenCoordinates(modeCorners[(i + 1) % modeCorners.Count]);

                    if (p1.HasValue && p2.HasValue)
                    {
                        if (i < modeCorners.Count - 1 || modeCorners.Count == 4)
                            g.DrawLine(pen, p1.Value, p2.Value);
                    }
                }
            }
        }

        // 绘制当前模式的角点（强调显示）
        for (int i = 0; i < Corners.Count; i++)
        {
            var screenPoint = GetScreenCoordinates(Corners[i]);
            if (screenPoint.HasValue)
            {
                var p = screenPoint.Value;
                var borderColor = currentMode.GetBorderColor();

                // 绘制彩色圆圈
                g.FillEllipse(new SolidBrush(borderColor), p.X - 8, p.Y - 8, 16, 16);
                g.DrawEllipse(new Pen(Color.White, 2), p.X - 8, p.Y - 8, 16, 16);

                // 绘制编号
                var label = (i + 1).ToString();
                g.DrawString(label, new Font("Arial", 10, FontStyle.Bold),
                           Brushes.White, p.X - 5, p.Y - 6);
            }
        }

        // 绘制非当前模式的角点（淡化显示）
        foreach (var mode in new[] { AnnotationMode.PptBorder, AnnotationMode.ScreenEdge })
        {
            if (mode == currentMode) continue;

            var modeCorners = allCorners[mode];
            for (int i = 0; i < modeCorners.Count; i++)
            {
                var screenPoint = GetScreenCoordinates(modeCorners[i]);
                if (screenPoint.HasValue)
                {
                    var p = screenPoint.Value;
                    var borderColor = mode.GetBorderColor();

                    // 绘制小号空心圆圈
                    g.DrawEllipse(new Pen(borderColor, 1), p.X - 5, p.Y - 5, 10, 10);
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
        if (currentIndex < 0 || Corners.Count != 4)
            return false;

        string imagePath = imageFiles[currentIndex];
        string jsonPath = Path.ChangeExtension(imagePath, ".json");

        // 加载已有数据（如果存在）
        AnnotationData data;
        if (File.Exists(jsonPath))
        {
            try
            {
                var existingJson = File.ReadAllText(jsonPath);
                var existingData = JsonSerializer.Deserialize<AnnotationData>(existingJson);
                data = existingData ?? new AnnotationData();
            }
            catch
            {
                data = new AnnotationData();
            }
        }
        else
        {
            data = new AnnotationData();
        }

        // 更新当前模式的角点
        if (currentMode == AnnotationMode.PptBorder)
        {
            data.PptCorners = allCorners[AnnotationMode.PptBorder]
                .Select(p => new CornerPoint { X = p.X, Y = p.Y })
                .ToList();
        }
        else
        {
            data.ScreenCorners = allCorners[AnnotationMode.ScreenEdge]
                .Select(p => new CornerPoint { X = p.X, Y = p.Y })
                .ToList();
        }

        // 序列化选项（忽略 null 值）
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull
        };

        var json = JsonSerializer.Serialize(data, options);
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
            case Keys.Tab:
                // Tab 键切换模式
                var newMode = currentMode == AnnotationMode.PptBorder
                    ? AnnotationMode.ScreenEdge
                    : AnnotationMode.PptBorder;
                SwitchToMode(newMode);
                e.Handled = true;
                e.SuppressKeyPress = true;
                break;

            case Keys.Enter:
                if (Corners.Count == 4)
                {
                    if (SaveCurrentAnnotation())
                    {
                        NavigateImage(1);
                    }
                }
                else
                {
                    // 提示用户当前模式角点未完成
                    var unfinishedMode = currentMode == AnnotationMode.PptBorder
                        ? "PPT 边框"
                        : "幕布边缘";
                    MessageBox.Show(
                        $"当前{unfinishedMode}模式仅标注了 {Corners.Count} 个角点，需要 4 个角点才能保存。",
                        "角点未完成",
                        MessageBoxButtons.OK,
                        MessageBoxIcon.Information
                    );
                }
                e.Handled = true;
                break;

            case Keys.Space:
                // 撤销上一个点
                if (Corners.Count > 0)
                {
                    Corners.RemoveAt(Corners.Count - 1);
                    UpdateStatus();
                    pictureBox.Invalidate();
                }
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
                Corners.Clear();
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

// 标注模式枚举
public enum AnnotationMode
{
    PptBorder,      // PPT 边框（绿色）
    ScreenEdge      // 幕布边缘（蓝色）
}

// 枚举扩展方法
public static class AnnotationModeExtensions
{
    public static string GetDisplayName(this AnnotationMode mode)
    {
        return mode switch
        {
            AnnotationMode.PptBorder => "PPT 边框",
            AnnotationMode.ScreenEdge => "幕布边缘",
            _ => "未知模式"
        };
    }

    public static Color GetBorderColor(this AnnotationMode mode)
    {
        return mode switch
        {
            AnnotationMode.PptBorder => Color.Lime,        // 绿色
            AnnotationMode.ScreenEdge => Color.DodgerBlue,  // 蓝色
            _ => Color.White
        };
    }
}

// JSON 数据模型
public class AnnotationData
{
    // 新格式：双模式字段
    public List<CornerPoint>? PptCorners { get; set; }
    public List<CornerPoint>? ScreenCorners { get; set; }

    // 旧格式字段（向后兼容，仅用于读取）
    public List<CornerPoint>? Corners { get; set; }
}

public class CornerPoint
{
    public int X { get; set; }
    public int Y { get; set; }
}
