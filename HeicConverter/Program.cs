using ImageMagick;

class Program
{
    static void Main(string[] args)
    {
        string folder = args.Length > 0 ? args[0] : @"D:\Programing\C#\MauiScan\AnnotationTool\data\ios";

        Console.WriteLine($"Scanning folder: {folder}");

        var files = Directory.GetFiles(folder, "*.png")
            .Concat(Directory.GetFiles(folder, "*.heic"))
            .Concat(Directory.GetFiles(folder, "*.HEIC"))
            .ToList();

        Console.WriteLine($"Found {files.Count} files to check\n");

        int converted = 0;
        int skipped = 0;
        int failed = 0;

        foreach (var file in files)
        {
            string fileName = Path.GetFileName(file);

            // Check if file is actually HEIC by reading header
            byte[] header = new byte[12];
            using (var fs = new FileStream(file, FileMode.Open, FileAccess.Read))
            {
                fs.Read(header, 0, 12);
            }

            // Check for HEIC signature: ftyp heic (at offset 4)
            bool isHeic = header[4] == 0x66 && header[5] == 0x74 && header[6] == 0x79 && header[7] == 0x70 &&
                          header[8] == 0x68 && header[9] == 0x65 && header[10] == 0x69 && header[11] == 0x63;

            // Check for PNG signature
            bool isPng = header[0] == 0x89 && header[1] == 0x50 && header[2] == 0x4E && header[3] == 0x47;

            // Check for JPEG signature
            bool isJpeg = header[0] == 0xFF && header[1] == 0xD8;

            if (isPng || isJpeg)
            {
                Console.WriteLine($"[SKIP] {fileName} - Already valid format");
                skipped++;
                continue;
            }

            if (!isHeic)
            {
                Console.WriteLine($"[SKIP] {fileName} - Unknown format");
                skipped++;
                continue;
            }

            // Convert HEIC to JPG
            string outputPath = Path.ChangeExtension(file, ".jpg");

            try
            {
                Console.Write($"[CONV] {fileName} -> ");

                using (var image = new MagickImage(file))
                {
                    image.Format = MagickFormat.Jpeg;
                    image.Quality = 90;
                    image.Write(outputPath);
                }

                // Delete original file
                File.Delete(file);

                Console.WriteLine($"{Path.GetFileName(outputPath)} OK");
                converted++;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"FAILED: {ex.Message}");
                failed++;
            }
        }

        Console.WriteLine($"\n========================================");
        Console.WriteLine($"Converted: {converted}");
        Console.WriteLine($"Skipped: {skipped}");
        Console.WriteLine($"Failed: {failed}");
    }
}
