using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Threading;

namespace CUETools.Updater
{
    internal class Program
    {
        const string UpdateFolder = ".cueupdate";
        static readonly string CurrentProcess = Path.GetFileNameWithoutExtension(Assembly.GetExecutingAssembly().Location);

        enum ResultCode
        {
            UnsupportedOutput = -6
            , OutputDirExists = -5
            , VerifyFailed = -4
            , FileLock = -3
            , IncorrectArguments = -2
            , WrongDirectory = -1
            , Success = 0    
        }

        static void PrintHelp()
        {
            Console.WriteLine("Usage: ");
            Console.WriteLine("CUETools.Updater Apply Update-X.Y.Z");
            Console.WriteLine(" - Do not prepend a path to the file.");
            Console.WriteLine("CUETools.Updater SelfUpdate");
        }

        static void Main(string[] args)
        {
            var updateDirectory = Path.Combine(Directory.GetCurrentDirectory(), UpdateFolder);
            if (!Directory.Exists(updateDirectory))
            {
                Directory.CreateDirectory(updateDirectory);
            }

            if (args.Length == 0)
            {
                PrintHelp();
                Environment.Exit((int)ResultCode.IncorrectArguments);
            }

            switch (args[0].ToLower())
            {
                case "apply":
                    try
                    {
                        var result = ApplyRoutine(args, updateDirectory);
                        if (result != ResultCode.Success) Environment.Exit((int)result);

                        TriggerSelfUpdate();
                    }
                    catch(Exception ex)
                    {
                        Console.WriteLine(ex.Message);
                        Console.ReadKey();
                    }
                    break;
                case "selfupdate":
                    try
                    {
                        SelfUpdateRoutine();
                    }
                    catch(Exception ex)
                    {
                        Console.WriteLine(ex.Message);
                        Console.ReadKey();
                    }
                    break;
                default:
                    PrintHelp();
                    break;
            }
        }

        static ResultCode ApplyRoutine(string[] args, string updateDirectory)
        {
            if (args.Length != 2)
            {
                PrintHelp();
                return ResultCode.IncorrectArguments;
            }

            Console.WriteLine("Checking if folder is writable.");

            int retry = 0;
            while(IsAnyFileLocked(Directory.GetCurrentDirectory()))
            {
                Thread.Sleep(500);

                Console.WriteLine("Some files are locked, please close them before updating.");
                if(++retry >= 10) return ResultCode.FileLock;
            }

            string zipPath = Path.Combine(updateDirectory, $"{args[1]}.zip");
            string hashPath = Path.Combine(updateDirectory, $"{args[1]}.sha256");

            if (!VerifyFile(zipPath, hashPath))
            {
                Console.WriteLine("Patch files are invalid.");
                return ResultCode.VerifyFailed;
            }

            Console.WriteLine("Patch has been verified.");
            Console.WriteLine("Extracting archive...");

            var extractPath = Path.Combine(updateDirectory, Path.GetFileNameWithoutExtension(zipPath));
            if (Directory.Exists(extractPath))
            {
                Console.WriteLine($"Folder '{extractPath}' already exists; Press Y to delete and continue.");
                if (Console.ReadKey().Key == ConsoleKey.Y)
                {
                    Console.WriteLine();
                    Console.WriteLine("Deleting folder...");
                    Directory.Delete(extractPath, true);
                }
                else
                {
                    Console.WriteLine("Can't extract archive.");

                    return ResultCode.OutputDirExists;
                }
            }

            Directory.CreateDirectory(extractPath);
            ZipFile.ExtractToDirectory(zipPath, extractPath);

            Console.WriteLine("Verifying output");

            var contentPath = GetContentPath(extractPath);
            if (string.IsNullOrWhiteSpace(contentPath))
            {
                Console.WriteLine("Extracted files are not in expected format.");

                return ResultCode.UnsupportedOutput;
            }

            Console.WriteLine("Applying files...");

            CopyFilesToDestination(contentPath, Directory.GetCurrentDirectory());

            Console.WriteLine("Cleaning up");

            Directory.Delete(extractPath, true);

            DeleteUnsupportedFiles(Directory.GetCurrentDirectory());

            return ResultCode.Success;
        }

        static void TriggerSelfUpdate()
        {
            var exeLocation = Path.Combine(Directory.GetCurrentDirectory(), UpdateFolder, CurrentProcess + ".exe");
            if (!File.Exists(exeLocation))
            {
                Console.WriteLine("Self Update not found.");
                return;
            }

            Console.WriteLine("Start self updating");

            string arguments = "SelfUpdate";

            Process.Start(new ProcessStartInfo
            {
                FileName = exeLocation,
                Arguments = arguments,
                UseShellExecute = true,
                RedirectStandardOutput = false,
                RedirectStandardError = false,
                CreateNoWindow = false
            });
        }

        static void SelfUpdateRoutine()
        {
            var currentDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            if (currentDir == null) return;

            var parentDir = Directory.GetParent(currentDir);
            if (parentDir == null) return;

            var targetExecutable = Path.Combine(parentDir.FullName, CurrentProcess + ".exe");

            if(!File.Exists(targetExecutable)) return; // Are we running in an unexpected directory?

            int retry = 0;
            while (IsFileLocked(targetExecutable))
            {
                Thread.Sleep(500);

                if (++retry >= 10)
                {
                    Console.WriteLine("Can't replace file as it's opened by another program.");
                    Console.ReadKey();
                    return;
                }
            }

            var files = Directory.GetFiles(currentDir, $"{CurrentProcess}*", SearchOption.AllDirectories);
            foreach(var file in files)
            {
                File.Copy(file, Path.Combine(parentDir.FullName, Path.GetFileName(file)), true);
            }
        }

        static string GetSHA256Hash(string filePath)
        {
            using SHA256 sha256 = SHA256.Create();
            using FileStream stream = File.OpenRead(filePath);

            byte[] hashBytes = sha256.ComputeHash(stream);

            var hashBuilder = new StringBuilder();
            for (int i = 0; i < hashBytes.Length; ++i)
            {
                hashBuilder.Append(hashBytes[i].ToString("x2"));
            }

            return hashBuilder.ToString();
        }

        static string ReadSHA256FromHashFile(string hashFile)
        {
            var fileContent = File.ReadAllLines(hashFile);
            if (fileContent.Length == 0) return string.Empty;

            return fileContent[0].Split(' ')[0];
        }

        static bool VerifyFile(string zipFile, string hashFile)
        {
            var actualHash = GetSHA256Hash(zipFile);
            var validationHash = ReadSHA256FromHashFile(hashFile);

            return string.Compare(actualHash, validationHash, true) == 0;
        }

        static bool IsFileLocked(string filePath)
        {
            if (!File.Exists(filePath)) return false;

            try
            {
                using FileStream stream = File.Open(filePath, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
                return false;
            }
            catch (IOException)
            {
                return true;
            }
        }

        static bool IsAnyFileLocked(string dir)
        {
            bool result = false;
            foreach (var file in Directory.GetFiles(dir, "*", SearchOption.AllDirectories))
            {
                if (IsFileLocked(file))
                {
                    if (Path.GetDirectoryName(file) != Directory.GetCurrentDirectory())
                    {
                        Console.WriteLine($"File locked: {file}");
                        result = true;
                    }
                }
            }

            return result;
        }

        static string GetContentPath(string output)
        {
            var dirs = Directory.GetDirectories(output, "*", SearchOption.TopDirectoryOnly);
            if (dirs.Length != 1) return string.Empty;
            var files = Directory.GetFiles(dirs[0], "*", SearchOption.TopDirectoryOnly);

            return files.Any(f => f.EndsWith("License.txt"))
                ? dirs[0]
                : string.Empty;
        }

        static string GetRelativePath(string basePath, string filePath)
        {
            var baseUri = new Uri(basePath.TrimEnd('\\') + '\\');
            var fileUri = new Uri(filePath);
            
            var relativeUri = baseUri.MakeRelativeUri(fileUri)
                .ToString()
                .Replace('/', Path.DirectorySeparatorChar);

            return Uri.UnescapeDataString(relativeUri);
        }

        static void CopyFilesToDestination(string srcPath, string destPath)
        {
            var files = Directory.GetFiles(srcPath, "*", SearchOption.AllDirectories);

            foreach (var file in files)
            {
                bool isSelf = Path.GetFileName(file)
                    .StartsWith(CurrentProcess, StringComparison.InvariantCultureIgnoreCase);

                string relativePath = GetRelativePath(srcPath, file);
                string destFilePath = isSelf 
                    ? Path.Combine(destPath, UpdateFolder, relativePath)
                    : Path.Combine(destPath, relativePath);

                var path = Path.GetDirectoryName(destFilePath) ?? throw new IOException("Can't determine directory name.");
                Directory.CreateDirectory(path);

                if (File.Exists(destFilePath))
                {
                    File.Delete(destFilePath);
                }

                File.Move(file, destFilePath);
            }
        }

        static void DeleteUnsupportedFiles(string path)
        {
            // Not implemented.

/* 
            var files = Directory.GetFiles(path, "*", SearchOption.AllDirectories);
            foreach (var file in files)
            {
            } 
*/
        }
    }
}
