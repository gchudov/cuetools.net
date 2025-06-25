using CUERipper.Avalonia.Configuration.Abstractions;
using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Models.Github;
using CUERipper.Avalonia.Services.Abstractions;
using CUETools.Processor;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.Services
{
    public class UpdateService : IUpdateService
    {
        private readonly HttpClient _httpClient;
        private readonly ICUEConfigFacade _config;
        private readonly ILogger _logger;

        public UpdateMetadata? UpdateMetadata { get; private set; }

        public UpdateService(HttpClient httpClient
            , ICUEConfigFacade config
            , ILogger<UpdateService> logger)
        {
            _httpClient = httpClient;
            _config = config;
            _logger = logger;
        }

        public async Task<bool> FetchAsync()
        {
            if (UpdateMetadata != null) return true;

            if (!_config.CheckForUpdates)
            {
                _logger.LogWarning("Skip checking for updates.");
                return false;
            }

            var latestRelease = await GetLatestReleaseAsync();
            if (latestRelease.Content == null)
            {
                _logger.LogWarning("No releases found.");
                return false;
            }

            string versionPattern = @"^v\d+\.\d+\.\d+[a-zA-Z]?$";
            Regex regex = new(versionPattern);
            if (!regex.IsMatch(latestRelease.Content.TagName))
            {
                _logger.LogError("Release tag '{TAG}' doesn't match expected format.", latestRelease.Content.TagName);
                return false;
            }

            var setupAsset = GetSetupAsset(latestRelease.Content);
            var hashAsset = GetHashAsset(latestRelease.Content);
            if (setupAsset == null || hashAsset == null)
            {
                _logger.LogWarning("Github assets are incomplete.");
                return false;
            }

            UpdateMetadata = new UpdateMetadata(
                Version: latestRelease.Content.TagName.Substring(1)
                , CurrentVersion: CUESheet.CUEToolsVersion
                , Author: string.IsNullOrWhiteSpace(latestRelease.Author)
                            ? Constants.ApplicationName
                            : latestRelease.Author!
                , Description: latestRelease.Content.Body
                , Uri: setupAsset.BrowserDownloadUrl
                , Size: setupAsset.Size
                , HashUri: hashAsset.BrowserDownloadUrl
                , HashSize: hashAsset.Size
                , Date: latestRelease.Content.PublishedAt
            );

            return true;
        }

        private async Task<GithubReleaseContainer> GetLatestReleaseAsync()
        {
            GithubReleaseContainer latestRelease = GetLatestReleaseFromDiskCache();
            if (latestRelease.IsFromCache) return latestRelease; 

            try
            {
                using var result = await _httpClient.GetAsync(Constants.GithubApiUri);
                result.EnsureSuccessStatusCode();

                string response = await result.Content.ReadAsStringAsync();
                var githubReleases = JsonConvert.DeserializeObject<GithubRelease[]>(response)
                    ?? throw new NullReferenceException("Failed to deserialize object...");

                var recentRelease = githubReleases
                    .Where(r => !r.Draft && !r.PreRelease && r.TargetCommitish == Constants.GithubBranch)
                    .OrderByDescending(r => r.PublishedAt)
                    .FirstOrDefault();

                string? author = null;
                if (recentRelease != null)
                {
                    author = await GetAuthorAsync(recentRelease);
                }

                WriteReleasesToDiskCache(recentRelease, author);
                return new GithubReleaseContainer(false, recentRelease, author);
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to retrieve latest release.");
            }

            return new GithubReleaseContainer(false, null, null);
        }


        /// <summary>
        /// Mechanism that prevents spamming the GitHub API by limiting automated requests to once every 3 days.
        /// </summary>
        /// <returns></returns>
        private GithubReleaseContainer GetLatestReleaseFromDiskCache()
        {
            if (!File.Exists(Constants.PathUpdateCacheFile)) return new(false, null, null);

            string[] content = File.ReadAllLines(Constants.PathUpdateCacheFile);
            if (content.Length != 4)
            {
                _logger.LogError("Content of {File} is incorrect.", Constants.PathUpdateCacheFile);
                return new(false, null, null);
            }

            if (!DateTime.TryParseExact(content[0], "yyyyMMdd", null, System.Globalization.DateTimeStyles.None
                , out DateTime lastUpdateCheck))
            {
                _logger.LogError("Content of {File} is incorrect, can't parse to datetime.", Constants.PathUpdateCacheFile);
                return new(false, null, null);
            }

            bool isCacheValid = (DateTime.Now - lastUpdateCheck).Days < 3;
            _logger.LogInformation("{State} check GitHub for update.", isCacheValid ? "Should" : "Should not");
            if (!isCacheValid) return new(false, null, null);

            try
            {
                var jsonBytes = Convert.FromBase64String(content[1]);
                var githubRelease = JsonConvert.DeserializeObject<GithubRelease?>(Encoding.UTF8.GetString(jsonBytes));

                _logger.LogInformation("Found valid update information in disk cache.");
                return new(true, githubRelease, content[2]);
            }
            catch(Exception ex)
            {
                _logger.LogError(ex, "Failed to parse Github JSON from disk.");
                return new(true, null, null);
            }
        }

        private void WriteReleasesToDiskCache(GithubRelease? githubRelease, string? author)
        {
            try
            {
                var jsonBytes = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(githubRelease));
                var base64String = Convert.ToBase64String(jsonBytes);

                var fileContent = new StringBuilder();
                fileContent.Append(DateTime.Now.ToString("yyyyMMdd"));
                fileContent.Append(Environment.NewLine);
                fileContent.Append(base64String);
                fileContent.Append(Environment.NewLine);
                fileContent.Append(author ?? string.Empty);
                fileContent.Append(Environment.NewLine);
                fileContent.Append("EOF");

                File.WriteAllText(Constants.PathUpdateCacheFile, fileContent.ToString());
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to write Github JSON to disk.");
            }
        }

        private static GithubAsset? GetSetupAsset(GithubRelease latestRelease)
        {
            const string EXE_PATTERN = @"^CUETools_Setup_\d+\.\d+\.\d+[a-zA-Z]?\.exe";
            Regex regex = new(EXE_PATTERN);

            return latestRelease.Assets
                .Where(a => regex.IsMatch(a.Name))
                .FirstOrDefault();
        }

        private static GithubAsset? GetHashAsset(GithubRelease latestRelease)
        {
            const string HASH_PATTERN = @"^CUETools_Setup_\d+\.\d+\.\d+[a-zA-Z]?\.exe\.sha256$";
            Regex regex = new(HASH_PATTERN);

            return latestRelease.Assets
                .Where(a => regex.IsMatch(a.Name))
                .FirstOrDefault();
        }

        private async Task<string> GetAuthorAsync(GithubRelease release)
        {
            try
            {
                using var result = await _httpClient.GetAsync(release.Author.Url);
                result.EnsureSuccessStatusCode();

                string response = await result.Content.ReadAsStringAsync();
                var githubUser = JsonConvert.DeserializeObject<GithubUser>(response)
                    ?? throw new NullReferenceException("Failed to deserialize object...");

                return githubUser.Name;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to retrieve release author.");
                return string.Empty;
            }
        }

        private async Task DownloadFile(string uri
            , long contentSize
            , string filePath
            , EventHandler<GenericProgressEventArgs>? progressEvent)
        {
            using var response = await _httpClient.GetAsync(uri, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            long totalBytes = response.Content.Headers.ContentLength ?? contentSize;
            using var httpStream = await response.Content.ReadAsStreamAsync();
            using var fileStream = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None);

            byte[] buffer = new byte[8192];
            long totalReadBytes = 0;
            int bytesRead;

            while ((bytesRead = await httpStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
            {
                await fileStream.WriteAsync(buffer, 0, bytesRead);
                totalReadBytes += bytesRead;

                if (totalBytes >= 0)
                {
                    var eventArgs = new GenericProgressEventArgs((float)totalReadBytes / totalBytes * 100);
                    progressEvent?.Invoke(this, eventArgs);
                }
            }
        }

        public async Task<bool> DownloadAsync(EventHandler<GenericProgressEventArgs> progressEvent)
        {
#if !NET47
            if (!OperatingSystem.IsWindows())
            {
                _logger.LogWarning("Updater is not implemented for this operating system.");
                return false;
            }
#endif

            if (!UpdateMetadata.UpdateAvailable()) return false;

            if (!Directory.Exists(Constants.PathUpdateFolder))
            {
                Directory.CreateDirectory(Constants.PathUpdateFolder);
            }

            try
            {
                var setupFile = $"{Constants.PathUpdateFolder}Update-{UpdateMetadata!.Version}.exe";
                var hashFile = $"{Constants.PathUpdateFolder}Update-{UpdateMetadata.Version}.sha256";

                await DownloadFile(UpdateMetadata!.Uri
                    , contentSize: UpdateMetadata.Size
                    , filePath: setupFile
                    , progressEvent);

                await DownloadFile(UpdateMetadata.HashUri
                    , contentSize: UpdateMetadata.HashSize
                    , filePath: hashFile
                    , progressEvent: null);

                return VerifyFile(setupFile, hashFile);
            }
            catch(Exception ex)
            {
                _logger.LogError(ex, "Failed to download update.");
                return false;
            }
        }

        public static string GetSHA256Hash(string filePath)
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

        private string ParseSHA256FromHashFile(string hashFile)
        {
            var fileContent = File.ReadAllLines(hashFile);
            if (fileContent.Length == 0) return string.Empty;

            return fileContent[0].Split(' ')[0];
        }

        private bool VerifyFile(string setupFile, string hashFile)
        {
            try
            {
                var actualHash = GetSHA256Hash(setupFile);
                var validationHash = ParseSHA256FromHashFile(hashFile);

                return string.Compare(actualHash, validationHash, true) == 0;
            }
            catch(Exception ex)
            {
                _logger.LogError(ex, "Failed to verify hash.");
                return false;
            }
        }

        public void Install()
        {
#if !NET47
            if (!OperatingSystem.IsWindows())
            {
                _logger.LogWarning("Updater is not implemented for this operating system.");
                return;
            }
#endif

            var setupFile = $"{Constants.PathUpdateFolder}Update-{UpdateMetadata!.Version}.exe";
            Process.Start(new ProcessStartInfo
            {
                FileName = Path.GetFullPath(setupFile),
                UseShellExecute = true,
                RedirectStandardOutput = false,
                RedirectStandardError = false,
                CreateNoWindow = false
            });
        }
    }
}
