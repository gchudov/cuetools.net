using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Models;
using CUERipper.Avalonia.Models.Github;
using CUERipper.Avalonia.Services.Abstractions;
using CUETools.Processor;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
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
        private const string UpdateCheckFilePath = "CT_LAST_UPDATE_CHECK";

        private readonly HttpClient _httpClient;
        private readonly ILogger _logger;

        public UpdateMetadata? UpdateMetadata { get; private set; }

        public UpdateService(HttpClient httpClient
            , ILogger<UpdateService> logger)
        {
            _httpClient = httpClient;
            _logger = logger;
        }

        public async Task<bool> FetchAsync()
        {
#if !NET47
            if (!OperatingSystem.IsWindows())
            {
                _logger.LogWarning("Updater is not implemented for this operating system.");
                return false;
            }
#endif

            if (UpdateMetadata != null) return true;

            var githubReleases = await GetReleaseAsync();
            var latestRelease = githubReleases
                    .Where(r => !r.Draft && !r.PreRelease && r.TargetCommitish == "master")
                    .OrderByDescending(r => r.PublishedAt)
                    .FirstOrDefault();

            if (latestRelease == null)
            {
                _logger.LogWarning("No releases found.");
                return false;
            }

            string versionPattern = @"^v\d+\.\d+\.\d+$";
            Regex regex = new(versionPattern);
            if (!regex.IsMatch(latestRelease.TagName))
            {
                _logger.LogError("Release tag '{TAG}' doesn't match expected format.", latestRelease.TagName);
                return false;
            }

            var zipAsset = GetZipAsset(latestRelease);
            var hashAsset = GetHashAsset(latestRelease);
            if (zipAsset == null || hashAsset == null)
            {
                _logger.LogWarning("Github assets are incomplete.");
                return false;
            }

            UpdateMetadata = new UpdateMetadata(
                Version: latestRelease.TagName.Substring(1)
                , CurrentVersion: CUESheet.CUEToolsVersion
                , Author: await GetAuthorAsync(latestRelease)
                , Description: latestRelease.Body
                , Uri: zipAsset.BrowserDownloadUrl
                , Size: zipAsset.Size
                , HashUri: hashAsset.BrowserDownloadUrl
                , HashSize: hashAsset.Size
                , Date: latestRelease.PublishedAt
            );

            return true;
        }

        private async Task<IEnumerable<GithubRelease>> GetReleaseAsync()
        {
            IEnumerable<GithubRelease> githubReleases = GetReleasesFromDiskCache();
            if (githubReleases.Any()) return githubReleases;

            try
            {
                using var result = await _httpClient.GetAsync(Constants.GithubApiUri);
                result.EnsureSuccessStatusCode();

                string response = await result.Content.ReadAsStringAsync();
                WriteReleasesToDiskCache(response);

                githubReleases = JsonConvert.DeserializeObject<GithubRelease[]>(response)
                    ?? throw new NullReferenceException("Failed to deserialize object...");
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to retrieve latest release.");
            }

            return githubReleases;
        }


        /// <summary>
        /// Mechanism that prevents spamming the GitHub API by limiting automated requests to once every 3 days.
        /// </summary>
        /// <returns></returns>
        private IEnumerable<GithubRelease> GetReleasesFromDiskCache()
        {
            if (!File.Exists(UpdateCheckFilePath)) return [];

            string[] content = File.ReadAllLines(UpdateCheckFilePath);
            if (content.Length != 2)
            {
                _logger.LogError("Content of {File} is incorrect.", UpdateCheckFilePath);
                return [];
            }

            if (!DateTime.TryParseExact(content[0], "yyyyMMdd", null, System.Globalization.DateTimeStyles.None
                , out DateTime lastUpdateCheck))
            {
                _logger.LogError("Content of {File} is incorrect, can't parse to datetime.", UpdateCheckFilePath);
                return [];
            }

            bool shouldUpdate = (DateTime.Now - lastUpdateCheck).Days >= 3;
            _logger.LogInformation("{State} check for update.", shouldUpdate ? "Should" : "Should not");
            if (shouldUpdate) return [];

            try
            {
                var jsonBytes = Convert.FromBase64String(content[1]);
                var json = Encoding.UTF8.GetString(jsonBytes);

                var result = JsonConvert.DeserializeObject<GithubRelease[]>(json)
                    ?? throw new NullReferenceException("Failed to deserialize object...");

                _logger.LogInformation("Found valid update information in disk cache.");
                return result;
            }
            catch(Exception ex)
            {
                _logger.LogError(ex, "Failed to parse Github JSON from disk.");
                return [];
            }
        }

        private void WriteReleasesToDiskCache(string json)
        {
            try
            {
                var jsonBytes = Encoding.UTF8.GetBytes(json);
                var base64String = Convert.ToBase64String(jsonBytes);

                var fileContent = new StringBuilder();
                fileContent.Append(DateTime.Now.ToString("yyyyMMdd"));
                fileContent.Append(Environment.NewLine);
                fileContent.Append(base64String);

                File.WriteAllText(UpdateCheckFilePath, fileContent.ToString());
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to write Github JSON to disk.");
            }
        }

        private static GithubAsset? GetZipAsset(GithubRelease latestRelease)
        {
            const string ZIP_PATTERN = @"^CUETools_\d+\.\d+\.\d+\.zip$";
            Regex regex = new(ZIP_PATTERN);

            return latestRelease.Assets
                .Where(a => regex.IsMatch(a.Name))
                .FirstOrDefault();
        }

        private static GithubAsset? GetHashAsset(GithubRelease latestRelease)
        {
            const string ZIP_PATTERN = @"^CUETools_\d+\.\d+\.\d+\.zip.sha256$";
            Regex regex = new(ZIP_PATTERN);

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
            if (!UpdateMetadata.UpdateAvailable()) return false;

            if (!Directory.Exists(Constants.PathUpdate))
            {
                Directory.CreateDirectory(Constants.PathUpdate);
            }

            try
            {
                var zipFile = $"{Constants.PathUpdate}Update-{UpdateMetadata!.Version}.zip";
                var hashFile = $"{Constants.PathUpdate}Update-{UpdateMetadata.Version}.sha256";

                await DownloadFile(UpdateMetadata!.Uri
                    , contentSize: UpdateMetadata.Size
                    , filePath: zipFile
                    , progressEvent);

                await DownloadFile(UpdateMetadata.HashUri
                    , contentSize: UpdateMetadata.HashSize
                    , filePath: hashFile
                    , progressEvent: null);

                return VerifyFile(zipFile, hashFile);
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

        private bool VerifyFile(string zipFile, string hashFile)
        {
            try
            {
                var actualHash = GetSHA256Hash(zipFile);
                var validationHash = ParseSHA256FromHashFile(hashFile);

                return string.Compare(actualHash, validationHash, true) == 0;
            }
            catch(Exception ex)
            {
                _logger.LogError(ex, "Failed to verify hash.");
                return false;
            }
        }
    }
}
