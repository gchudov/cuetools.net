#region Copyright (C) 2025 Max Visser
/*
    Copyright (C) 2025 Max Visser

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, see <https://www.gnu.org/licenses/>.
*/
#endregion
using System;
using Newtonsoft.Json;

namespace CUERipper.Avalonia.Models.Github
{
    public class GithubAsset
    {
        [JsonProperty("id")] public long Id { get; set; }
        [JsonProperty("url")] public required string Url { get; set; }
        [JsonProperty("name")] public required string Name { get; set; }
        [JsonProperty("content_type")] public required string ContentType { get; set; }
        [JsonProperty("state")] public required string State { get; set; }
        [JsonProperty("size")] public long Size { get; set; }
        [JsonProperty("download_count")] public int DownloadCount { get; set; }
        [JsonProperty("created_at")] public DateTime CreatedAt { get; set; }
        [JsonProperty("updated_at")] public DateTime UpdatedAt { get; set; }
        [JsonProperty("browser_download_url")] public required string BrowserDownloadUrl { get; set; }
        [JsonProperty("uploader")] public required GithubReleaseUser Uploader { get; set; }
    }
}
