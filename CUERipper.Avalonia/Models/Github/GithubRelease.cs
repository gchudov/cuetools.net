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
using System.Collections.Generic;
using Newtonsoft.Json;

namespace CUERipper.Avalonia.Models.Github
{
    public class GithubRelease
    {
        [JsonProperty("id")] public long Id { get; set; }
        [JsonProperty("tag_name")] public required string TagName { get; set; }
        [JsonProperty("target_commitish")] public required string TargetCommitish { get; set; }
        [JsonProperty("name")] public required string Name { get; set; }
        [JsonProperty("draft")] public bool Draft { get; set; }
        [JsonProperty("prerelease")] public bool PreRelease { get; set; }
        [JsonProperty("created_at")] public DateTime CreatedAt { get; set; }
        [JsonProperty("published_at")] public DateTime PublishedAt { get; set; }
        [JsonProperty("body")] public required string Body { get; set; }
        [JsonProperty("author")] public required GithubReleaseUser Author { get; set; }
        [JsonProperty("assets")] public required List<GithubAsset> Assets { get; set; }
    }
}
