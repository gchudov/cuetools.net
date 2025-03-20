namespace CUERipper.Avalonia.Models.Github;
public record GithubReleaseContainer (bool IsFromCache, GithubRelease? Content, string? Author);
