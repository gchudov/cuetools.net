using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Models;
using System;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.Services.Abstractions
{
    public interface IUpdateService
    {
        public UpdateMetadata? UpdateMetadata { get; }

        public Task<bool> FetchAsync();
        public Task<bool> DownloadAsync(EventHandler<GenericProgressEventArgs> progressEvent);
    }
}
