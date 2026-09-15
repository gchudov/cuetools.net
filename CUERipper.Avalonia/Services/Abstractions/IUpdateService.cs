using CUERipper.Avalonia.Events;
using CUERipper.Avalonia.Models;
using System;
using System.Threading;
using System.Threading.Tasks;

namespace CUERipper.Avalonia.Services.Abstractions
{
    public interface IUpdateService
    {
        public UpdateMetadata? UpdateMetadata { get; }

        public Task<bool> FetchAsync();

        /// <summary>
        /// progressEvent is raised on the caller's synchronization context.
        /// </summary>
        public Task<bool> DownloadAsync(EventHandler<GenericProgressEventArgs> progressEvent
            , CancellationToken ct);
        void Install();
    }
}
