using System.Threading;

namespace CUERipper.Avalonia.Compatibility
{
#if NET47
    internal static class CancellationTokenSourceExtensions
    {
        public static bool TryReset(this CancellationTokenSource _) => false;
    }
#endif
}
