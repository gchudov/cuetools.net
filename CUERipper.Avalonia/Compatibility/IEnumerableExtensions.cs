using System.Collections.Generic;
using System.Linq;

namespace CUERipper.Avalonia.Compatibility
{
#if NET47
    internal static class IEnumerableExtensions
    {
        public static IEnumerable<T> Prepend<T>(this IEnumerable<T> src, T item)
            => new[] { item }.Concat(src);
    }
#endif
}