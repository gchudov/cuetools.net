using System;

namespace CUERipper.Avalonia.Compatibility
{
#if NET47
    internal static class StringExtensions
    {
        public static string[] Split(this string input, string separator)
            => input.Split([separator], StringSplitOptions.None);
    }
#endif
}
