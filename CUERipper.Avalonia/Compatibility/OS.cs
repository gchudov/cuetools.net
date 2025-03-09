using System;

namespace CUERipper.Avalonia.Compatibility
{
    internal static class OS
    {
#if NET47
        public static bool IsWindows() => true;
        public static bool IsLinux() => false;
#else
        public static bool IsWindows() => OperatingSystem.IsWindows();
        public static bool IsLinux() => OperatingSystem.IsLinux();
#endif
    }
}
