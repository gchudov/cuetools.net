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
using System.Runtime.InteropServices;
using System.Reflection;
using System.IO;

namespace CUERipper.Avalonia.Utilities;

/// <summary>
/// Helper class that'll allow loading libraries from the plugins folder on Linux.
/// </summary>
public static class LibraryResolver
{
    public static void Init()
    {
#if NET8_0_OR_GREATER
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux)) return;

        AppDomain.CurrentDomain.AssemblyLoad += (sender, args) =>
        {
            try
            {
                if (args.LoadedAssembly.Location?.StartsWith(Path.Combine(AppContext.BaseDirectory, "plugins"))
                    ?? false)
                {
                    NativeLibrary.SetDllImportResolver(args.LoadedAssembly, Resolve);
                }
            }
            catch { }
        };
#endif
    }

#if NET8_0_OR_GREATER
    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName.EndsWith(".so"))
        {
            libraryName = libraryName[..^3];
        }

        if (string.IsNullOrWhiteSpace(libraryName)) return IntPtr.Zero;
        if (string.Compare(libraryName, "libc", StringComparison.OrdinalIgnoreCase) == 0) return IntPtr.Zero;

        var libraryPath = Path.Combine(AppContext.BaseDirectory, "plugins", "x64", libraryName);
        libraryPath += ".so";

        if (File.Exists(libraryPath))
        {
            return NativeLibrary.Load(libraryPath);
        }

        return IntPtr.Zero;
    }
#endif    
}
