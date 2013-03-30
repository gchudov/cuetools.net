#region license

/*
WindowsMediaLib - Provide access to Windows Media interfaces via .NET
Copyright (C) 2008
http://sourceforge.net/projects/windowsmedianet/

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#endregion

using System;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Permissions;

[assembly: AssemblyTitle("Windows Media .NET library")]
[assembly: AssemblyDescription(".NET Interfaces for calling Windows Media.  See http://windowsmedianet.sourceforge.net/")]
[assembly: AssemblyConfiguration("")]
[assembly: AssemblyCompany("")]
[assembly: Guid("FA40A71D-13EA-4890-B916-FD76A57D4BAE")]
[assembly: AssemblyVersion("1.1.0.*")]
#if DEBUG
[assembly: AssemblyProduct("Debug Version")]
#else
[assembly : AssemblyProduct("Release Version")]
#endif
[assembly: AssemblyCopyright("Lesser General Public License Version 2.1")]
[assembly: AssemblyTrademark("")]
[assembly: AssemblyCulture("")]
[assembly: AssemblyDelaySign(false)]
[assembly: AssemblyKeyFile("")]
[assembly: AssemblyKeyName("")]

[assembly: ComVisible(false)]
[assembly: CLSCompliant(true)]
[assembly: SecurityPermission(SecurityAction.RequestMinimum, UnmanagedCode = true)]
