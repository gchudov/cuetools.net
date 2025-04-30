// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

// Modified for CUERipper

using System.ComponentModel;

namespace System.Runtime.CompilerServices
{
#if NET47
    /// <summary>
    /// Reserved to be used by the compiler for tracking metadata.
    /// This class should not be used by developers in source code.
    /// </summary>
    [EditorBrowsable(EditorBrowsableState.Never)]
#if SYSTEM_PRIVATE_CORELIB
    public
#else
    internal
#endif
    static class IsExternalInit
    {
    }
#endif
}