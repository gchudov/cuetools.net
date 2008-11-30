// MusicBrainzException.cs
//
// Copyright (c) 2008 Scott Peterson <lunchtimemama@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

using System;

namespace MusicBrainz
{
    public sealed class MusicBrainzInvalidParameterException : Exception
    {
        public MusicBrainzInvalidParameterException ()
            : base ("One of the parameters is invalid. The ID may be invalid, or you may be using an illegal parameter for this resource type.")
        {
        }
    }

    public sealed class MusicBrainzNotFoundException : Exception
    {
        public MusicBrainzNotFoundException ()
            : base ("Specified resource was not found. Perhaps it was merged or deleted.")
        {
        }
    }

    public sealed class MusicBrainzUnauthorizedException : Exception
    {
        public MusicBrainzUnauthorizedException ()
            : base ("The client is not authorized to perform this action. You may not have authenticated, or the username or password may be incorrect.")
        {
        }
    }
}
