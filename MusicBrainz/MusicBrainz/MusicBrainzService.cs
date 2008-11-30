// MusicBrainzService.cs
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
using System.Net.Cache;

namespace MusicBrainz
{
    public static class MusicBrainzService
    {
        static Uri service_url = new Uri (@"http://musicbrainz.org/ws/1/");
        public static Uri ServiceUrl {
            get { return service_url; }
            set {
                if (value == null) throw new ArgumentNullException ("value");
                service_url = value;
            }
        }
        
        static RequestCachePolicy cache_policy;
        public static RequestCachePolicy CachePolicy {
            get { return cache_policy; }
            set { cache_policy = value; }
        }
        
        public static event EventHandler<XmlRequestEventArgs> XmlRequest;
        
        internal static void OnXmlRequest (string url, bool fromCache)
        {
            EventHandler<XmlRequestEventArgs> handler = XmlRequest;
            if (handler != null) handler (null, new XmlRequestEventArgs (new Uri (url), fromCache));
        }
    }
}
