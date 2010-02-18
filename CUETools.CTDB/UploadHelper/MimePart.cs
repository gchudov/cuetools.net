// http://aspnetupload.com
// Copyright © 2009 Krystalware, Inc.
//
// This work is licensed under a Creative Commons Attribution-Share Alike 3.0 United States License
// http://creativecommons.org/licenses/by-sa/3.0/us/

using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Collections.Specialized;

namespace Krystalware.UploadHelper
{
    public abstract class MimePart
    {
        NameValueCollection _headers = new NameValueCollection();
        byte[] _header;

        public NameValueCollection Headers
        {
            get { return _headers; }
        }

        public byte[] Header
        {
            get { return _header; }
        }

        public long GenerateHeaderFooterData(string boundary)
        {
            StringBuilder sb = new StringBuilder();

            sb.Append("--");
            sb.Append(boundary);
            sb.AppendLine();
            foreach (string key in _headers.AllKeys)
            {
                sb.Append(key);
                sb.Append(": ");
                sb.AppendLine(_headers[key]);
            }
            sb.AppendLine();

            _header = Encoding.UTF8.GetBytes(sb.ToString());

            return _header.Length + Data.Length + 2;
        }

        public abstract Stream Data { get; }
    }
}