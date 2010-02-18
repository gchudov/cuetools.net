// http://aspnetupload.com
// Copyright © 2009 Krystalware, Inc.
//
// This work is licensed under a Creative Commons Attribution-Share Alike 3.0 United States License
// http://creativecommons.org/licenses/by-sa/3.0/us/

using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace Krystalware.UploadHelper
{
    public class StreamMimePart : MimePart
    {
        Stream _data;

        public void SetStream(Stream stream)
        {
            _data = stream;
        }

        public override Stream Data
        {
            get
            {
                return _data;
            }
        }
    }
}
