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
    public class UploadFile
    {
        Stream _data;
        string _fieldName;
        string _fileName;
        string _contentType;

        public UploadFile(Stream data, string fieldName, string fileName, string contentType)
        {
            _data = data;
            _fieldName = fieldName;
            _fileName = fileName;
            _contentType = contentType;
        }

        public UploadFile(string fileName, string fieldName, string contentType)
            : this(File.OpenRead(fileName), fieldName, Path.GetFileName(fileName), contentType)
        { }

        public UploadFile(string fileName)
            : this(fileName, null, "application/octet-stream")
        { }

        public Stream Data
        {
            get { return _data; }
            set { _data = value; }
        }

        public string FieldName
        {
            get { return _fieldName; }
            set { _fieldName = value; }
        }

        public string FileName
        {
            get { return _fileName; }
            set { _fileName = value; }
        }

        public string ContentType
        {
            get { return _contentType; }
            set { _contentType = value; }
        }
    }
}
