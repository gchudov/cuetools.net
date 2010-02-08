using System;
using System.IO;
using System.Collections.Generic;
using CUETools.Compression;
using ICSharpCode.SharpZipLib.Zip;

namespace CUETools.Compression.Zip
{
	[CompressionProviderClass("zip")]
	public class ZipCompressionProvider : ICompressionProvider
	{
		private ZipFile _zipFile;

		public ZipCompressionProvider(string path)
		{
			_zipFile = new ZipFile(path);
		}

		public void Close()
		{
			if (_zipFile != null) _zipFile.Close();
			_zipFile = null;
		}

		~ZipCompressionProvider()
		{
			Close();
		}

		public Stream Decompress(string file)
		{
			ZipEntry zipEntry = _zipFile.GetEntry(file);
			if (zipEntry == null)
				throw new Exception("Archive entry not found.");
			//if (zipEntry.IsCrypted && PasswordRequired != null)
			//{
			//    CompressionPasswordRequiredEventArgs e = new CompressionPasswordRequiredEventArgs();
			//    PasswordRequired(this, e);
			//    if (e.ContinueOperation && e.Password.Length > 0)
			//        _zipFile.Password = e.Password;
			//}
			SeekableZipStream stream = new SeekableZipStream(_zipFile, zipEntry);
			stream.PasswordRequired += PasswordRequired;
			stream.ExtractionProgress += ExtractionProgress;
			return stream;
		}

		public IEnumerable<string> Contents
		{
			get
			{
				foreach (ZipEntry e in _zipFile)
					if (e.IsFile)
						yield return (e.Name);
			}
		}

		public event EventHandler<CompressionPasswordRequiredEventArgs> PasswordRequired;
		public event EventHandler<CompressionExtractionProgressEventArgs> ExtractionProgress;
	}
}
