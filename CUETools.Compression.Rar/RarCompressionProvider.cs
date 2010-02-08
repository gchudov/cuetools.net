using System;
using System.IO;
using System.Collections.Generic;
using CUETools.Compression;

namespace CUETools.Compression.Rar
{
	[CompressionProviderClass("rar")]
	public class RarCompressionProvider: ICompressionProvider
	{
		private string _archivePath;

		public RarCompressionProvider(string path)
		{
			_archivePath = path;
		}

		public void Close()
		{
		}

		~RarCompressionProvider()
		{
			Close();
		}

		public Stream Decompress(string file)
		{
			RarStream stream = new RarStream(_archivePath, file);
			stream.PasswordRequired += PasswordRequired;
			stream.ExtractionProgress += ExtractionProgress;
			return stream;
		}

		public IEnumerable<string> Contents
		{
			get
			{
				using (Unrar _unrar = new Unrar())
				{
					_unrar.PasswordRequired += PasswordRequired;
					_unrar.Open(_archivePath, Unrar.OpenMode.List);
					while (_unrar.ReadHeader())
					{
						if (!_unrar.CurrentFile.IsDirectory)
							yield return _unrar.CurrentFile.FileName;
						_unrar.Skip();
					}
					_unrar.Close();
				}
			}
		}

		public event EventHandler<CompressionPasswordRequiredEventArgs> PasswordRequired;
		public event EventHandler<CompressionExtractionProgressEventArgs> ExtractionProgress;
	}
}
