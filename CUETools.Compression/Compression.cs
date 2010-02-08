using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace CUETools.Compression
{
	public interface ICompressionProvider
	{
		Stream Decompress(string file);
		void Close();
		IEnumerable<string> Contents { get; }

		event EventHandler<CompressionPasswordRequiredEventArgs> PasswordRequired;
		event EventHandler<CompressionExtractionProgressEventArgs> ExtractionProgress;
	}

	#region Event Argument Classes
	public class CompressionPasswordRequiredEventArgs : EventArgs
	{
		public string Password = string.Empty;
		public bool ContinueOperation = true;
	}

	public class CompressionExtractionProgressEventArgs : EventArgs
	{
		public string FileName;
		public long FileSize;
		public long BytesExtracted;
		public double PercentComplete;
		public bool ContinueOperation = true;
	}
	#endregion

	[AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
	public sealed class CompressionProviderClass : Attribute
	{
		private string _extension;

		public CompressionProviderClass(string extension)
		{
			_extension = extension;
		}

		public string Extension
		{
			get { return _extension; }
		}
	}
}
