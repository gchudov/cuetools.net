using System;
using System.IO;
using ICSharpCode.SharpZipLib.Zip;

namespace CUETools.Processor
{
	#region Event Delegate Definitions

	/// <summary>
	/// Represents the method that will handle extraction progress events
	/// </summary>
	public delegate void ZipExtractionProgressHandler(object sender, ZipExtractionProgressEventArgs e);
	/// <summary>
	/// Represents the method that will handle password required events
	/// </summary>
	public delegate void ZipPasswordRequiredHandler(object sender, ZipPasswordRequiredEventArgs e);

	#endregion

	public class SeekableZipStream : Stream
	{
		ZipFile zipFile;
		ZipEntry zipEntry;
		Stream zipStream;
		long position;
		byte[] temp;

		public SeekableZipStream(string path, string fileName)
		{
			zipFile = new ZipFile(path);
			zipEntry = zipFile.GetEntry(fileName);
			if (zipEntry == null)
				throw new Exception("Archive entry not found.");
			zipStream = zipFile.GetInputStream(zipEntry);
			temp = new byte[65536];
			position = 0;
		}

		public override bool CanRead
		{
			get { return true; }
		}

		public override bool CanSeek
		{
			get { return true; }
		}

		public override bool CanWrite
		{
			get { return false; }
		}

		public override long Length
		{
			get
			{
				return zipEntry.Size;
			}
		}

		public override long Position
		{
			get { return position; }
			set { Seek(value, SeekOrigin.Begin); }
		}

		public override void Close()
		{
			zipStream.Close();
			zipEntry = null;
			zipFile.Close();
		}

		public override void Flush()
		{
			throw new NotSupportedException();
		}

		public override void SetLength(long value)
		{
			throw new NotSupportedException();
		}

		public override int Read(byte[] buffer, int offset, int count)
		{
			if (position == 0 && zipEntry.IsCrypted && ((ZipInputStream)zipStream).Password == null && PasswordRequired != null)
			{
				ZipPasswordRequiredEventArgs e = new ZipPasswordRequiredEventArgs();
				PasswordRequired(this, e);
				if (e.ContinueOperation && e.Password.Length > 0)
					((ZipInputStream)zipStream).Password = e.Password;
			}
			// TODO: always save to a local temp circular buffer for optimization of the backwards seek.
			int total = zipStream.Read(buffer, offset, count);
			position += total;
			if (ExtractionProgress != null)
			{
				ZipExtractionProgressEventArgs e = new ZipExtractionProgressEventArgs();
				e.BytesExtracted = position;
				e.FileName = zipEntry.Name;
				e.FileSize = zipEntry.Size;
				e.PercentComplete = 100.0 * position / zipEntry.Size;
				ExtractionProgress(this, e);
			}
			return total;
		}

		public override long Seek(long offset, SeekOrigin origin)
		{
			long seek_to;
			switch (origin)
			{
				case SeekOrigin.Begin:
					seek_to = offset;
					break;
				case SeekOrigin.Current:
					seek_to = Position + offset;
					break;
				case SeekOrigin.End:
					seek_to = Length + offset;
					break;
				default:
					throw new NotSupportedException();
			}
			if (seek_to < 0 || seek_to > Length)
				throw new IOException("Invalid seek");
			if (seek_to < position)
			{
				zipStream.Close();
				zipStream = zipFile.GetInputStream(zipEntry);
				position = 0;
			}
			while (seek_to > position)
				if (Read(temp, 0, (int)Math.Min(seek_to - position, (long)temp.Length)) <= 0)
					throw new IOException("Invalid seek");
			return position;
		}

		public override void Write(byte[] array, int offset, int count)
		{
			throw new NotSupportedException();
		}

		public event ZipPasswordRequiredHandler PasswordRequired;
		public event ZipExtractionProgressHandler ExtractionProgress;
	}

	#region Event Argument Classes

	public class ZipPasswordRequiredEventArgs
	{
		public string Password = string.Empty;
		public bool ContinueOperation = true;
	}

	public class ZipExtractionProgressEventArgs
	{
		public string FileName;
		public long FileSize;
		public long BytesExtracted;
		public double PercentComplete;
		public bool ContinueOperation = true;
	}

	#endregion
}
