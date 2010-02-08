using System;
using System.IO;
using System.Collections.Generic;
using CUETools.Compression;
using ICSharpCode.SharpZipLib.Zip;

namespace CUETools.Compression.Zip
{
	public class SeekableZipStream : Stream
	{
		ZipFile zipFile;
		ZipEntry zipEntry;
		Stream zipStream;
		long position;
		byte[] temp;

		public SeekableZipStream(ZipFile file, ZipEntry entry)
		{
			zipFile = file;
			zipEntry = entry;
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
			zipEntry = null;
			zipStream.Close();
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
				CompressionPasswordRequiredEventArgs e = new CompressionPasswordRequiredEventArgs();
				PasswordRequired(this, e);
				if (e.ContinueOperation && e.Password.Length > 0)
					((ZipInputStream)zipStream).Password = e.Password;
			}
			// TODO: always save to a local temp circular buffer for optimization of the backwards seek.
			int total = zipStream.Read(buffer, offset, count);
			position += total;
			if (ExtractionProgress != null)
			{
				CompressionExtractionProgressEventArgs e = new CompressionExtractionProgressEventArgs();
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

		public event EventHandler<CompressionPasswordRequiredEventArgs> PasswordRequired;
		public event EventHandler<CompressionExtractionProgressEventArgs> ExtractionProgress;
	}
}
