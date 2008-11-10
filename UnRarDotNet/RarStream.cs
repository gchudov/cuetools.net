using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Threading;


/*  Author:  Gregory S. Chudov
 *  
 */

namespace UnRarDotNet
{
	public class RarStream : Stream
	{
		public RarStream(string archive, string fileName)
		{
			_stop = false;
			_unrar = new Unrar();
			_buffer = null;
			_offset = 0;
			_length = 0;
			_unrar.PasswordRequired += new PasswordRequiredHandler(unrar_PasswordRequired);
			_unrar.DataAvailable += new DataAvailableHandler(unrar_DataAvailable);
			_unrar.Open(archive, Unrar.OpenMode.Extract);
			_fileName = fileName;
			_workThread = new Thread(Decompress);
			_workThread.Priority = ThreadPriority.BelowNormal;
			_workThread.IsBackground = true;
			_workThread.Start(null);
		}
		public override bool CanRead
		{
			get { return true; }
		}
		public override bool CanSeek
		{
			get { return false; }
		}
		public override bool CanWrite
		{
			get { return false; }
		}
		public override long Length
		{
			get { throw new NotSupportedException(); }
		}
		public override long Position
		{
			get { throw new NotSupportedException(); }
			set { Seek(value, SeekOrigin.Begin); }
		}
		public override void Close()
		{
			lock (this)
			{
				_stop = true;
				Monitor.Pulse(this);
			}
			_workThread.Join();
			_workThread = null;
			_unrar.Close();
			base.Close();
		}
		public override void Flush()
		{
			throw new NotSupportedException();
		}
		public override void SetLength(long value)
		{
			throw new NotSupportedException();
		}
		public override int Read(byte[] array, int offset, int count)
		{
			int total = 0;
			while (count > 0)
			{
				lock (this)
				{
					while (_buffer == null && !_stop)
						Monitor.Wait(this);
					if (_buffer == null)
						return total;
					if (_length > count)
					{
						Array.Copy(_buffer, _offset, array, offset, count);
						total += count;
						_offset += count;
						_length -= count;
						return total;
					}
					Array.Copy(_buffer, _offset, array, offset, _length);
					total += _length;
					offset += _length;
					count -= _length;
					_buffer = null;
					Monitor.Pulse(this);
				}
			}
			return total;
		}
		public override long Seek(long offset, SeekOrigin origin)
		{
			throw new NotSupportedException();
		}
		public override void Write(byte[] array, int offset, int count)
		{
			throw new NotSupportedException();
		}

		private Unrar _unrar;
		private string _fileName;
		private Thread _workThread;
		private bool _stop;
		private byte[] _buffer;
		int _offset, _length;

		private void unrar_PasswordRequired(object sender, PasswordRequiredEventArgs e)
		{
			e.Password = "PARS";
			e.ContinueOperation = true;
		}

		private void unrar_DataAvailable(object sender, DataAvailableEventArgs e)
		{
			lock (this)
			{
				while (_buffer != null && !_stop)
					Monitor.Wait(this);
				if (_stop)
				{
					e.ContinueOperation = false;
					Monitor.Pulse(this);
					return;
				}
				_buffer = e.Data;
				_length = _buffer.Length;
				_offset = 0;
				e.ContinueOperation = true;
				Monitor.Pulse(this);
			}
		}

		private void Decompress(object o)
		{
			//try
			{
				while (_unrar.ReadHeader())
				{
					if (_unrar.CurrentFile.FileName == _fileName)
					{
						// unrar.CurrentFile.UnpackedSize;
						_unrar.Test();
						lock (this)
						{
							_stop = true;
							Monitor.Pulse(this);
						}
						break;
					}
					else
						_unrar.Skip();
				}
			}
			//catch (StopExtractionException)
			//{
			//}
		}
	}
}
