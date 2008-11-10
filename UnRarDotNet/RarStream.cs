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
			_pos = 0;
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
				lock (this)
				{
					while (_size == null && !_stop)
						Monitor.Wait(this);
				}
				if (_size == null)
					throw new NotSupportedException();
				return _size.Value;
			}
		}
		public override long Position
		{
			get { return _pos; }
			set { Seek(value, SeekOrigin.Begin); }
		}
		public override void Close()
		{
			lock (this)
			{
				_stop = true;
				Monitor.Pulse(this);
			}
			if (_workThread != null)
			{
				_workThread.Join();
				_workThread = null;
			}
			if (_unrar != null)
			{
				_unrar.Close();
				_unrar = null;
			}
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
					if (_seek_to != null)
					{
						if (_seek_to.Value < _pos)
							throw new NotSupportedException();
						if (_length <= _seek_to.Value - _pos)
						{
							_pos += _length;
							_buffer = null;
							Monitor.Pulse(this);
							continue;
						}
						_offset += (int)(_seek_to.Value - _pos);
						_length -= (int)(_seek_to.Value - _pos);
						_pos = _seek_to.Value;
						_seek_to = null;
					}
					if (_length > count)
					{
						Array.Copy(_buffer, _offset, array, offset, count);
						total += count;
						_pos += count;
						_offset += count;
						_length -= count;
						return total;
					}
					Array.Copy(_buffer, _offset, array, offset, _length);
					total += _length;
					_pos += _length;
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
			lock (this)
			{
				while (_size == null && !_stop)
					Monitor.Wait(this);
				if (_size == null)
					throw new NotSupportedException();
				switch (origin)
				{
					case SeekOrigin.Begin:
						_seek_to = offset;
						break;
					case SeekOrigin.Current:
						_seek_to = _pos + offset;
						break;
					case SeekOrigin.End:
						_seek_to = _size.Value + offset;
						break;
				}
				if (_seek_to.Value == _pos)
				{
					_seek_to = null;
					return _pos;
				}
				return _seek_to.Value;
			}
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
		long? _size;
		long? _seek_to;
		long _pos;

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
						lock (this)
						{
							_size = _unrar.CurrentFile.UnpackedSize;
							Monitor.Pulse(this);
						}
						_unrar.Test();
						break;
					}
					else
						_unrar.Skip();
				}
			}
			//catch (StopExtractionException)
			//{
			//}
			lock (this)
			{
				_stop = true;
				Monitor.Pulse(this);
			}
		}
	}
}
