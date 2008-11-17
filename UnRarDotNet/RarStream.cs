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
		public RarStream(string path, string fileName)
		{
			_close = false;
			_eof = false;
			_rewind = false;
			_unrar = new Unrar();
			_buffer = null;
			_offset = 0;
			_length = 0;
			_pos = 0;
			_path = path;
			_fileName = fileName;
			_workThread = null;
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
				Go();
				lock (this)
				{
					while (_size == null && !_close)
						Monitor.Wait(this);
				}
				if (_close)
					throw new IOException("Decompression failed", _ex);
				return _size.Value;
			}
		}
		public override long Position
		{
			get { return _seek_to == null ? _pos : _seek_to.Value; }
			set { Seek(value, SeekOrigin.Begin); }
		}
		public override void Close()
		{
			lock (this)
			{
				_close = true;
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
			Go();
			while (count > 0)
			{
				lock (this)
				{
					while (_buffer == null && !_eof && !_close)
						Monitor.Wait(this);
					if (_close)
						throw new IOException("Decompression failed", _ex);
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
			switch (origin)
			{
				case SeekOrigin.Begin:
					_seek_to = offset;
					break;
				case SeekOrigin.Current:
					_seek_to = Position + offset;
					break;
				case SeekOrigin.End:
					_seek_to = Length + offset;
					break;
			}
			if (_seek_to.Value > Length)
			{
				_seek_to = null;
				throw new IOException("Invalid seek");
			}
			if (_seek_to.Value == _pos)
			{
				_seek_to = null;
				return _pos;
			}
			if (_seek_to.Value < _pos)
			{
				lock (this)
				{
					_pos = 0;
					_rewind = true;
					_buffer = null;
					Monitor.Pulse(this);
				}
			}
			return _seek_to.Value;
		}
		public override void Write(byte[] array, int offset, int count)
		{
			throw new NotSupportedException();
		}
		public event PasswordRequiredHandler PasswordRequired;
		public event ExtractionProgressHandler ExtractionProgress;

		private Unrar _unrar;
		private string _fileName;
		private Thread _workThread;
		private bool _close, _rewind, _eof;
		private byte[] _buffer;
		private Exception _ex;
		int _offset, _length;
		long? _size;
		long? _seek_to;
		long _pos;
		string _path;

		private void Go()
		{
			if (_workThread != null) return;
			_workThread = new Thread(Decompress);
			_workThread.Priority = ThreadPriority.BelowNormal;
			_workThread.IsBackground = true;
			_workThread.Start(null);
		}

		private void unrar_DataAvailable(object sender, DataAvailableEventArgs e)
		{
			lock (this)
			{
				while (_buffer != null && !_close)
					Monitor.Wait(this);
				if (_close)
				{
					e.ContinueOperation = false;
					Monitor.Pulse(this);
					return;
				}
				if (_rewind)
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
			_unrar.DataAvailable += new DataAvailableHandler(unrar_DataAvailable);
			_unrar.PasswordRequired += PasswordRequired;
			_unrar.ExtractionProgress += ExtractionProgress;
			try
			{
				do
				{
					bool foundFile = false;
					_unrar.Open(_path, Unrar.OpenMode.Extract);
					while (_unrar.ReadHeader())
					{
						if (_unrar.CurrentFile.FileName == _fileName)
						{
							lock (this)
							{
								if (_size == null)
								{
									_size = _unrar.CurrentFile.UnpackedSize;
									Monitor.Pulse(this);
								}
							}
							_unrar.Test();
							foundFile = true;
							break;
						}
						else
							_unrar.Skip();
					}
					_unrar.Close();
					lock (this)
					{
						if (!foundFile)
						{
							_ex = new FileNotFoundException();
							break;
						}
						else
						{
							_eof = true;
							Monitor.Pulse(this);
							while (!_rewind && !_close)
								Monitor.Wait(this);
							if (_close)
								break;
							_rewind = false;
							_eof = false;
						}
					}
				} while (true);
			}
			catch (Exception ex)
			{
				_ex = ex;
			}
			lock (this)
			{
				_close = true;
				Monitor.Pulse(this);
			}
		}
	}
}
