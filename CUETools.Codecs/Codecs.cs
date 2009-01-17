using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Threading;

namespace CUETools.Codecs
{
	public interface IAudioSource
	{
		uint Read(int[,] buff, uint sampleCount);
		int[,] Read(int[,] buff);
		ulong Length { get; }
		ulong Position { get; set; }
		NameValueCollection Tags { get; set; }
		bool UpdateTags(bool preserveTime);
		ulong Remaining { get; }
		void Close();
		int BitsPerSample { get; }
		int ChannelCount { get; }
		int SampleRate { get; }
		string Path { get; }
	}

	public interface IAudioDest
	{
		void Write(int[,] buff, uint sampleCount);
		bool SetTags(NameValueCollection tags);
		void Close();
		void Delete();
		int BitsPerSample { get; }
		long FinalSampleCount { set; }
		long BlockSize { set; }
		string Path { get; }
	}

	public class AudioSamples
	{
		public static unsafe void FLACSamplesToBytes_16(int[,] inSamples, uint inSampleOffset,
			byte[] outSamples, uint outByteOffset, uint sampleCount, int channelCount)
		{
			uint loopCount = sampleCount * (uint)channelCount;

			if ((inSamples.GetLength(0) - inSampleOffset < sampleCount) ||
				(outSamples.Length - outByteOffset < loopCount * 2))
			{
				throw new IndexOutOfRangeException();
			}

			fixed (int* pInSamplesFixed = &inSamples[inSampleOffset, 0])
			{
				fixed (byte* pOutSamplesFixed = &outSamples[outByteOffset])
				{
					int* pInSamples = pInSamplesFixed;
					short* pOutSamples = (short*)pOutSamplesFixed;

					for (int i = 0; i < loopCount; i++)
					{
						*(pOutSamples++) = (short)*(pInSamples++);
					}
				}
			}
		}
		
		public static unsafe void FLACSamplesToBytes_24(int[,] inSamples, uint inSampleOffset,
			byte[] outSamples, uint outByteOffset, uint sampleCount, int channelCount, int wastedBits)
		{
			uint loopCount = sampleCount * (uint)channelCount;

			if ((inSamples.GetLength(0) - inSampleOffset < sampleCount) ||
				(outSamples.Length - outByteOffset < loopCount * 3))
			{
				throw new IndexOutOfRangeException();
			}

			fixed (int* pInSamplesFixed = &inSamples[inSampleOffset, 0])
			{
				fixed (byte* pOutSamplesFixed = &outSamples[outByteOffset])
				{
					int* pInSamples = pInSamplesFixed;
					byte* pOutSamples = pOutSamplesFixed;

					for (int i = 0; i < loopCount; i++)
					{
						uint sample_out = (uint)*(pInSamples++) << wastedBits;
						*(pOutSamples++) = (byte)(sample_out & 0xFF);
						sample_out >>= 8;
						*(pOutSamples++) = (byte)(sample_out & 0xFF);
						sample_out >>= 8;
						*(pOutSamples++) = (byte)(sample_out & 0xFF);
					}
				}
			}
		}

		public static unsafe void FLACSamplesToBytes(int[,] inSamples, uint inSampleOffset,
			byte[] outSamples, uint outByteOffset, uint sampleCount, int channelCount, int bitsPerSample)
		{
			if (bitsPerSample == 16)
				AudioSamples.FLACSamplesToBytes_16(inSamples, inSampleOffset, outSamples, outByteOffset, sampleCount, channelCount);
			else if (bitsPerSample > 16 && bitsPerSample <= 24)
				AudioSamples.FLACSamplesToBytes_24(inSamples, inSampleOffset, outSamples, outByteOffset, sampleCount, channelCount, 24 - bitsPerSample);
			else
				throw new Exception("Unsupported bitsPerSample value");
		}

		public static unsafe void BytesToFLACSamples_16(byte[] inSamples, int inByteOffset,
			int[,] outSamples, int outSampleOffset, uint sampleCount, int channelCount)
		{
			uint loopCount = sampleCount * (uint)channelCount;

			if ((inSamples.Length - inByteOffset < loopCount * 2) ||
				(outSamples.GetLength(0) - outSampleOffset < sampleCount))
			{
				throw new IndexOutOfRangeException();
			}

			fixed (byte* pInSamplesFixed = &inSamples[inByteOffset])
			{
				fixed (int* pOutSamplesFixed = &outSamples[outSampleOffset, 0])
				{
					short* pInSamples = (short*)pInSamplesFixed;
					int* pOutSamples = pOutSamplesFixed;

					for (int i = 0; i < loopCount; i++)
					{
						*(pOutSamples++) = (int)*(pInSamples++);
					}
				}
			}
		}

		public static int[,] Read(IAudioSource source, int[,] buff)
		{
			if (source.Remaining == 0) return null;
			uint toRead = Math.Min(65536U, (uint)source.Remaining);
			if (buff == null || (ulong)buff.GetLength(0) > source.Remaining)
				buff = new int[toRead, source.ChannelCount];
			else
				toRead = (uint)buff.GetLength(0);
			uint samplesRead = source.Read(buff, toRead);
			if (samplesRead != toRead) throw new Exception("samples read != requested");
			return buff;
		}
	}

	public class DummyWriter : IAudioDest
	{
		public DummyWriter(string path, int bitsPerSample, int channelCount, int sampleRate)
		{
			_bitsPerSample = bitsPerSample;
		}

		public bool SetTags(NameValueCollection tags)
		{
			return false;
		}

		public void Close()
		{
		}

		public void Delete()
		{
		}

		public long FinalSampleCount
		{
			set	{ }
		}

		public long BlockSize
		{
			set	{ }
		}

		public int BitsPerSample
		{
			get { return _bitsPerSample;  }
		}

		public void Write(int[,] buff, uint sampleCount)
		{
		}

		public string Path { get { return null; } }

		int _bitsPerSample;
	}

	public class SilenceGenerator : IAudioSource
	{
		private ulong _sampleOffset, _sampleCount;

		public SilenceGenerator(uint sampleCount)
		{
			_sampleOffset = 0;
			_sampleCount = sampleCount;
		}

		public ulong Length
		{
			get
			{
				return _sampleCount;
			}
		}

		public ulong Remaining
		{
			get
			{
				return _sampleCount - _sampleOffset;
			}
		}

		public ulong Position
		{
			get
			{
				return _sampleOffset;
			}
			set
			{
				_sampleOffset = value;
			}
		}

		public int BitsPerSample
		{
			get
			{
				return 16;
			}
		}

		public int ChannelCount
		{
			get
			{
				return 2;
			}
		}

		public int SampleRate
		{
			get
			{
				return 44100;
			}
		}

		public NameValueCollection Tags
		{
			get
			{
				return new NameValueCollection();
			}
			set
			{
			}
		}

		public bool UpdateTags(bool preserveTime)
		{
			return false;
		}

		public uint Read(int [,] buff, uint sampleCount)
		{
			uint samplesRemaining = (uint)(_sampleCount - _sampleOffset);
			if (sampleCount > samplesRemaining)
				sampleCount = samplesRemaining;

			for (uint i = 0; i < sampleCount; i++)
				for (int j = 0; j < buff.GetLength(1); j++)
					buff[i,j] = 0;

			_sampleOffset += sampleCount;
			return sampleCount;
		}

		public int[,] Read(int[,] buff)
		{
			if (buff != null && buff.GetLength(0) <= (int)Remaining)
			{
				_sampleOffset += (ulong) buff.GetLength(0);
				Array.Clear(buff, 0, buff.Length);
				return buff;
			}
			ulong samples = Math.Min(Remaining, (ulong)4096);
			_sampleCount += samples;
			return new int[samples, ChannelCount];
		}

		public void Close()
		{
		}

		public string Path { get { return null; } }
	}

	public class WAVReader : IAudioSource
	{
		Stream _IO;
		BinaryReader _br;
		ulong _dataOffset, _dataLen;
		ulong _samplePos, _sampleLen;
		int _bitsPerSample, _channelCount, _sampleRate, _blockAlign;
		bool _largeFile;
		string _path;
		private byte[] _sampleBuffer;

		public WAVReader(string path, Stream IO)
		{
			_path = path;
			_IO = IO != null ? IO : new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 0x10000, FileOptions.SequentialScan);
			_br = new BinaryReader(_IO);

			ParseHeaders();

			_sampleLen = _dataLen / (uint)_blockAlign;
		}

		public void Close()
		{
			if (_br != null)
			{
				_br.Close();
				_br = null;
			}
			_IO = null;
		}

		private void ParseHeaders()
		{
			const long maxFileSize = 0x7FFFFFFEL;
			const uint fccRIFF = 0x46464952;
			const uint fccWAVE = 0x45564157;
			const uint fccFormat = 0x20746D66;
			const uint fccData = 0x61746164;

			uint lenRIFF;
			long fileEnd;
			bool foundFormat, foundData;

			if (_br.ReadUInt32() != fccRIFF)
			{
				throw new Exception("Not a valid RIFF file.");
			}

			lenRIFF = _br.ReadUInt32();
			fileEnd = (long)lenRIFF + 8;

			if (_br.ReadUInt32() != fccWAVE)
			{
				throw new Exception("Not a valid WAVE file.");
			}

			_largeFile = false;
			foundFormat = false;
			foundData = false;
			long pos = 12;
			do
			{
				uint ckID, ckSize, ckSizePadded;
				long ckEnd;

				ckID = _br.ReadUInt32();
				ckSize = _br.ReadUInt32();
				ckSizePadded = (ckSize + 1U) & ~1U;
				pos += 8;
				ckEnd = pos + (long)ckSizePadded;

				if (ckID == fccFormat)
				{
					foundFormat = true;

					if (_br.ReadUInt16() != 1)
					{
						throw new Exception("WAVE must be PCM format.");
					}
					_channelCount = _br.ReadInt16();
					_sampleRate = _br.ReadInt32();
					_br.ReadInt32();
					_blockAlign = _br.ReadInt16();
					_bitsPerSample = _br.ReadInt16();
					pos += 16;
				}
				else if (ckID == fccData)
				{
					foundData = true;

					_dataOffset = (ulong)pos;
					if (!_IO.CanSeek || _IO.Length <= maxFileSize)
					{
						_dataLen = ckSize;
					}
					else
					{
						_largeFile = true;
						_dataLen = ((ulong)_IO.Length) - _dataOffset;
					}
				}

				if ((foundFormat & foundData) || _largeFile)
					break;
				if (_IO.CanSeek)
					_IO.Seek(ckEnd, SeekOrigin.Begin);
				else
					_br.ReadBytes((int)(ckEnd - pos));
				pos = ckEnd;
			} while (true);

			if ((foundFormat & foundData) == false)
				throw new Exception("Format or data chunk not found.");
			if (_channelCount <= 0)
				throw new Exception("Channel count is invalid.");
			if (_sampleRate <= 0)
				throw new Exception("Sample rate is invalid.");
			if (_blockAlign != (_channelCount * ((_bitsPerSample + 7) / 8)))
				throw new Exception("Block align is invalid.");
			if ((_bitsPerSample <= 0) || (_bitsPerSample > 32))
				throw new Exception("Bits per sample is invalid.");
			if (pos != (long)_dataOffset)
				Position = 0;
		}

		public ulong Position
		{
			get
			{
				return _samplePos;
			}
			set
			{
				ulong seekPos;

				if (value > _sampleLen)
				{
					_samplePos = _sampleLen;
				}
				else
				{
					_samplePos = value;
				}

				seekPos = _dataOffset + (_samplePos * (uint)_blockAlign);
				_IO.Seek((long)seekPos, SeekOrigin.Begin);
			}
		}

		public ulong Length
		{
			get
			{
				return _sampleLen;
			}
		}

		public ulong Remaining
		{
			get
			{
				return _sampleLen - _samplePos;
			}
		}

		public int ChannelCount
		{
			get
			{
				return _channelCount;
			}
		}

		public int SampleRate
		{
			get
			{
				return _sampleRate;
			}
		}

		public int BitsPerSample
		{
			get
			{
				return _bitsPerSample;
			}
		}

		public int BlockAlign
		{
			get
			{
				return _blockAlign;
			}
		}

		public NameValueCollection Tags
		{
			get
			{
				return new NameValueCollection();
			}
			set
			{
			}
		}

		public bool UpdateTags(bool preserveTime)
		{
			return false;
		}

		public uint Read(int[,] buff, uint sampleCount)
		{
			if (sampleCount > Remaining)
				sampleCount = (uint)Remaining;

			if (sampleCount == 0)
				return 0;
			int byteCount = (int) sampleCount * _blockAlign;
			if (_sampleBuffer == null || _sampleBuffer.Length < byteCount)
				_sampleBuffer = new byte[byteCount];
			if (_IO.Read(_sampleBuffer, 0, (int)byteCount) != byteCount)
				throw new Exception("Incomplete file read.");
			AudioSamples.BytesToFLACSamples_16(_sampleBuffer, 0, buff, 0,
				sampleCount, _channelCount);
			_samplePos += sampleCount;
			return sampleCount;
		}

		public int[,] Read(int[,] buff)
		{
			return AudioSamples.Read(this, buff);
		}

		public string Path { get { return _path; } }
	}

	public class WAVWriter : IAudioDest
	{
		Stream _IO;
		BinaryWriter _bw;
		int _bitsPerSample, _channelCount, _sampleRate, _blockAlign;
		long _sampleLen;
		string _path;
		private byte[] _sampleBuffer;
		long hdrLen = 0;
		bool _headersWritten = false;
		long _finalSampleCount;
		List<byte[]> _chunks = null;
		List<uint> _chunkFCCs = null;

		public WAVWriter(string path, int bitsPerSample, int channelCount, int sampleRate, Stream IO)
		{
			_path = path;
			_bitsPerSample = bitsPerSample;
			_channelCount = channelCount;
			_sampleRate = sampleRate;
			_blockAlign = _channelCount * ((_bitsPerSample + 7) / 8);

			_IO = IO != null ? IO : new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
			_bw = new BinaryWriter(_IO);
		}

		public bool SetTags(NameValueCollection tags)
		{
			return false;
		}

		public void WriteChunk(uint fcc, byte[] data)
		{
			if (_sampleLen > 0)
				throw new Exception("data already written, no chunks allowed");
			if (_chunks == null)
			{
				_chunks = new List<byte[]>();
				_chunkFCCs = new List<uint>();
			}
			_chunkFCCs.Add(fcc);
			_chunks.Add(data);
			hdrLen += 8 + data.Length + (data.Length & 1);
		}

		private void WriteHeaders()
		{
			const uint fccRIFF = 0x46464952;
			const uint fccWAVE = 0x45564157;
			const uint fccFormat = 0x20746D66;
			const uint fccData = 0x61746164;

			bool wavex = _bitsPerSample != 16 && _bitsPerSample != 24;

			hdrLen += 36 + (wavex ? 24 : 0) + 8;

			uint dataLen = (uint) (_finalSampleCount * _blockAlign);
			uint dataLenPadded = dataLen + (dataLen & 1);

			_bw.Write(fccRIFF);
			_bw.Write((uint)(dataLenPadded + hdrLen - 8));
			_bw.Write(fccWAVE);
			_bw.Write(fccFormat);
			if (wavex)
			{
				_bw.Write((uint)40);
				_bw.Write((ushort)0xfffe); // WAVEX follows
			}
			else
			{
				_bw.Write((uint)16);
				_bw.Write((ushort)1); // PCM
			}
			_bw.Write((ushort)_channelCount);
			_bw.Write((uint)_sampleRate);
			_bw.Write((uint)(_sampleRate * _blockAlign));
			_bw.Write((ushort)_blockAlign);
			_bw.Write((ushort)((_bitsPerSample+7)/8*8));
			if (wavex)
			{
				_bw.Write((ushort)22); // length of WAVEX structure
				_bw.Write((ushort)_bitsPerSample);
				_bw.Write((uint)3); // speaker positions (3 == stereo)
				_bw.Write((ushort)1); // PCM
				_bw.Write((ushort)0);
				_bw.Write((ushort)0);
				_bw.Write((ushort)0x10);
				_bw.Write((byte)0x80);
				_bw.Write((byte)0x00);
				_bw.Write((byte)0x00);
				_bw.Write((byte)0xaa);
				_bw.Write((byte)0x00);
				_bw.Write((byte)0x38);
				_bw.Write((byte)0x9b);
				_bw.Write((byte)0x71);
			}
			if (_chunks != null)
				for (int i = 0; i < _chunks.Count; i++)
				{
					_bw.Write(_chunkFCCs[i]);
					_bw.Write((uint)_chunks[i].Length);
					_bw.Write(_chunks[i]);
					if ((_chunks[i].Length & 1) != 0)
						_bw.Write((byte)0);
				}

			_bw.Write(fccData);
			_bw.Write(dataLen);

			_headersWritten = true;
		}

		public void Close()
		{
			if (_finalSampleCount == 0)
			{
				const long maxFileSize = 0x7FFFFFFEL;
				long dataLen = _sampleLen * _blockAlign;
				if ((dataLen & 1) == 1)
					_bw.Write((byte)0);
				if (dataLen + hdrLen > maxFileSize)
					dataLen = ((maxFileSize - hdrLen) / _blockAlign) * _blockAlign;
				long dataLenPadded = dataLen + (dataLen & 1);

				_bw.Seek(4, SeekOrigin.Begin);
				_bw.Write((uint)(dataLenPadded + hdrLen - 8));

				_bw.Seek((int)hdrLen - 4, SeekOrigin.Begin);
				_bw.Write((uint)dataLen);
			}

			_bw.Close();

			_bw = null;
			_IO = null;

			if (_finalSampleCount != 0 && _sampleLen != _finalSampleCount)
				throw new Exception("Samples written differs from the expected sample count.");
		}

		public void Delete()
		{
			_bw.Close();
			_bw = null;
			_IO = null;
			File.Delete(_path);
		}

		public long Position
		{
			get
			{
				return _sampleLen;
			}
		}

		public long FinalSampleCount
		{
			set { _finalSampleCount = value;  }
		}

		public long BlockSize
		{
			set { }
		}

		public int BitsPerSample
		{
			get { return _bitsPerSample; }
		}

		public void Write(int[,] buff, uint sampleCount)
		{
			if (sampleCount == 0)
				return;
			if (!_headersWritten)
				WriteHeaders();
			if (_sampleBuffer == null || _sampleBuffer.Length < sampleCount * _blockAlign)
				_sampleBuffer = new byte[sampleCount * _blockAlign];
			AudioSamples.FLACSamplesToBytes(buff, 0, _sampleBuffer, 0,
				sampleCount, _channelCount, _bitsPerSample);
			_IO.Write(_sampleBuffer, 0, (int)sampleCount * _blockAlign);
			_sampleLen += sampleCount;
		}

		public string Path { get { return _path; } }
	}

	public class AudioPipe : IAudioSource//, IDisposable
	{
		private readonly Queue<int[,]> _buffer = new Queue<int[,]>();
		int _bitsPerSample, _channelCount, _sampleRate, _bufferPos;
		ulong _sampleLen, _samplePos;
		private int _maxLength;
		private Thread _workThread;
		IAudioSource _source;
		bool _close = false;
		Exception _ex = null;

		public AudioPipe(IAudioSource source, int maxLength)
		{
			_source = source;
			_maxLength = maxLength;
			_bitsPerSample = _source.BitsPerSample;
			_channelCount = _source.ChannelCount;
			_sampleRate = _source.SampleRate;
			_sampleLen = _source.Length;
			_samplePos = 0;
			_bufferPos = 0;
		}

		private void Decompress(object o)
		{
			// catch
			try
			{
				do
				{
					//int[,] buff = new int[65536, 2];
					//uint toRead = Math.Min((uint)buff.GetLength(0), (uint)_source.Remaining);
					//uint samplesRead = _source.Read(buff, toRead);
					int[,] buff = _source.Read(null);
					if (buff == null) break;
					//uint samplesRead = buff.GetLength(0);
					//if (samplesRead == 0) break;
					//if (samplesRead != toRead)
					//    throw new Exception("samples read != samples requested");
					Write(buff);
				} while (true);
			}
			catch (Exception ex)
			{
				lock (_buffer)
				{
					_ex = ex;
					Monitor.Pulse(_buffer);
				}
			}
		}

		private void Go()
		{
			if (_workThread != null || _ex != null) return;
			_workThread = new Thread(Decompress);
			_workThread.Priority = ThreadPriority.BelowNormal;
			_workThread.IsBackground = true;
			_workThread.Start(null);
		}

		//public new void Dispose()
		//{
		//    _buffer.Clear();
		//}

		public void Close()
		{
			lock (_buffer)
			{
				_close = true;
				Monitor.Pulse(_buffer);
			}
			if (_workThread != null)
			{
				_workThread.Join();
				_workThread = null;
			}
			_buffer.Clear();
		}

		public ulong Position
		{
			get
			{
				return _samplePos;
			}
			set
			{
				throw new Exception("not supported");
			}
		}

		public ulong Length
		{
			get
			{
				return _sampleLen;
			}
		}

		public ulong Remaining
		{
			get
			{
				return _sampleLen - _samplePos;
			}
		}

		public int ChannelCount
		{
			get
			{
				return _channelCount;
			}
		}

		public int SampleRate
		{
			get
			{
				return _sampleRate;
			}
		}

		public int BitsPerSample
		{
			get
			{
				return _bitsPerSample;
			}
		}

		public NameValueCollection Tags
		{
			get
			{
				return _source.Tags;
				//return new NameValueCollection();
			}
			set
			{
			}
		}

		public bool UpdateTags(bool preserveTime)
		{
			return false;
		}

		public int[,] Read(int[,] buff)
		{
			Go();
			if (Remaining == 0)
				return null;
			if (_bufferPos != 0)
				throw new Exception("Mixed Read usage not yet suppoted");
			lock (_buffer)
			{
				while (_buffer.Count == 0 && _ex == null)
					Monitor.Wait(_buffer);
				if (_ex != null)
					throw _ex;
				buff = _buffer.Dequeue();
				Monitor.Pulse(_buffer);
			}
			return buff;
		}

		public uint Read(int[,] buff, uint sampleCount)
		{
			Go();
			if (sampleCount > Remaining)
				sampleCount = (uint)Remaining;
			int pos = 0;
			while (sampleCount > 0)
			{
				lock (_buffer)
				{
					while (_buffer.Count == 0 && _ex == null)
						Monitor.Wait(_buffer);
					if (_ex != null)
						throw _ex;
					int[,] chunk = _buffer.Peek();
					int copyCount = Math.Min((int)sampleCount, chunk.GetLength(0) - _bufferPos);
					Array.Copy(chunk, _bufferPos * _channelCount, buff, pos * _channelCount, copyCount * _channelCount);
					pos += copyCount;
					sampleCount -= (uint) copyCount;
					_samplePos += (ulong) copyCount;
					_bufferPos += copyCount;
					if (_bufferPos == chunk.GetLength(0))
					{
						_buffer.Dequeue(); // .Finalize?
						_bufferPos = 0;
						Monitor.Pulse(_buffer);
					}
				}
			}
			return (uint) pos;
		}

		public void Write(int[,] buff)
		{
			lock (_buffer)
			{
				while (_buffer.Count >= _maxLength && !_close)
					Monitor.Wait(_buffer);
				if (_close)
					throw new Exception("Decompression aborted");
				//_flushed = false;
				_buffer.Enqueue(buff);
				Monitor.Pulse(_buffer);
			}
		}

		public string Path { get { return _source.Path; } }
	}

	public class Crc32
	{
		uint[] table = new uint[256];

		public uint ComputeChecksum(uint crc, byte val)
		{
			return (crc >> 8) ^ table[(crc & 0xff) ^ val];
		}

		public uint ComputeChecksum(uint crc, byte[] bytes, int pos, int count)
		{
			for (int i = pos; i < pos + count; i++)
				crc = ComputeChecksum(crc, bytes[i]);
			return crc;
		}

		public uint ComputeChecksum(uint crc, uint s)
		{
			return ComputeChecksum(ComputeChecksum(ComputeChecksum(ComputeChecksum(
				crc, (byte)s), (byte)(s >> 8)), (byte)(s >> 16)), (byte)(s >> 24));
		}

		public unsafe uint ComputeChecksum(uint crc, int * samples, uint count)
		{
			for (uint i = 0; i < count; i++)
			{
				int s1 = samples[2 * i], s2 = samples[2 * i + 1];
				crc = ComputeChecksum(ComputeChecksum(ComputeChecksum(ComputeChecksum(
					crc, (byte)s1), (byte)(s1 >> 8)), (byte)(s2)), (byte)(s2 >> 8));
			}
			return crc;
		}

		uint Reflect(uint val, int ch)
		{
			uint value = 0;
			// Swap bit 0 for bit 7
			// bit 1 for bit 6, etc.
			for (int i = 1; i < (ch + 1); i++)
			{
				if (0 != (val & 1))
					value |= 1U << (ch - i);
				val >>= 1;
			}
			return value;
		}

		const uint ulPolynomial = 0x04c11db7;

		public Crc32()
		{
			for (uint i = 0; i < table.Length; i++)
			{
				table[i] = Reflect(i, 8) << 24;
				for (int j = 0; j < 8; j++)
					table[i] = (table[i] << 1) ^ ((table[i] & (1U << 31)) == 0 ? 0 : ulPolynomial);
				table[i] = Reflect(table[i], 32);
			}
		}
	}
}
