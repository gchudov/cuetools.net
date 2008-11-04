using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Collections.Specialized;

namespace AudioCodecsDotNet
{
	public interface IAudioSource
	{
		uint Read(byte[] buff, uint sampleCount);
		ulong Length { get; }
		ulong Position { get; set; }
		NameValueCollection Tags { get; set; }
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
		long FinalSampleCount { set; }
		string Path { get; }
	}

	public class AudioCodecsDotNet
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
			byte[] outSamples, uint outByteOffset, uint sampleCount, int channelCount)
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
						uint sample_out = (uint)*(pInSamples++);
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
				AudioCodecsDotNet.FLACSamplesToBytes_16(inSamples, inSampleOffset, outSamples, outByteOffset, sampleCount, channelCount);
			else if (bitsPerSample == 24)
				AudioCodecsDotNet.FLACSamplesToBytes_24(inSamples, inSampleOffset, outSamples, outByteOffset, sampleCount, channelCount);
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
	}

	public class DummyWriter : IAudioDest
	{
		public DummyWriter(string path, int bitsPerSample, int channelCount, int sampleRate)
		{
		}

		public bool SetTags(NameValueCollection tags)
		{
			return false;
		}

		public void Close()
		{
		}

		public long FinalSampleCount
		{
			set
			{
			}
		}

		public void Write(int[,] buff, uint sampleCount)
		{
		}

		public string Path { get { return null; } }
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

		public uint Read(byte[] buff, uint sampleCount)
		{
			uint samplesRemaining, byteCount, i;

			samplesRemaining = (uint)(_sampleCount - _sampleOffset);
			if (sampleCount > samplesRemaining)
			{
				sampleCount = samplesRemaining;
			}

			byteCount = sampleCount * 2 * 2;
			for (i = 0; i < byteCount; i++)
			{
				buff[i] = 0;
			}

			_sampleOffset += sampleCount;

			return sampleCount;
		}

		public void Close()
		{
		}

		public string Path { get { return null; } }
	}

	public class WAVReader : IAudioSource
	{
		FileStream _fs;
		BinaryReader _br;
		ulong _dataOffset, _dataLen;
		ulong _samplePos, _sampleLen;
		int _bitsPerSample, _channelCount, _sampleRate, _blockAlign;
		bool _largeFile;
		string _path;

		public WAVReader(string path)
		{
			_path = path;
			//_fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
			_fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 0x10000, FileOptions.SequentialScan);
			_br = new BinaryReader(_fs);

			ParseHeaders();

			_sampleLen = _dataLen / (uint)_blockAlign;
			Position = 0;
		}

		public void Close()
		{
			_br.Close();

			_br = null;
			_fs = null;
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

			while (_fs.Position < fileEnd)
			{
				uint ckID, ckSize, ckSizePadded;
				long ckEnd;

				ckID = _br.ReadUInt32();
				ckSize = _br.ReadUInt32();
				ckSizePadded = (ckSize + 1U) & ~1U;
				ckEnd = _fs.Position + (long)ckSizePadded;

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
				}
				else if (ckID == fccData)
				{
					foundData = true;

					_dataOffset = (ulong)_fs.Position;
					if (_fs.Length <= maxFileSize)
					{
						_dataLen = ckSize;
					}
					else
					{
						_largeFile = true;
						_dataLen = ((ulong)_fs.Length) - _dataOffset;
					}
				}

				if ((foundFormat & foundData) || _largeFile)
				{
					break;
				}

				_fs.Seek(ckEnd, SeekOrigin.Begin);
			}

			if ((foundFormat & foundData) == false)
			{
				throw new Exception("Format or data chunk not found.");
			}

			if (_channelCount <= 0)
			{
				throw new Exception("Channel count is invalid.");
			}
			if (_sampleRate <= 0)
			{
				throw new Exception("Sample rate is invalid.");
			}
			if (_blockAlign != (_channelCount * ((_bitsPerSample + 7) / 8)))
			{
				throw new Exception("Block align is invalid.");
			}
			if ((_bitsPerSample <= 0) || (_bitsPerSample > 32))
			{
				throw new Exception("Bits per sample is invalid.");
			}
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
				_fs.Seek((long)seekPos, SeekOrigin.Begin);
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

		public void GetTags(out List<string> names, out List<string> values)
		{
			names = new List<string>();
			values = new List<string>();
		}

		public uint Read(byte[] buff, uint sampleCount)
		{
			if (sampleCount > Remaining)
				sampleCount = (uint)Remaining;

			uint byteCount = sampleCount * (uint)_blockAlign;

			if (sampleCount != 0)
			{
				if (_fs.Read(buff, 0, (int)byteCount) != byteCount)
				{
					throw new Exception("Incomplete file read.");
				}
				_samplePos += sampleCount;
			}

			return sampleCount;
		}

		public string Path { get { return _path; } }
	}

	public class WAVWriter : IAudioDest
	{
		FileStream _fs;
		BinaryWriter _bw;
		int _bitsPerSample, _channelCount, _sampleRate, _blockAlign;
		long _sampleLen;
		string _path;
		private byte[] _sampleBuffer;

		public WAVWriter(string path, int bitsPerSample, int channelCount, int sampleRate)
		{
			_path = path;
			_bitsPerSample = bitsPerSample;
			_channelCount = channelCount;
			_sampleRate = sampleRate;
			_blockAlign = _channelCount * ((_bitsPerSample + 7) / 8);

			_fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
			_bw = new BinaryWriter(_fs);

			WriteHeaders();
		}

		public bool SetTags(NameValueCollection tags)
		{
			return false;
		}

		private void WriteHeaders()
		{
			const uint fccRIFF = 0x46464952;
			const uint fccWAVE = 0x45564157;
			const uint fccFormat = 0x20746D66;
			const uint fccData = 0x61746164;

			_bw.Write(fccRIFF);
			_bw.Write((uint)0);
			_bw.Write(fccWAVE);

			_bw.Write(fccFormat);
			_bw.Write((uint)16);
			_bw.Write((ushort)1);
			_bw.Write((ushort)_channelCount);
			_bw.Write((uint)_sampleRate);
			_bw.Write((uint)(_sampleRate * _blockAlign));
			_bw.Write((ushort)_blockAlign);
			_bw.Write((ushort)_bitsPerSample);

			_bw.Write(fccData);
			_bw.Write((uint)0);
		}

		public void Close()
		{
			const long maxFileSize = 0x7FFFFFFEL;
			long dataLen, dataLenPadded;

			dataLen = _sampleLen * _blockAlign;

			if ((dataLen & 1) == 1)
			{
				_bw.Write((byte)0);
			}

			if ((dataLen + 44) > maxFileSize)
			{
				dataLen = ((maxFileSize - 44) / _blockAlign) * _blockAlign;
			}

			dataLenPadded = ((dataLen & 1) == 1) ? (dataLen + 1) : dataLen;

			_bw.Seek(4, SeekOrigin.Begin);
			_bw.Write((uint)(dataLenPadded + 36));

			_bw.Seek(40, SeekOrigin.Begin);
			_bw.Write((uint)dataLen);

			_bw.Close();

			_bw = null;
			_fs = null;
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
			set
			{
			}
		}

		public void Write(int[,] buff, uint sampleCount)
		{
			if (sampleCount == 0)
				return;
			if (_sampleBuffer == null || _sampleBuffer.Length < sampleCount * _channelCount)
				_sampleBuffer = new byte[sampleCount * _blockAlign];
			AudioCodecsDotNet.FLACSamplesToBytes(buff, 0, _sampleBuffer, 0,
				sampleCount, _channelCount, _bitsPerSample);
			_fs.Write(_sampleBuffer, 0, (int)sampleCount * _blockAlign);
			_sampleLen += sampleCount;
		}

		public string Path { get { return _path; } }
	}
}
