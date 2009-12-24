/**
 * CUETools.Codecs: common audio encoder/decoder routines
 * Copyright (c) 2009 Gregory S. Chudov
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Threading;
using System.Diagnostics;

namespace CUETools.Codecs
{
	public interface IAudioSource
	{
		uint Read(int[,] buff, uint sampleCount);
		int[,] Read(int[,] buff);
		ulong Length { get; }
		ulong Position { get; set; }
		ulong Remaining { get; }
		void Close();
		int BitsPerSample { get; }
		int ChannelCount { get; }
		int SampleRate { get; }
		string Path { get; }
	}

	public interface IAudioDest
	{
		void Write(int[,] buff, int pos, int sampleCount);
		void Close();
		void Delete();
		int BitsPerSample { get; }
		long FinalSampleCount { set; }
		long BlockSize { set; }
		string Path { get; }
	}

	public class AudioSamples
	{
		public static unsafe void FLACSamplesToBytes_16(int[,] inSamples, int inSampleOffset,
			byte[] outSamples, int outByteOffset, int sampleCount, int channelCount)
		{
			int loopCount = sampleCount * channelCount;

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
		
		public static unsafe void FLACSamplesToBytes_24(int[,] inSamples, int inSampleOffset,
			byte[] outSamples, int outByteOffset, int sampleCount, int channelCount, int wastedBits)
		{
			int loopCount = sampleCount * channelCount;

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

		public static unsafe void FLACSamplesToBytes(int[,] inSamples, int inSampleOffset,
			byte[] outSamples, int outByteOffset, int sampleCount, int channelCount, int bitsPerSample)
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

		public static unsafe void BytesToFLACSamples_24(byte[] inSamples, int inByteOffset,
			int[,] outSamples, int outSampleOffset, uint sampleCount, int channelCount, int wastedBits)
		{
			uint loopCount = sampleCount * (uint)channelCount;

			if ((inSamples.Length - inByteOffset < loopCount * 3) ||
				(outSamples.GetLength(0) - outSampleOffset < sampleCount))
				throw new IndexOutOfRangeException();

			fixed (byte* pInSamplesFixed = &inSamples[inByteOffset])
			{
				fixed (int* pOutSamplesFixed = &outSamples[outSampleOffset, 0])
				{
					byte* pInSamples = (byte*)pInSamplesFixed;
					int* pOutSamples = pOutSamplesFixed;
					for (int i = 0; i < loopCount; i++)
					{
						int sample = (int)*(pInSamples++);
						sample += (int)*(pInSamples++) << 8;
						sample += (int)*(pInSamples++) << 16;
						*(pOutSamples++) = (sample << 8) >> (8 + wastedBits);
					}
				}
			}
		}

		public static unsafe void BytesToFLACSamples(byte[] inSamples, int inByteOffset,
			int[,] outSamples, int outSampleOffset, uint sampleCount, int channelCount, int bitsPerSample)
		{
			if (bitsPerSample == 16)
				AudioSamples.BytesToFLACSamples_16(inSamples, inByteOffset, outSamples, outSampleOffset, sampleCount, channelCount);
			else if (bitsPerSample > 16 && bitsPerSample <= 24)
				AudioSamples.BytesToFLACSamples_24(inSamples, inByteOffset, outSamples, outSampleOffset, sampleCount, channelCount, 24 - bitsPerSample);
			else
				throw new Exception("Unsupported bitsPerSample value");
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

		unsafe public static void Interlace(int* res, int* src1, int* src2, int n)
		{
			for (int i = n; i > 0; i--)
			{
				*(res++) = *(src1++);
				*(res++) = *(src2++);
			}
		}

		unsafe public static void Deinterlace(int* dst1, int* dst2, int* src, int n)
		{
			for (int i = n; i > 0; i--)
			{
				*(dst1++) = *(src++);
				*(dst2++) = *(src++);
			}
		}

		unsafe public static bool MemCmp(int* res, int* smp, int n)
		{
			for (int i = n; i > 0; i--)
				if (*(res++) != *(smp++))
					return true;
			return false;
		}

		unsafe public static void MemCpy(uint* res, uint* smp, int n)
		{
			for (int i = n; i > 0; i--)
				*(res++) = *(smp++);
		}

		unsafe public static void MemCpy(int* res, int* smp, int n)
		{
			for (int i = n; i > 0; i--)
				*(res++) = *(smp++);
		}

		unsafe public static void MemCpy(short* res, short* smp, int n)
		{
			for (int i = n; i > 0; i--)
				*(res++) = *(smp++);
		}

		unsafe public static void MemCpy(byte* res, byte* smp, int n)
		{
			for (int i = n; i > 0; i--)
				*(res++) = *(smp++);
		}

		unsafe public static void MemSet(int* res, int smp, int n)
		{
			for (int i = n; i > 0; i--)
				*(res++) = smp;
		}

		public const uint UINT32_MAX = 0xffffffff;
	}

	public class DummyWriter : IAudioDest
	{
		public DummyWriter(string path, int bitsPerSample, int channelCount, int sampleRate)
		{
			_bitsPerSample = bitsPerSample;
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

		public void Write(int[,] buff, int pos, int sampleCount)
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
		ulong _dataOffset, _samplePos, _sampleLen;
		long _dataLen;
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

			if (_dataLen < 0)
				//_sampleLen = 0;
				throw new Exception("WAVE stream length unknown");
			else
				_sampleLen = (ulong)(_dataLen / _blockAlign);
		}

		public WAVReader(Stream IO)
		{
			_path = "";
			_IO = IO;
			_br = new BinaryReader(_IO);
			ParseHeaders();
			if (_dataLen < 0)
				_sampleLen = 0;
			else
				_sampleLen = (ulong)(_dataLen / _blockAlign);
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
						if (ckSize >= 0x7fffffff)
							_dataLen = -1;
						else
							_dataLen = (long)ckSize;
					}
					else
					{
						_largeFile = true;
						_dataLen = _IO.Length - pos;
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

				if (_sampleLen != 0 && value > _sampleLen)
					_samplePos = _sampleLen;
				else
					_samplePos = value;

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

		public uint Read(int[,] buff, uint sampleCount)
		{
			if (_sampleLen > 0 && sampleCount > Remaining)
				sampleCount = (uint)Remaining;

			if (sampleCount == 0)
				return 0;
			int byteCount = (int) sampleCount * _blockAlign;
			if (_sampleBuffer == null || _sampleBuffer.Length < byteCount)
				_sampleBuffer = new byte[byteCount];
			int pos = 0;
			do
			{
				int len = _IO.Read(_sampleBuffer, pos, (int)byteCount - pos);
				if (len <= 0)
				{
					if ((pos % BlockAlign) != 0 || _sampleLen > 0)
						throw new Exception("Incomplete file read.");
					sampleCount = (uint)(pos / BlockAlign);
					_sampleLen = _samplePos + sampleCount;
					break;
				}
				pos += len;
			} while (pos < byteCount);
			AudioSamples.BytesToFLACSamples(_sampleBuffer, 0, buff, 0,
				sampleCount, _channelCount, _bitsPerSample);
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

		public void Write(int[,] buff, int pos, int sampleCount)
		{
			if (sampleCount == 0)
				return;
			if (!_headersWritten)
				WriteHeaders();
			if (_sampleBuffer == null || _sampleBuffer.Length < sampleCount * _blockAlign)
				_sampleBuffer = new byte[sampleCount * _blockAlign];
			AudioSamples.FLACSamplesToBytes(buff, pos, _sampleBuffer, 0,
				sampleCount, _channelCount, _bitsPerSample);
			_IO.Write(_sampleBuffer, 0, (int)sampleCount * _blockAlign);
			_sampleLen += sampleCount;
		}

		public string Path { get { return _path; } }
	}

	public class CyclicBuffer
	{
		public delegate void FlushOutput(byte[] buffer, int pos, int chunk, object to);
		public delegate void CloseOutput(object to);

		private byte[] _buffer;
		private int _size;
		private int _start = 0; // moved only by Write
		private int _end = 0; // moved only by Read
		private bool _eof = false;
		private Thread _readThread = null, _writeThread = null;

		public event FlushOutput flushOutput;
		public event CloseOutput closeOutput;

		public CyclicBuffer(int len)
		{
			_size = len;
			_buffer = new byte[len];
		}

		public CyclicBuffer(int len, Stream input, Stream output)
		{
			_size = len;
			_buffer = new byte[len];
			ReadFrom(input);
			WriteTo(output);
		}

		public void ReadFrom(Stream input)
		{
			_readThread = new Thread(PumpRead);
			_readThread.Priority = ThreadPriority.Highest;
			_readThread.IsBackground = true;
			_readThread.Start(input);
		}

		public void WriteTo(Stream output)
		{
			WriteTo(flushOutputToStream, closeOutputToStream, ThreadPriority.Highest, output);
		}

		public void WriteTo(FlushOutput flushOutputDelegate, CloseOutput closeOutputDelegate, ThreadPriority priority, object to)
		{
			if (flushOutputDelegate != null)
				flushOutput += flushOutputDelegate;
			if (closeOutputDelegate != null)
				closeOutput += closeOutputDelegate;
			_writeThread = new Thread(FlushThread);
			_writeThread.Priority = priority;
			_writeThread.IsBackground = true;
			_writeThread.Start(to);
		}

		void closeOutputToStream(object to)
		{
			((Stream)to).Close();
		}

		void flushOutputToStream(byte[] buffer, int pos, int chunk, object to)
		{
			((Stream)to).Write(buffer, pos, chunk);
		}

		int DataAvailable
		{
			get
			{
				return _end - _start;
			}
		}

		int FreeSpace
		{
			get
			{
				return _size - DataAvailable;
			}
		}

		private void PumpRead(object o)
		{
			while (Read((Stream)o))
				;
			SetEOF();
		}

		public void Close()
		{
			if (_readThread != null)
			{
				_readThread.Join();
				_readThread = null;
			}
			SetEOF();
			if (_writeThread != null)
			{
				_writeThread.Join();
				_writeThread = null;
			}
		}

		public void SetEOF()
		{
			lock (this)
			{
				_eof = true;
				Monitor.Pulse(this);
			}
		}

		public bool Read(Stream input)
		{
			int pos, chunk;
			lock (this)
			{
				while (FreeSpace == 0)
					Monitor.Wait(this);
				pos = _end % _size;
				chunk = Math.Min(FreeSpace, _size - pos);
			}
			chunk = input.Read(_buffer, pos, chunk);
			if (chunk == 0)
				return false;
			lock (this)
			{
				_end += chunk;
				Monitor.Pulse(this);
			}
			return true;
		}

		public void Read(byte[] array, int offset, int count)
		{
			int pos, chunk;
			while (count > 0)
			{
				lock (this)
				{
					while (FreeSpace == 0)
						Monitor.Wait(this);
					pos = _end % _size;
					chunk = Math.Min(FreeSpace, _size - pos);
					chunk = Math.Min(chunk, count);
				}
				Array.Copy(array, offset, _buffer, pos, chunk);
				lock (this)
				{
					_end += chunk;
					Monitor.Pulse(this);
				}
				count -= chunk;
				offset += chunk;
			}
		}

		private void FlushThread(object to)
		{
			while (true)
			{
				int pos, chunk;
				lock (this)
				{
					while (DataAvailable == 0 && !_eof)
						Monitor.Wait(this);
					if (DataAvailable == 0)
						break;
					pos = _start % _size;
					chunk = Math.Min(DataAvailable, _size - pos);
				}
				if (flushOutput != null)
					flushOutput(_buffer, pos, chunk, to);
				lock (this)
				{
					_start += chunk;
					Monitor.Pulse(this);
				}
			}
			if (closeOutput != null)
				closeOutput(to);
		}
	}

	public class CycilcBufferOutputStream : Stream
	{
		CyclicBuffer _buffer;

		public CycilcBufferOutputStream(CyclicBuffer buffer)
		{
			_buffer = buffer;
		}

		public CycilcBufferOutputStream(Stream output, int size)
		{
			_buffer = new CyclicBuffer(size);
			_buffer.WriteTo(output);
		}

		public override bool CanRead
		{
			get { return false; }
		}

		public override bool CanSeek
		{
			get { return false; }
		}

		public override bool CanWrite
		{
			get { return true; }
		}

		public override long Length
		{
			get
			{
				throw new NotSupportedException();
			}
		}

		public override long Position
		{
			get { throw new NotSupportedException(); }
			set { throw new NotSupportedException(); }
		}

		public override void Close()
		{
			_buffer.Close();
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
			throw new NotSupportedException();
		}

		public override long Seek(long offset, SeekOrigin origin)
		{
			throw new NotSupportedException();
		}

		public override void Write(byte[] array, int offset, int count)
		{
			_buffer.Read(array, offset, count);
		}
	}

	public class UserDefinedWriter : IAudioDest
	{
		string _path, _encoder, _encoderParams, _encoderMode;
		Process _encoderProcess;
		WAVWriter wrt;
		CyclicBuffer outputBuffer = null;

		public UserDefinedWriter(string path, int bitsPerSample, int channelCount, int sampleRate, Stream IO, string encoder, string encoderParams, string encoderMode, int padding)
		{
			_path = path;
			_encoder = encoder;
			_encoderParams = encoderParams;
			_encoderMode = encoderMode;

			_encoderProcess = new Process();
			_encoderProcess.StartInfo.FileName = _encoder;
			_encoderProcess.StartInfo.Arguments = _encoderParams.Replace("%O", "\"" + path + "\"").Replace("%M", encoderMode).Replace("%P", padding.ToString());
			_encoderProcess.StartInfo.CreateNoWindow = true;
			_encoderProcess.StartInfo.RedirectStandardInput = true;
			_encoderProcess.StartInfo.UseShellExecute = false;
			if (!_encoderParams.Contains("%O"))
				_encoderProcess.StartInfo.RedirectStandardOutput = true;
			bool started = false;
			Exception ex = null;
			try
			{
				started = _encoderProcess.Start();
				if (started)
					_encoderProcess.PriorityClass = Process.GetCurrentProcess().PriorityClass;
			}
			catch (Exception _ex)
			{
				ex = _ex;
			}
			if (!started)
				throw new Exception(_encoder + ": " + (ex == null ? "please check the path" : ex.Message));
			if (_encoderProcess.StartInfo.RedirectStandardOutput)
			{
				Stream outputStream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
				outputBuffer = new CyclicBuffer(2 * 1024 * 1024, _encoderProcess.StandardOutput.BaseStream, outputStream);
			}
			Stream inputStream = new CycilcBufferOutputStream(_encoderProcess.StandardInput.BaseStream, 128 * 1024);
			wrt = new WAVWriter(path, bitsPerSample, channelCount, sampleRate, inputStream);
		}

		public void Close()
		{
			wrt.Close();
			if (!_encoderProcess.HasExited)
				_encoderProcess.WaitForExit();
			if (outputBuffer != null)
				outputBuffer.Close();
			if (_encoderProcess.ExitCode != 0)
				throw new Exception(String.Format("{0} returned error code {1}", _encoder, _encoderProcess.ExitCode));
		}

		public void Delete()
		{
			Close();
			File.Delete(_path);
		}

		public long Position
		{
			get
			{
				return wrt.Position;
			}
		}

		public long FinalSampleCount
		{
			set { wrt.FinalSampleCount = value; }
		}

		public long BlockSize
		{
			set { }
		}

		public int BitsPerSample
		{
			get { return wrt.BitsPerSample; }
		}

		public void Write(int[,] buff, int pos, int sampleCount)
		{
			wrt.Write(buff, pos, sampleCount);
			//_sampleLen += sampleCount;
		}

		public string Path { get { return _path; } }
	}

	public class UserDefinedReader : IAudioSource
	{
		string _path, _decoder, _decoderParams;
		Process _decoderProcess;
		WAVReader rdr;

		public UserDefinedReader(string path, Stream IO, string decoder, string decoderParams)
		{
			_path = path;
			_decoder = decoder;
			_decoderParams = decoderParams;
			_decoderProcess = null;
			rdr = null;
		}

		void Initialize()
		{
			if (_decoderProcess != null)
				return;
			_decoderProcess = new Process();
			_decoderProcess.StartInfo.FileName = _decoder;
			_decoderProcess.StartInfo.Arguments = _decoderParams.Replace("%I", "\"" + _path + "\"");
			_decoderProcess.StartInfo.CreateNoWindow = true;
			_decoderProcess.StartInfo.RedirectStandardOutput = true;
			_decoderProcess.StartInfo.UseShellExecute = false;
			bool started = false;
			Exception ex = null;
			try
			{
				started = _decoderProcess.Start();
				if (started)
					_decoderProcess.PriorityClass = Process.GetCurrentProcess().PriorityClass;
			}
			catch (Exception _ex)
			{
				ex = _ex;
			}
			if (!started)
				throw new Exception(_decoder + ": " + (ex == null ? "please check the path" : ex.Message));
			rdr = new WAVReader(_path, _decoderProcess.StandardOutput.BaseStream);
		}

		public void Close()
		{
			if (rdr != null)
				rdr.Close();
			if (_decoderProcess != null && !_decoderProcess.HasExited)
				try { _decoderProcess.Kill(); _decoderProcess.WaitForExit(); }
				catch { }
		}

		public ulong Position
		{
			get
			{
				Initialize();
				return rdr.Position;
			}
			set
			{
				Initialize();
				rdr.Position = value;
			}
		}

		public ulong Length
		{
			get
			{
				Initialize();
				return rdr.Length;
			}
		}

		public ulong Remaining
		{
			get
			{
				Initialize();
				return rdr.Remaining;
			}
		}

		public int ChannelCount
		{
			get
			{
				Initialize();
				return rdr.ChannelCount;
			}
		}

		public int SampleRate
		{
			get
			{
				Initialize();
				return rdr.SampleRate;
			}
		}

		public int BitsPerSample
		{
			get
			{
				Initialize();
				return rdr.BitsPerSample;
			}
		}

		public int BlockAlign
		{
			get
			{
				Initialize();
				return rdr.BlockAlign;
			}
		}

		public uint Read(int[,] buff, uint sampleCount)
		{
			Initialize();
			return rdr.Read(buff, sampleCount);
		}

		public int[,] Read(int[,] buff)
		{
			return AudioSamples.Read(this, buff);
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

	public class BufferedWriter: IAudioDest
	{
		IAudioDest _writer;
		Thread _flushThread = null;
		private int[,] _buffer;
		private int _size;
		private int _channels = 2;
		private int _start = 0; // moved only by Write
		private int _end = 0; // moved only by Read
		private bool _eof = false, _delete = false;
		Exception exceptionOnFlush = null;

		public long FinalSampleCount
		{
			//get { return _writer.FinalSampleCount; }
			set	{ _writer.FinalSampleCount = value; }
		}

		public long BlockSize
		{
			set	{ _writer.BlockSize = value; }
		}

		public int BitsPerSample
		{
			get { return _writer.BitsPerSample;  }
		}

		public int Channels
		{
			get { return _channels; } // !!!! writer.Channels
		}

		// public bool ReadSource(IAudioSource input)

		public unsafe void Write(int[,] samples, int offset, int count)
		{
			int pos, chunk;
			while (count > 0)
			{
				lock (this)
				{
					while (FreeSpace == 0 && exceptionOnFlush == null)
						Monitor.Wait(this);
					if (exceptionOnFlush != null)
					{
						Exception ex = exceptionOnFlush;
						exceptionOnFlush = null;
						throw ex;
					}
					pos = _end % _size;
					chunk = Math.Min(FreeSpace, _size - pos);
					chunk = Math.Min(chunk, count);
				}
				fixed (int* src = &samples[offset, 0], dst = &_buffer[pos, 0])
					AudioSamples.MemCpy(dst, src, chunk * Channels);
				//Array.Copy(samples, offset * Channels, _buffer, pos * Channels, chunk * Channels);
				lock (this)
				{
					_end += chunk;
					Monitor.Pulse(this);
				}
				count -= chunk;
				offset += chunk;
			}
		}

		public string Path { get { return _writer.Path; } }

		public BufferedWriter(IAudioDest writer, int size)
		{
			_writer = writer;
			_size = size;
			_buffer = new int[_size, Channels]; 
			_flushThread = new Thread(Flush);
			//_writeThread.Priority = ThreadPriority.Normal;
			_flushThread.IsBackground = true;
			_flushThread.Start(_writer);
		}

		int DataAvailable
		{
			get
			{
				return _end - _start;
			}
		}

		int FreeSpace
		{
			get
			{
				return _size - DataAvailable;
			}
		}

		public void Delete()
		{
			SetEOF(true);
			if (_flushThread != null)
			{
				_flushThread.Join();
				_flushThread = null;
			}
		}

		public void Close()
		{
			SetEOF(false);
			if (_flushThread != null)
			{
				_flushThread.Join();
				_flushThread = null;
			}
			if (exceptionOnFlush != null)
			{
				Exception ex = exceptionOnFlush;
				exceptionOnFlush = null;
				throw ex;
			}
		}

		public void SetEOF(bool delete)
		{
			lock (this)
			{
				_eof = true;
				_delete = delete;
				Monitor.Pulse(this);
			}
		}
		
		public void Flush(object o)
		{
			IAudioDest dest = (IAudioDest)o;
			try
			{
				do
				{
					int pos, chunk;
					lock (this)
					{
						while (DataAvailable == 0 && !_eof)
							Monitor.Wait(this);
						if (DataAvailable == 0)
						{
							if (_delete)
								dest.Delete();
							else
								dest.Close();
							return;
						}
						pos = _start % _size;
						chunk = Math.Min(DataAvailable, _size - pos);
					}
					dest.Write(_buffer, pos, chunk);
					lock (this)
					{
						_start += chunk;
						Monitor.Pulse(this);
					}
				}
				while (true);
			}
			catch (Exception ex)
			{
				lock (this)
				{
					exceptionOnFlush = ex;
					Monitor.Pulse(this);
					dest.Delete();
				}
			}
		}
	}
	
	public class NullStream : Stream
	{
		public NullStream()
		{
		}

		public override bool CanRead
		{
			get { return false; }
		}

		public override bool CanSeek
		{
			get { return false; }
		}

		public override bool CanWrite
		{
			get { return true; }
		}

		public override long Length
		{
			get
			{
				throw new NotSupportedException();
			}
		}

		public override long Position
		{
			get { throw new NotSupportedException(); }
			set { throw new NotSupportedException(); }
		}

		public override void Close()
		{
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
			throw new NotSupportedException();
		}

		public override long Seek(long offset, SeekOrigin origin)
		{
			throw new NotSupportedException();
		}

		public override void Write(byte[] array, int offset, int count)
		{
		}
	}
}
