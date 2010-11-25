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
		int Read(AudioBuffer buffer, int maxLength);
		void Close();

		AudioPCMConfig PCM { get; }
		string Path { get; }

		long Length { get; }
		long Position { get; set; }
		long Remaining { get; }
	}

	public interface IAudioDest
	{
		void Write(AudioBuffer buffer);
		void Close();
		void Delete();

		AudioPCMConfig PCM { get; }
		string Path { get; }

		int CompressionLevel { get; set; }
		object Settings { get;  set; }
		long FinalSampleCount { set; }
		long BlockSize { set; }
		long Padding { set; }
	}

	public interface IAudioFilter
	{
		IAudioDest AudioDest { set; }
	}

	/// <summary>
	///    This class provides an attribute for marking
	///    classes that provide <see cref="IAudioDest" />.
	/// </summary>
	/// <remarks>
	///    When plugins with classes that provide <see cref="IAudioDest" /> are
	///    registered, their <see cref="AudioEncoderClass" /> attributes are read.
	/// </remarks>
	/// <example>
	///    <code lang="C#">using CUETools.Codecs;
	///
	///[AudioEncoderClass("libFLAC", "flac", true, "0 1 2 3 4 5 6 7 8", "5", 1)]
	///public class MyEncoder : IAudioDest {
	///	...
	///}</code>
	/// </example>
	[AttributeUsage(AttributeTargets.Class, AllowMultiple = true)]
	public sealed class AudioEncoderClass : Attribute
	{
		private string _encoderName, _extension, _supportedModes, _defaultMode;
		bool _lossless;
		int _priority;
		Type _settings;

		public AudioEncoderClass(string encoderName, string extension, bool lossless, string supportedModes, string defaultMode, int priority, Type settings)
		{
			_encoderName = encoderName;
			_extension = extension;
			_supportedModes = supportedModes;
			_defaultMode = defaultMode;
			_lossless = lossless;
			_priority = priority;
			_settings = settings;
		}

		public string EncoderName
		{
			get { return _encoderName; }
		}

		public string Extension
		{
			get { return _extension; }
		}

		public string SupportedModes
		{
			get { return _supportedModes; }
		}

		public string DefaultMode
		{
			get { return _defaultMode; }
		}

		public bool Lossless
		{
			get { return _lossless; }
		}

		public int Priority
		{
			get { return _priority; }
		}

		public Type Settings
		{
			get { return _settings; }
		}
	}

	/// <summary>
	///    This class provides an attribute for marking
	///    classes that provide <see cref="IAudioSource" />.
	/// </summary>
	/// <remarks>
	///    When plugins with classes that provide <see cref="IAudioSource" /> are
	///    registered, their <see cref="AudioDecoderClass" /> attributes are read.
	/// </remarks>
	/// <example>
	///    <code lang="C#">using CUETools.Codecs;
	///
	///[AudioDecoderClass("libFLAC", "flac")]
	///public class MyDecoder : IAudioSource {
	///	...
	///}</code>
	/// </example>
	[AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
	public sealed class AudioDecoderClass : Attribute
	{
		private string _decoderName, _extension;

		public AudioDecoderClass(string decoderName, string extension)
		{
			_decoderName = decoderName;
			_extension = extension;
		}

		public string DecoderName
		{
			get { return _decoderName; }
		}

		public string Extension
		{
			get { return _extension; }
		}
	}

	public class AudioPCMConfig
	{
		private int _bitsPerSample;
		private int _channelCount;
		private int _sampleRate;
		
		public AudioPCMConfig(int bitsPerSample, int channelCount, int sampleRate)
		{
			_bitsPerSample = bitsPerSample;
			_channelCount = channelCount;
			_sampleRate = sampleRate;
		}

		public static readonly AudioPCMConfig RedBook = new AudioPCMConfig(16, 2, 44100);

		public int BitsPerSample { get { return _bitsPerSample; } }
		public int ChannelCount { get { return _channelCount; } }
		public int SampleRate { get { return _sampleRate; } }
		public int BlockAlign { get { return _channelCount * ((_bitsPerSample + 7) / 8); } }
		public bool IsRedBook { get { return _bitsPerSample == 16 && _channelCount == 2 && _sampleRate == 44100; } }
	}

	public class AudioBuffer
	{
		private int[,] samples;
		private float[,] fsamples;
		private byte[] bytes;
		private int length;
		private int size;
		private AudioPCMConfig pcm;
		private bool dataInSamples = false;
		private bool dataInBytes = false;
		private bool dataInFloat = false;

		public int Length
		{
			get
			{
				return length;
			}
			set
			{
				length = value;
			}
		}

		public int Size
		{
			get
			{
				return size;
			}
		}

		public AudioPCMConfig PCM { get { return pcm; } }

		public int ByteLength
		{
			get
			{
				return length * pcm.BlockAlign;
			}
		}

		public int[,] Samples
		{
			get
			{
				if (samples == null || samples.GetLength(0) < length)
					samples = new int[size, pcm.ChannelCount];
				if (!dataInSamples && dataInBytes && length != 0)
					BytesToFLACSamples(bytes, 0, samples, 0, length, pcm.ChannelCount, pcm.BitsPerSample);
				dataInSamples = true;
				return samples;
			}
		}

		public float[,] Float
		{
			get
			{
				if (fsamples == null || fsamples.GetLength(0) < length)
					fsamples = new float[size, pcm.ChannelCount];
				if (!dataInFloat && dataInBytes && length != 0)
				{
					if (pcm.BitsPerSample == 16)
						Bytes16ToFloat(bytes, 0, fsamples, 0, length, pcm.ChannelCount);
					//else if (pcm.BitsPerSample > 16 && PCM.BitsPerSample <= 24)
					//    BytesToFLACSamples_24(bytes, 0, fsamples, 0, length, pcm.ChannelCount, 24 - pcm.BitsPerSample);
					else if (pcm.BitsPerSample == 32)
						Buffer.BlockCopy(bytes, 0, fsamples, 0, length * 4 * pcm.ChannelCount);
					else
						throw new Exception("Unsupported bitsPerSample value");
				}
				dataInFloat = true;
				return fsamples;
			}
		}

		public byte[] Bytes
		{
			get
			{
				if (bytes == null || bytes.Length < length * pcm.BlockAlign)
					bytes = new byte[size * pcm.BlockAlign];
				if (!dataInBytes && length != 0)
				{
					if (dataInSamples)
						FLACSamplesToBytes(samples, 0, bytes, 0, length, pcm.ChannelCount, pcm.BitsPerSample);
					else if (dataInFloat)
						FloatToBytes(fsamples, 0, bytes, 0, length, pcm.ChannelCount, pcm.BitsPerSample);
				}
				dataInBytes = true;
				return bytes;
			}
		}

		public AudioBuffer(AudioPCMConfig _pcm, int _size)
		{
			pcm = _pcm;
			size = _size;
			length = 0;
		}

		public AudioBuffer(AudioPCMConfig _pcm, int[,] _samples, int _length)
		{
			pcm = _pcm;
			// assert _samples.GetLength(1) == pcm.ChannelCount
			Prepare(_samples, _length);
		}

		public AudioBuffer(AudioPCMConfig _pcm, byte[] _bytes, int _length)
		{
			pcm = _pcm;
			Prepare(_bytes, _length);
		}

		public AudioBuffer(IAudioSource source, int _size)
		{
			pcm = source.PCM;
			size = _size;
		}

		public void Prepare(IAudioDest dest)
		{
			if (dest.PCM.ChannelCount != pcm.ChannelCount || dest.PCM.BitsPerSample != pcm.BitsPerSample)
				throw new Exception("AudioBuffer format mismatch");
		}

		public void Prepare(IAudioSource source, int maxLength)
		{
			if (source.PCM.ChannelCount != pcm.ChannelCount || source.PCM.BitsPerSample != pcm.BitsPerSample)
				throw new Exception("AudioBuffer format mismatch");
			length = size;
			if (maxLength >= 0)
				length = Math.Min(length, maxLength);
			if (source.Remaining >= 0)
				length = (int)Math.Min((long)length, source.Remaining);
			dataInBytes = false;
			dataInSamples = false;
			dataInFloat = false;
		}

		public void Prepare(int maxLength)
		{
			length = size;
			if (maxLength >= 0)
				length = Math.Min(length, maxLength);
			dataInBytes = false;
			dataInSamples = false;
			dataInFloat = false;
		}

		public void Prepare(int[,] _samples, int _length)
		{
			length = _length;
			size = _samples.GetLength(0);
			samples = _samples;
			dataInSamples = true;
			dataInBytes = false;
			dataInFloat = false;
			if (length > size)
				throw new Exception("Invalid length");
		}

		public void Prepare(byte[] _bytes, int _length)
		{
			length = _length;
			size = _bytes.Length / PCM.BlockAlign;
			bytes = _bytes;
			dataInSamples = false;
			dataInBytes = true;
			dataInFloat = false;
			if (length > size)
				throw new Exception("Invalid length");
		}

		internal unsafe void Load(int dstOffset, AudioBuffer src, int srcOffset, int copyLength)
		{
			if (dataInBytes)
				Buffer.BlockCopy(src.Bytes, srcOffset * pcm.BlockAlign, Bytes, dstOffset * pcm.BlockAlign, copyLength * pcm.BlockAlign);
			if (dataInSamples)
				Buffer.BlockCopy(src.Samples, srcOffset * pcm.ChannelCount * 4, Samples, dstOffset * pcm.ChannelCount * 4, copyLength * pcm.ChannelCount * 4);
			if (dataInFloat)
				Buffer.BlockCopy(src.Float, srcOffset * pcm.ChannelCount * 4, Float, dstOffset * pcm.ChannelCount * 4, copyLength * pcm.ChannelCount * 4);
		}

		public unsafe void Prepare(AudioBuffer _src, int _offset, int _length)
		{
			length = Math.Min(size, _src.Length - _offset);
			if (_length >= 0)
				length = Math.Min(length, _length);
			dataInBytes = false;
			dataInFloat = false;
			dataInSamples = false;
			if (_src.dataInBytes)
				dataInBytes = true;
			else if (_src.dataInSamples)
				dataInSamples = true;
			else if (_src.dataInFloat)
				dataInFloat = true;
			Load(0, _src, _offset, length);
		}

		public void Swap(AudioBuffer buffer)
		{
			if (pcm.BitsPerSample != buffer.PCM.BitsPerSample || pcm.ChannelCount != buffer.PCM.ChannelCount)
				throw new Exception("AudioBuffer format mismatch");

			int[,] samplesTmp = samples;
			float[,] floatsTmp = fsamples;
			byte[] bytesTmp = bytes;

			fsamples = buffer.fsamples;
			samples = buffer.samples;
			bytes = buffer.bytes;
			length = buffer.length;
			size = buffer.size;
			dataInSamples = buffer.dataInSamples;
			dataInBytes = buffer.dataInBytes;
			dataInFloat = buffer.dataInFloat;

			buffer.samples = samplesTmp;
			buffer.bytes = bytesTmp;
			buffer.fsamples = floatsTmp;
			buffer.length = 0;
			buffer.dataInSamples = false;
			buffer.dataInBytes = false;
			buffer.dataInFloat = false;
		}

		unsafe public void Interlace(int pos, int* src1, int* src2, int n)
		{
			if (PCM.ChannelCount != 2 || PCM.BitsPerSample != 16)
				throw new Exception("");
			fixed (byte* bs = Bytes)
			{
				int* res = ((int*)bs) + pos;
				for (int i = n; i > 0; i--)
					*(res++) = (*(src1++) & 0xffff) ^ (*(src2++) << 16);
			}
		}

		//public void Clear()
		//{
		//    length = 0;
		//}

		public static unsafe void FLACSamplesToBytes_16(int[,] inSamples, int inSampleOffset,
			byte* outSamples, int sampleCount, int channelCount)
		{
			int loopCount = sampleCount * channelCount;

			if (inSamples.GetLength(0) - inSampleOffset < sampleCount)
				throw new IndexOutOfRangeException();

			fixed (int* pInSamplesFixed = &inSamples[inSampleOffset, 0])
			{
				int* pInSamples = pInSamplesFixed;
				short* pOutSamples = (short*)outSamples;
				for (int i = 0; i < loopCount; i++)
					pOutSamples[i] = (short)pInSamples[i];
			}
		}

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

		public static unsafe void FloatToBytes_16(float[,] inSamples, int inSampleOffset,
			byte[] outSamples, int outByteOffset, int sampleCount, int channelCount)
		{
			int loopCount = sampleCount * channelCount;

			if ((inSamples.GetLength(0) - inSampleOffset < sampleCount) ||
				(outSamples.Length - outByteOffset < loopCount * 2))
			{
				throw new IndexOutOfRangeException();
			}

			fixed (float* pInSamplesFixed = &inSamples[inSampleOffset, 0])
			{
				fixed (byte* pOutSamplesFixed = &outSamples[outByteOffset])
				{
					float* pInSamples = pInSamplesFixed;
					short* pOutSamples = (short*)pOutSamplesFixed;

					for (int i = 0; i < loopCount; i++)
					{
						*(pOutSamples++) = (short)(32758*(*(pInSamples++)));
					}
				}
			}
		}

		public static unsafe void FloatToBytes(float[,] inSamples, int inSampleOffset,
			byte[] outSamples, int outByteOffset, int sampleCount, int channelCount, int bitsPerSample)
		{
			if (bitsPerSample == 16)
				FloatToBytes_16(inSamples, inSampleOffset, outSamples, outByteOffset, sampleCount, channelCount);
			//else if (bitsPerSample > 16 && bitsPerSample <= 24)
			//    FLACSamplesToBytes_24(inSamples, inSampleOffset, outSamples, outByteOffset, sampleCount, channelCount, 24 - bitsPerSample);
			else if (bitsPerSample == 32)
				Buffer.BlockCopy(inSamples, inSampleOffset * 4 * channelCount, outSamples, outByteOffset, sampleCount * 4 * channelCount);
			else
				throw new Exception("Unsupported bitsPerSample value");
		}

		public static unsafe void FLACSamplesToBytes(int[,] inSamples, int inSampleOffset,
			byte[] outSamples, int outByteOffset, int sampleCount, int channelCount, int bitsPerSample)
		{
			if (bitsPerSample == 16)
				FLACSamplesToBytes_16(inSamples, inSampleOffset, outSamples, outByteOffset, sampleCount, channelCount);
			else if (bitsPerSample > 16 && bitsPerSample <= 24)
				FLACSamplesToBytes_24(inSamples, inSampleOffset, outSamples, outByteOffset, sampleCount, channelCount, 24 - bitsPerSample);
			else
				throw new Exception("Unsupported bitsPerSample value");
		}

		public static unsafe void FLACSamplesToBytes(int[,] inSamples, int inSampleOffset,
			byte* outSamples, int sampleCount, int channelCount, int bitsPerSample)
		{
			if (bitsPerSample == 16)
				FLACSamplesToBytes_16(inSamples, inSampleOffset, outSamples, sampleCount, channelCount);
			else
				throw new Exception("Unsupported bitsPerSample value");
		}

		public static unsafe void Bytes16ToFloat(byte[] inSamples, int inByteOffset,
			float[,] outSamples, int outSampleOffset, int sampleCount, int channelCount)
		{
			int loopCount = sampleCount * channelCount;

			if ((inSamples.Length - inByteOffset < loopCount * 2) ||
				(outSamples.GetLength(0) - outSampleOffset < sampleCount))
				throw new IndexOutOfRangeException();

			fixed (byte* pInSamplesFixed = &inSamples[inByteOffset])
			{
				fixed (float* pOutSamplesFixed = &outSamples[outSampleOffset, 0])
				{
					short* pInSamples = (short*)pInSamplesFixed;
					float* pOutSamples = pOutSamplesFixed;
					for (int i = 0; i < loopCount; i++)
						*(pOutSamples++) = *(pInSamples++) / 32768.0f;
				}
			}
		}

		public static unsafe void BytesToFLACSamples_16(byte[] inSamples, int inByteOffset,
			int[,] outSamples, int outSampleOffset, int sampleCount, int channelCount)
		{
			int loopCount = sampleCount * channelCount;

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
			int[,] outSamples, int outSampleOffset, int sampleCount, int channelCount, int wastedBits)
		{
			int loopCount = sampleCount * channelCount;

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
			int[,] outSamples, int outSampleOffset, int sampleCount, int channelCount, int bitsPerSample)
		{
			if (bitsPerSample == 16)
				BytesToFLACSamples_16(inSamples, inByteOffset, outSamples, outSampleOffset, sampleCount, channelCount);
			else if (bitsPerSample > 16 && bitsPerSample <= 24)
				BytesToFLACSamples_24(inSamples, inByteOffset, outSamples, outSampleOffset, sampleCount, channelCount, 24 - bitsPerSample);
			else
				throw new Exception("Unsupported bitsPerSample value");
		}
	}

	public class AudioSamples
	{
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

		unsafe public static void MemCpy(long* res, long* smp, int n)
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
			if ((((IntPtr)smp).ToInt64() & 7) == (((IntPtr)res).ToInt64() & 7) && n > 32)
			{
				int delta = (int)((8 - (((IntPtr)smp).ToInt64() & 7)) & 7);
				for (int i = delta; i > 0; i--)
					*(res++) = *(smp++);
				n -= delta;

				MemCpy((long*)res, (long*)smp, n >> 3);
				int n8 = (n >> 3) << 3;
				n -= n8;
				smp += n8;
				res += n8;
			}
			if ((((IntPtr)smp).ToInt64() & 3) == (((IntPtr)res).ToInt64() & 3) && n > 16)
			{
				int delta = (int)((4 - (((IntPtr)smp).ToInt64() & 3)) & 3);
				for (int i = delta; i > 0; i--)
					*(res++) = *(smp++);				
				n -= delta;

				MemCpy((int*)res, (int*)smp, n >> 2);
				int n4 = (n >> 2) << 2;
				n -= n4;
				smp += n4;
				res += n4;
			}
			for (int i = n; i > 0; i--)
				*(res++) = *(smp++);
		}

		unsafe public static void MemSet(int* res, int smp, int n)
		{
			for (int i = n; i > 0; i--)
				*(res++) = smp;
		}

		unsafe public static void MemSet(long* res, long smp, int n)
		{
			for (int i = n; i > 0; i--)
				*(res++) = smp;
		}

		unsafe public static void MemSet(byte* res, byte smp, int n)
		{
			if (IntPtr.Size == 8 && (((IntPtr)res).ToInt64() & 7) == 0 && smp == 0 && n > 8)
			{
				MemSet((long*)res, 0, n >> 3);
				int n8 = (n >> 3) << 3;
				n -= n8;
				res += n8;
			}
			if ((((IntPtr)res).ToInt64() & 3) == 0 && smp == 0 && n > 4)
			{
				MemSet((int*)res, 0, n >> 2);
				int n4 = (n >> 2) << 2;
				n -= n4;
				res += n4;
			}
			for (int i = n; i > 0; i--)
				*(res++) = smp;
		}

		unsafe public static void MemSet(byte[] res, byte smp, int offs, int n)
		{
			fixed (byte* pres = &res[offs])
				MemSet(pres, smp, n);
		}

		unsafe public static void MemSet(int[] res, int smp, int offs, int n)
		{
			fixed (int* pres = &res[offs])
				MemSet(pres, smp, n);
		}

		unsafe public static void MemSet(long[] res, long smp, int offs, int n)
		{
			fixed (long* pres = &res[offs])
				MemSet(pres, smp, n);
		}

		public const uint UINT32_MAX = 0xffffffff;
	}

	/// <summary>
	/// Represents the interface to a device that can play a WaveFile
	/// </summary>
	public interface IWavePlayer : IDisposable, IAudioDest
	{
		/// <summary>
		/// Begin playback
		/// </summary>
		void Play();

		/// <summary>
		/// Stop playback
		/// </summary>
		void Stop();

		/// <summary>
		/// Pause Playback
		/// </summary>        
		void Pause();

		/// <summary>
		/// Current playback state
		/// </summary>
		PlaybackState PlaybackState { get; }

		/// <summary>
		/// The volume 1.0 is full scale
		/// </summary>
		float Volume { get; set; }

		/// <summary>
		/// Indicates that playback has gone into a stopped state due to 
		/// reaching the end of the input stream
		/// </summary>
		event EventHandler PlaybackStopped;
	}

	/// <summary>
	/// Playback State
	/// </summary>
	public enum PlaybackState
	{
		/// <summary>
		/// Stopped
		/// </summary>
		Stopped,
		/// <summary>
		/// Playing
		/// </summary>
		Playing,
		/// <summary>
		/// Paused
		/// </summary>
		Paused
	}

	public class DummyWriter : IAudioDest
	{
		AudioPCMConfig _pcm;

		public DummyWriter(string path, AudioPCMConfig pcm)
		{
			_pcm = pcm;
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

		public int CompressionLevel
		{
			get { return 0; }
			set { }
		}

		public object Settings 
		{
			get
			{
				return null;
			}
			set
			{
				if (value != null && value.GetType() != typeof(object))
					throw new Exception("Unsupported options " + value);
			}
		}

		public long Padding
		{
			set { }
		}

		public long BlockSize
		{
			set	{ }
		}

		public AudioPCMConfig PCM
		{
			get { return _pcm;  }
		}

		public void Write(AudioBuffer buff)
		{
		}

		public string Path { get { return null; } }
	}

	public class SilenceGenerator : IAudioSource
	{
		private long _sampleOffset, _sampleCount;
		private AudioPCMConfig pcm;
		private int _sampleVal;

		public SilenceGenerator(long sampleCount, int sampleVal)
		{
			_sampleVal = sampleVal;
			_sampleOffset = 0;
			_sampleCount = sampleCount;
			pcm = AudioPCMConfig.RedBook;
		}

		public SilenceGenerator(long sampleCount)
			: this(sampleCount, 0)
		{
		}

		public long Length
		{
			get
			{
				return _sampleCount;
			}
		}

		public long Remaining
		{
			get
			{
				return _sampleCount - _sampleOffset;
			}
		}

		public long Position
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

		public AudioPCMConfig PCM { get { return pcm; } }

		public int Read(AudioBuffer buff, int maxLength)
		{
			buff.Prepare(this, maxLength);

			int[,] samples = buff.Samples;
			for (int i = 0; i < buff.Length; i++)
				for (int j = 0; j < PCM.ChannelCount; j++)
					samples[i, j] = _sampleVal;

			_sampleOffset += buff.Length;
			return buff.Length;
		}

		public void Close()
		{
		}

		public string Path { get { return null; } }
	}

	[AudioDecoderClass("builtin wav", "wav")]
	public class WAVReader : IAudioSource
	{
		Stream _IO;
		BinaryReader _br;
		long _dataOffset, _samplePos, _sampleLen;
		private AudioPCMConfig pcm;
		long _dataLen;
		bool _largeFile;
		string _path;

		public WAVReader(string path, Stream IO)
		{
			_path = path;
			_IO = IO != null ? IO : new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 0x10000, FileOptions.SequentialScan);
			_br = new BinaryReader(_IO);

			ParseHeaders();

			if (_dataLen < 0)
				_sampleLen = -1;
			else
				_sampleLen = _dataLen / pcm.BlockAlign;
		}

		public WAVReader(string path, Stream IO, AudioPCMConfig _pcm)
		{
			_path = path;
			_IO = IO != null ? IO : new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 0x10000, FileOptions.SequentialScan);
			_br = new BinaryReader(_IO);

			_largeFile = false;
			_dataOffset = 0;
			_samplePos = 0;
			pcm = _pcm;
			_dataLen = _IO.CanSeek ? _IO.Length : -1;
			if (_dataLen < 0)
				_sampleLen = -1;
			else
			{
				_sampleLen = _dataLen / pcm.BlockAlign;
				if ((_dataLen % pcm.BlockAlign) != 0)
					throw new Exception("odd file size");
			}
		}

		public static AudioBuffer ReadAllSamples(string path, Stream IO)
		{
			WAVReader reader = new WAVReader(path, IO);
			AudioBuffer buff = new AudioBuffer(reader, (int)reader.Length);
			reader.Read(buff, -1);
			if (reader.Remaining != 0)
				throw new Exception("couldn't read the whole file");
			reader.Close();
			return buff;
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
					int _channelCount = _br.ReadInt16();
					int _sampleRate = _br.ReadInt32();
					_br.ReadInt32();
					int _blockAlign = _br.ReadInt16();
					int _bitsPerSample = _br.ReadInt16();
					pcm = new AudioPCMConfig(_bitsPerSample, _channelCount, _sampleRate);
					if (pcm.BlockAlign != _blockAlign)
						throw new Exception("WAVE has strange BlockAlign");
					pos += 16;
				}
				else if (ckID == fccData)
				{
					foundData = true;

					_dataOffset = pos;
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

			if ((foundFormat & foundData) == false || pcm == null)
				throw new Exception("Format or data chunk not found.");
			if (pcm.ChannelCount <= 0)
				throw new Exception("Channel count is invalid.");
			if (pcm.SampleRate <= 0)
				throw new Exception("Sample rate is invalid.");
			if ((pcm.BitsPerSample <= 0) || (pcm.BitsPerSample > 32))
				throw new Exception("Bits per sample is invalid.");
			if (pos != _dataOffset)
				Position = 0;
		}

		public long Position
		{
			get
			{
				return _samplePos;
			}
			set
			{
				long seekPos;

				if (_sampleLen >= 0 && value > _sampleLen)
					_samplePos = _sampleLen;
				else
					_samplePos = value;

				seekPos = _dataOffset + _samplePos * PCM.BlockAlign;
				_IO.Seek(seekPos, SeekOrigin.Begin);
			}
		}

		public long Length
		{
			get
			{
				return _sampleLen;
			}
		}

		public long Remaining
		{
			get
			{
				return _sampleLen - _samplePos;
			}
		}

		public AudioPCMConfig PCM { get { return pcm; } }

		public int Read(AudioBuffer buff, int maxLength)
		{
			buff.Prepare(this, maxLength);

			byte[] bytes = buff.Bytes;
			int byteCount = (int)buff.ByteLength;
			int pos = 0;

			while (pos < byteCount)
			{
				int len = _IO.Read(bytes, pos, byteCount - pos);
				if (len <= 0)
				{
					if ((pos % PCM.BlockAlign) != 0 || _sampleLen >= 0)
						throw new Exception("Incomplete file read.");
					buff.Length = pos / PCM.BlockAlign;
					_samplePos += buff.Length;
					_sampleLen = _samplePos;
					return buff.Length;
				}
				pos += len;
			}
			_samplePos += buff.Length;
			return buff.Length;
		}

		public string Path { get { return _path; } }
	}

	[AudioEncoderClass("builtin wav", "wav", true, "", "", 10, typeof(object))]
	public class WAVWriter : IAudioDest
	{
		Stream _IO;
		BinaryWriter _bw;
		AudioPCMConfig _pcm;
		long _sampleLen;
		string _path;
		long hdrLen = 0;
		bool _headersWritten = false;
		long _finalSampleCount = -1;
		List<byte[]> _chunks = null;
		List<uint> _chunkFCCs = null;

		public WAVWriter(string path, Stream IO, AudioPCMConfig pcm)
		{
			_pcm = pcm;
			_path = path;
			_IO = IO != null ? IO : new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
			_bw = new BinaryWriter(_IO);
		}

		public WAVWriter(string path, AudioPCMConfig pcm)
			: this(path, null, pcm)
		{
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

			bool wavex = _pcm.BitsPerSample != 16 && _pcm.BitsPerSample != 24;

			hdrLen += 36 + (wavex ? 24 : 0) + 8;

			uint dataLen = (uint)(_finalSampleCount * _pcm.BlockAlign);
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
			_bw.Write((ushort)_pcm.ChannelCount);
			_bw.Write((uint)_pcm.SampleRate);
			_bw.Write((uint)(_pcm.SampleRate * _pcm.BlockAlign));
			_bw.Write((ushort)_pcm.BlockAlign);
			_bw.Write((ushort)((_pcm.BitsPerSample+7)/8*8));
			if (wavex)
			{
				_bw.Write((ushort)22); // length of WAVEX structure
				_bw.Write((ushort)_pcm.BitsPerSample);
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
			if (_finalSampleCount <= 0)
			{
				const long maxFileSize = 0x7FFFFFFEL;
				long dataLen = _sampleLen * _pcm.BlockAlign;
				if ((dataLen & 1) == 1)
					_bw.Write((byte)0);
				if (dataLen + hdrLen > maxFileSize)
					dataLen = ((maxFileSize - hdrLen) / _pcm.BlockAlign) * _pcm.BlockAlign;
				long dataLenPadded = dataLen + (dataLen & 1);

				_bw.Seek(4, SeekOrigin.Begin);
				_bw.Write((uint)(dataLenPadded + hdrLen - 8));

				_bw.Seek((int)hdrLen - 4, SeekOrigin.Begin);
				_bw.Write((uint)dataLen);
			}

			_bw.Close();

			_bw = null;
			_IO = null;

			if (_finalSampleCount > 0 && _sampleLen != _finalSampleCount)
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

		public int CompressionLevel
		{
			get { return 0; }
			set { }
		}

		public object Settings
		{
			get
			{
				return null;
			}
			set
			{
				if (value != null && value.GetType() != typeof(object))
					throw new Exception("Unsupported options " + value);
			}
		}

		public long Padding
		{
			set { }
		}

		public AudioPCMConfig PCM
		{
			get { return _pcm; }
		}

		public void Write(AudioBuffer buff)
		{
			if (buff.Length == 0)
				return;
			buff.Prepare(this);
			if (!_headersWritten)
				WriteHeaders();
			_IO.Write(buff.Bytes, 0, buff.ByteLength);
			_sampleLen += buff.Length;
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
		private Exception _ex = null;

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
				while (FreeSpace == 0 && _ex == null)
					Monitor.Wait(this);
				if (_ex != null)
					throw _ex;
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
					while (FreeSpace == 0 && _ex == null)
						Monitor.Wait(this);
					if (_ex != null)
						throw _ex;
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

		public void Write(byte[] buff, int offs, int count)
		{
			while (count > 0)
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
					Array.Copy(_buffer, pos, buff, offs, chunk);
				offs += chunk;
				lock (this)
				{
					_start += chunk;
					Monitor.Pulse(this);
				}
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
					try
					{
						flushOutput(_buffer, pos, chunk, to);
					}
					catch (Exception ex)
					{
						lock (this)
						{
							_ex = ex;
							Monitor.Pulse(this);
							return;
						}
					}
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

	public class CycilcBufferInputStream : Stream
	{
		CyclicBuffer _buffer;

		public CycilcBufferInputStream(CyclicBuffer buffer)
		{
			_buffer = buffer;
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
			_buffer.Write(array, offset, count);
			return count;
		}

		public override long Seek(long offset, SeekOrigin origin)
		{
			throw new NotSupportedException();
		}

		public override void Write(byte[] array, int offset, int count)
		{
			throw new NotSupportedException();
		}
	}

	public class UserDefinedWriter : IAudioDest
	{
		string _path, _encoder, _encoderParams, _encoderMode;
		Process _encoderProcess;
		WAVWriter wrt;
		CyclicBuffer outputBuffer = null;
		bool useTempFile = false;
		string tempFile = null;
		long _finalSampleCount = -1;
		bool closed = false;

		public UserDefinedWriter(string path, Stream IO, AudioPCMConfig pcm, string encoder, string encoderParams, string encoderMode, int padding)
		{
			_path = path;
			_encoder = encoder;
			_encoderParams = encoderParams;
			_encoderMode = encoderMode;
			useTempFile = _encoderParams.Contains("%I");
			tempFile = path + ".tmp.wav";

			_encoderProcess = new Process();
			_encoderProcess.StartInfo.FileName = _encoder;
			_encoderProcess.StartInfo.Arguments = _encoderParams.Replace("%O", "\"" + path + "\"").Replace("%M", encoderMode).Replace("%P", padding.ToString()).Replace("%I", "\"" + tempFile + "\"");
			_encoderProcess.StartInfo.CreateNoWindow = true;
			if (!useTempFile)
				_encoderProcess.StartInfo.RedirectStandardInput = true;
			_encoderProcess.StartInfo.UseShellExecute = false;
			if (!_encoderParams.Contains("%O"))
				_encoderProcess.StartInfo.RedirectStandardOutput = true;
			if (useTempFile)
			{
				wrt = new WAVWriter(tempFile, null, pcm);
				return;
			}
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
			wrt = new WAVWriter(path, inputStream, pcm);
		}

		public void Close()
		{
			if (closed)
				return;
			closed = true;
			wrt.Close();
			if (useTempFile && (_finalSampleCount < 0 || wrt.Position == _finalSampleCount))
			{
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
			}
			wrt = null;
			if (!_encoderProcess.HasExited)
				_encoderProcess.WaitForExit();
			if (useTempFile)
				File.Delete(tempFile);
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
			set { _finalSampleCount = wrt.FinalSampleCount = value; }
		}

		public long BlockSize
		{
			set { }
		}

		public int CompressionLevel
		{
			get { return 0; }
			set { } // !!!! Must not start the process in constructor, so that we can set CompressionLevel!
		}

		public object Settings
		{
			get
			{
				return null;
			}
			set
			{
				if (value != null && value.GetType() != typeof(object))
					throw new Exception("Unsupported options " + value);
			}
		}

		public long Padding
		{
			set { }
		}

		public AudioPCMConfig PCM
		{
			get { return wrt.PCM; }
		}

		public void Write(AudioBuffer buff)
		{
			try
			{
				wrt.Write(buff);
			}
			catch (IOException ex)
			{
				if (_encoderProcess.HasExited)
					throw new IOException(string.Format("{0} has exited prematurely with code {1}", _encoder, _encoderProcess.ExitCode), ex);
				else
					throw ex;
			}
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

		public long Position
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

		public long Length
		{
			get
			{
				Initialize();
				return rdr.Length;
			}
		}

		public long Remaining
		{
			get
			{
				Initialize();
				return rdr.Remaining;
			}
		}

		public AudioPCMConfig PCM
		{
			get
			{
				Initialize();
				return rdr.PCM;
			}
		}

		public int Read(AudioBuffer buff, int maxLength)
		{
			Initialize();
			return rdr.Read(buff, maxLength);
		}

		public string Path { get { return _path; } }
	}

	public class AudioPipe : IAudioSource//, IDisposable
	{
		private AudioBuffer _readBuffer, _writeBuffer;
		private AudioPCMConfig pcm;
		long _sampleLen, _samplePos;
		private int _maxLength;
		private Thread _workThread;
		IAudioSource _source;
		bool _close = false;
		bool _haveData = false;
		int _bufferPos = 0;
		Exception _ex = null;
		bool own;
		ThreadPriority priority;

		public AudioPipe(AudioPCMConfig pcm, int size)
		{
			this.pcm = pcm;
			_readBuffer = new AudioBuffer(pcm, size);
			_writeBuffer = new AudioBuffer(pcm, size);
			_maxLength = size;
			_sampleLen = -1;
			_samplePos = 0;
		}

		public AudioPipe(IAudioSource source, int size, bool own, ThreadPriority priority)
			: this(source.PCM, size)
		{
			this.own = own;
			this.priority = priority;
			_source = source;
			_sampleLen = _source.Length;
			_samplePos = _source.Position;
		}

		public AudioPipe(IAudioSource source, int size)
			: this(source, size, true, ThreadPriority.BelowNormal)
		{
		}

		private void Decompress(object o)
		{
#if !DEBUG
			try
#endif
			{
				bool done = false;
				do
				{
					done = _source.Read(_writeBuffer, -1) == 0;
					lock (this)
					{
						while (_haveData && !_close)
							Monitor.Wait(this);
						if (_close)
							break;
						AudioBuffer temp = _writeBuffer;
						_writeBuffer = _readBuffer;
						_readBuffer = temp;
						_haveData = true;
						Monitor.Pulse(this);
					}
				} while (!done);
			}
#if !DEBUG
			catch (Exception ex)
			{
				lock (this)
				{
					_ex = ex;
					Monitor.Pulse(this);
				}
			}
#endif
		}

		private void Go()
		{
			if (_workThread != null || _ex != null || _source == null) return;
			_workThread = new Thread(Decompress);
			_workThread.Priority = priority;
			_workThread.IsBackground = true;
			_workThread.Name = "AudioPipe";
			_workThread.Start(null);
		}

		//public new void Dispose()
		//{
		//    _buffer.Clear();
		//}

		public void Close()
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
			if (_source != null)
			{
				if (own) _source.Close();
				_source = null;
			}
			if (_readBuffer != null)
			{
				//_readBuffer.Clear();
				_readBuffer = null;
			}
			if (_writeBuffer != null)
			{
				//_writeBuffer.Clear();
				_writeBuffer = null;
			}
		}

		public long Position
		{
			get
			{
				return _samplePos;
			}
			set
			{
				if (value == _samplePos)
					return;

				if (_source == null)
					throw new NotSupportedException();

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
				_source.Position = value;
				_samplePos = value;
				_bufferPos = 0;
				_haveData = false;
				_close = false;
				//Go();
				//throw new Exception("not supported");
			}
		}

		public long Length
		{
			get
			{
				return _sampleLen;
			}
		}

		public long Remaining
		{
			get
			{
				return _sampleLen - _samplePos;
			}
		}

		public AudioPCMConfig PCM
		{
			get
			{
				return pcm;
			}
		}

		public int Write(AudioBuffer buff)
		{
			if (_writeBuffer.Size < _writeBuffer.Length + buff.Length)
			{
				AudioBuffer realloced = new AudioBuffer(pcm, _writeBuffer.Size + buff.Size);
				realloced.Prepare(_writeBuffer, 0, _writeBuffer.Length);
				_writeBuffer = realloced;
			}
			if (_writeBuffer.Length == 0)
				_writeBuffer.Prepare(buff, 0, buff.Length);
			else
			{
				_writeBuffer.Load(_writeBuffer.Length, buff, 0, buff.Length);
				_writeBuffer.Length += buff.Length;
			}
			lock (this)
			{
				if (!_haveData)
				{
					AudioBuffer temp = _writeBuffer;
					_writeBuffer = _readBuffer;
					_writeBuffer.Length = 0;
					_readBuffer = temp;
					_haveData = true;
					Monitor.Pulse(this);
				}
			}
			return _writeBuffer.Length;
		}

		public int Read(AudioBuffer buff, int maxLength)
		{
			Go();

			bool needToCopy = false;
			if (_bufferPos != 0)
				needToCopy = true;
			else
				lock (this)
				{
					while (!_haveData && _ex == null)
						Monitor.Wait(this);
					if (_ex != null)
						throw _ex;
					if (_bufferPos == 0 && (maxLength < 0 || _readBuffer.Length <= maxLength))
					{
						buff.Swap(_readBuffer);
						_haveData = false;
						Monitor.Pulse(this);
					}
					else
						needToCopy = true;
				}
			if (needToCopy)
			{
				buff.Prepare(_readBuffer, _bufferPos, maxLength);
				_bufferPos += buff.Length;
				if (_bufferPos == _readBuffer.Length)
				{
					_bufferPos = 0;
					lock (this)
					{
						_haveData = false;
						Monitor.Pulse(this);
					}
				}
			}
			_samplePos += buff.Length;
			return buff.Length;
		}

		public string Path
		{
			get
			{
				if (_source == null)
					return "";
				return _source.Path;
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
