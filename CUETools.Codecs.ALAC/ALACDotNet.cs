using System;
using System.Text;
using System.IO;
using System.Collections.Generic;
using System.Collections.Specialized;
using CUETools.Codecs;

//Copyright (c) 2008 Gregory S. Chudov.
//This library is based on ALAC decoder by David Hammerton.
//See http://crazney.net/programs/itunes/alac.html for details.
//Copyright (c) 2004 David Hammerton.
//Permission is hereby granted, free of charge, to any person obtaining a copy 
// of this software and associated documentation files (the "Software"), to deal 
// in the Software without restriction, including without limitation the rights 
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
// copies of the Software, and to permit persons to whom the Software is furnished 
// to do so, subject to the following conditions:
//The above copyright notice and this permission notice shall be included in all 
//copies or substantial portions of the Software.
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, 
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES 
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. 
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, 
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, 
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE 
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

namespace CUETools.Codecs.ALAC
{
	[AudioDecoderClass("builtin alac", "m4a")]
	public class ALACReader : IAudioSource
	{
		public ALACReader(string path, Stream IO)
		{
			_path = path;
			_IO = IO != null ? IO : new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
			_buff = new byte[512];
			_tags = new NameValueCollection();
			qtmovie_read();
			if (!_formatRead || pcm.BitsPerSample != 16 || pcm.ChannelCount != 2 || pcm.SampleRate != 44100)
				throw new Exception("Invalid ALAC file.");
			_saved_mdat_pos = _IO.Position;
			calculate_length();
		}

		public ALACReader(AudioPCMConfig _pcm, int rice_historymult, int rice_initialhistory, int rice_kmodifier, int blocksize)
		{
			pcm = _pcm;

			setinfo_max_samples_per_frame = blocksize;
			setinfo_rice_historymult = (byte)rice_historymult;
			setinfo_rice_initialhistory = (byte)rice_initialhistory;
			setinfo_rice_kmodifier = (byte)rice_kmodifier;

			_predicterror_buffer_a = new int[setinfo_max_samples_per_frame];
			_predicterror_buffer_b = new int[setinfo_max_samples_per_frame];
			_outputsamples_buffer_a = new int[setinfo_max_samples_per_frame];
			_outputsamples_buffer_b = new int[setinfo_max_samples_per_frame];
			_framesBuffer = new byte[65536];
		}

		private void InitTables()
		{
			if (_predicterror_buffer_a != null)
				return;

			setinfo_max_samples_per_frame = (int)read_uint32(_codecData, 24);
			byte setinfo_7a = read_uint8(_codecData, 28);
			byte setinfo_sample_size = read_uint8(_codecData, 29);
			setinfo_rice_historymult = read_uint8(_codecData, 30);
			setinfo_rice_initialhistory = read_uint8(_codecData, 31);
			setinfo_rice_kmodifier = read_uint8(_codecData, 32);
			byte setinfo_7f = read_uint8(_codecData, 33);
			ushort setinfo_80 = read_uint16(_codecData, 34);
			uint setinfo_82 = read_uint32(_codecData, 36); // maxframesize
			uint setinfo_86 = read_uint32(_codecData, 40); // bitrate
			uint setinfo_8a_rate = read_uint32(_codecData, 44); // samplerate

			_predicterror_buffer_a = new int[setinfo_max_samples_per_frame];
			_predicterror_buffer_b = new int[setinfo_max_samples_per_frame];
			_outputsamples_buffer_a = new int[setinfo_max_samples_per_frame];
			_outputsamples_buffer_b = new int[setinfo_max_samples_per_frame];

			_samplesInBuffer = 0;
			_framesBuffer = new byte[65536];
		}

		public int Read(AudioBuffer buff, int maxLength)
		{
			InitTables();

			buff.Prepare(this, maxLength);

			int offset = 0;
			int sampleCount = buff.Length;

			while (_samplesInBuffer < sampleCount)
			{
				if (_samplesInBuffer > 0)
				{
					deinterlace(buff.Samples, offset, _samplesInBuffer);
					sampleCount -= _samplesInBuffer;
					offset += _samplesInBuffer;
					_samplesInBuffer = 0;
					_samplesBufferOffset = 0;
				}

				int sampleDuration;
				int sampleSize;
				if (_iSample >= _sample_byte_size.Length)
					return offset;
				get_sample_info(_iSample, out sampleDuration, out sampleSize);
				_IO.Read(_framesBuffer, 0, sampleSize);
				decodeFrame(sampleSize);
				if (sampleDuration != _samplesInBuffer)
					throw new Exception("sample count mismatch");
				_samplesInBuffer -= _samplesBufferOffset;
				_sampleOffset += _samplesInBuffer;
				_iSample++;
			}

			deinterlace(buff.Samples, offset, sampleCount);
			_samplesInBuffer -= sampleCount;
			_samplesBufferOffset += sampleCount;
			if (_samplesInBuffer == 0)
				_samplesBufferOffset = 0;
			return offset + sampleCount;
		}

		public void Close()
		{
			_IO.Close();
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
				return _sampleCount - _sampleOffset + _samplesInBuffer;
			}
		}

		public long Position
		{
			get
			{
				return _sampleOffset - _samplesInBuffer;
			}
			set
			{
				_sampleOffset = value;
				_samplesInBuffer = 0;
				_samplesBufferOffset = 0;
				
				_iSample = 0;
				long durOffs = 0;
				int sampleDuration;
				long fileOffs = 0;
				int sampleSize;
				do
				{
					if (durOffs == value)
					{
						_IO.Position = _saved_mdat_pos + fileOffs;
						return;
					}
					if ((int)_iSample >= _sample_byte_size.Length)
						throw new Exception("seeking past end of stream");
					get_sample_info(_iSample, out sampleDuration, out sampleSize);
					durOffs += sampleDuration;
					fileOffs += sampleSize;
					_iSample++;
				} while (durOffs <= value);
				_IO.Position = _saved_mdat_pos + fileOffs - sampleSize;
				_samplesBufferOffset = (int) (value + sampleDuration - durOffs);
				_iSample--;
			}
		}

		public AudioPCMConfig PCM { get { return pcm; } }

		public string Path 
		{
			get 
			{ 
				return _path; 
			} 
		}

		private void get_sample_info(long iSample, out int sampleDuration, out int sampleSize)
		{
			// if (iSample >= _sample_byte_size.Length)
			int duration_index_accum = 0;
			int duration_cur_index = 0;
			while (_time_to_sample_count[duration_cur_index] + duration_index_accum <= iSample)
			{
				duration_index_accum += _time_to_sample_count[duration_cur_index];
				if (duration_cur_index == _time_to_sample_count.Length - 1)
					throw new Exception("seeking past end of stream");
				duration_cur_index ++;
			}
			sampleDuration = _time_to_sample_duration[duration_cur_index];
			sampleSize = _sample_byte_size[iSample];
		}

		private void calculate_length()
		{
			_sampleCount = 0;
			uint duration_cur_index = 0;
			for (duration_cur_index = 0; duration_cur_index < _time_to_sample_count.Length; duration_cur_index++)
				_sampleCount += _time_to_sample_count[duration_cur_index] * _time_to_sample_duration[duration_cur_index];
			// try a work around for ffdshow-generated buggy files
			if (_time_to_sample_count.Length == 1 && _IO.CanSeek)
			{
				int sample_count_0 = _time_to_sample_count[0] - 1;
				int sample_duration_0 = _time_to_sample_duration[0];
				Position = sample_count_0 * sample_duration_0;
				int sampleDuration;
				int sampleSize;
				if ((int)_iSample < _sample_byte_size.Length)
				{
					get_sample_info(_iSample, out sampleDuration, out sampleSize);
					InitTables();
					_IO.Read(_framesBuffer, 0, (int)sampleSize);
					decodeFrame(sampleSize);
					if (_samplesInBuffer < sampleDuration)
					{
						_time_to_sample_duration = new int[2] { sample_duration_0, _samplesInBuffer };
						_time_to_sample_count = new int[2] { sample_count_0, 1 };
						_sampleCount -= sampleDuration - _samplesInBuffer;
					}
				}
				Position = 0;
			}
		}

		private byte [] stream_read_bytes(int len)
		{
			if (len > 512)
				throw new Exception("Decoding failed.");
			if (_IO.Read(_buff, 0, len) != len)
				throw new Exception("Decoding failed.");
			return _buff;
		}

		private uint read_uint32(byte [] buff, int pos)
		{
			return (uint)((buff[pos] << 24) + (buff[pos+1] << 16) + (buff[pos+2] << 8) + (buff[pos+3] << 0));
		}

		private uint stream_read_uint32()
		{
			return read_uint32(stream_read_bytes(4), 0);
		}

		private ushort read_uint16(byte[] buff, int pos)
		{
			return (ushort)((buff[pos] << 8) + buff[pos + 1]);
		}

		private ushort stream_read_uint16()
		{
			return read_uint16(stream_read_bytes(2), 0);
		}

		private byte read_uint8(byte[] buff, int pos)
		{
			return buff[pos];
		}

		private byte stream_read_uint8()
		{
			return stream_read_bytes(1)[0];
		}

		private void stream_skip (UInt32 skip)
		{
			_IO.Position += skip;
		}

		/* supports reading 1 to 24 bits, in big endian format */
		private uint readbits_24 (byte []buff, ref int pos, int bits)
		{
			uint result = (((uint)buff[pos]) << 24) | (((uint)buff[pos + 1]) << 16) | (((uint)buff[pos + 2]) << 8) | ((uint)buff[pos + 3]);
			result <<= _bitaccumulator;
			result >>= 32 - bits;

			int new_accumulator = (_bitaccumulator + bits);
			pos += (new_accumulator >> 3);
			_bitaccumulator = (new_accumulator & 7);
			return result;
		}

		/* supports reading 1 to 24 bits, in big endian format */
		private unsafe uint readbits_24(byte* buff, ref int pos, int bits)
		{
			uint result = (((uint)buff[pos]) << 24) | (((uint)buff[pos + 1]) << 16) | (((uint)buff[pos + 2]) << 8) | ((uint)buff[pos + 3]);
			result <<= _bitaccumulator;
			result >>= 32 - bits;

			int new_accumulator = (_bitaccumulator + bits);
			pos += (new_accumulator >> 3);
			_bitaccumulator = (new_accumulator & 7);
			return result;
		}

		/* supports reading 1 to 16 bits, in big endian format */
		private unsafe uint peekbits_9(byte* buff, int pos)
		{
			uint result = (((uint)buff[pos]) << 8) | (((uint)buff[pos + 1]));
			result <<= _bitaccumulator;
			result &= 0x0000ffff;
			result >>= 7;
			return result;
		}

		/* supports reading 1 to 16 bits, in big endian format */
		private void skipbits(ref int pos, int bits)
		{
			int new_accumulator = (_bitaccumulator + bits);
			pos += (new_accumulator >> 3);
			_bitaccumulator = (new_accumulator & 7);
		}

		/* supports reading 1 to 32 bits, in big endian format */
		private uint readbits(byte[] buff, ref int pos, int bits)
		{
			if (bits <= 24)
				return readbits_24(buff, ref pos, bits);

			ulong result = (((ulong)buff[pos]) << 32) | (((ulong)buff[pos + 1]) << 24) | (((ulong)buff[pos + 2]) << 16) | (((ulong)buff[pos + 3]) << 8) | ((ulong)buff[pos + 4]);
			result <<= _bitaccumulator;
			result &= 0x00ffffffffff;
			result >>= 40 - bits;
			int new_accumulator = (_bitaccumulator + bits);
			pos += (new_accumulator >> 3);
			_bitaccumulator = (new_accumulator & 7);
			return (uint)result;
		}

		/* supports reading 1 to 32 bits, in big endian format */
		private unsafe uint readbits(byte * buff, ref int pos, int bits)
		{
			if (bits <= 24)
				return readbits_24(buff, ref pos, bits);

			ulong result = (((ulong)buff[pos]) << 32) | (((ulong)buff[pos + 1]) << 24) | (((ulong)buff[pos + 2]) << 16) | (((ulong)buff[pos + 3]) << 8) | ((ulong)buff[pos + 4]);
			result <<= _bitaccumulator;
			result &= 0x00ffffffffff;
			result >>= 40 - bits;
			int new_accumulator = (_bitaccumulator + bits);
			pos += (new_accumulator >> 3);
			_bitaccumulator = (new_accumulator & 7);
			return (uint)result;
		}

		/* reads a single bit */
		private uint readbit(byte[] buff, ref int pos)
		{
			int new_accumulator;
			uint result = buff[pos];
			result <<= _bitaccumulator;
			result = result >> 7 & 1;
			new_accumulator = (_bitaccumulator + 1);
			pos += (new_accumulator / 8);
			_bitaccumulator = (new_accumulator % 8);
			return result;
		}

		private void unreadbits(ref int pos, int bits)
		{
			int new_accumulator = (_bitaccumulator - bits);
			pos += (new_accumulator >> 3);

			_bitaccumulator = (new_accumulator & 7);
			if (_bitaccumulator < 0)
				_bitaccumulator *= -1;
		}

		private static int count_leading_zeroes(uint input)
		{
			int zeroes = 0;
			uint shifted_input = input >> 16;
			if (shifted_input == 0)
				zeroes += 16;
			else
				input = shifted_input;
			shifted_input = input >> 8;
			if (shifted_input == 0)
				zeroes += 8;
			else
				input = shifted_input;
			return zeroes + BitReader.byte_to_unary_table[input];
		}

		private unsafe void readPredictor(ref int pos, ref predictor_t predictor_info)
		{
			fixed (predictor_t* pr = &predictor_info)
			{
				pr->prediction_type = (int)readbits(_framesBuffer, ref pos, 4);
				pr->prediction_quantitization = (int)readbits(_framesBuffer, ref pos, 4);
				pr->ricemodifier = (int)readbits(_framesBuffer, ref pos, 3);
				pr->predictor_coef_num = (int)readbits(_framesBuffer, ref pos, 5);

				/* read the predictor table */
				pr->predictor_coef_table_sum = 0;
				for (int i = pr->predictor_coef_num - 1; i >= 0; i--)
				{
					pr->predictor_coef_table[i] = (short)readbits(_framesBuffer, ref pos, 16);
					pr->predictor_coef_table_sum += pr->predictor_coef_table[i];
				}
			}
		}

		private unsafe int decode_scalar(byte * buff, ref int pos, int k, int limit, int readsamplesize)
		{
			uint next = peekbits_9(buff, pos);
			int x = (next >> 8 == 0) ? 0 : 
				1 + BitReader.byte_to_unary_table[(~next) & 0xff];
			if (x == 9) /* RICE THRESHOLD 9 bits */
			{
				skipbits(ref pos, 9);
				return (int)readbits(buff, ref pos, readsamplesize);
			}
			skipbits(ref pos, x + 1);
			if (k >= limit)
				k = limit;
			if (k == 1)
				return x;
			int extrabits = (int) readbits(buff, ref pos, k);
			x = (x << k) - x; // /* multiply x by 2^k - 1, as part of their strange algorithm */
			if (extrabits > 1)
				return x + extrabits - 1;
			unreadbits(ref pos, 1);
			return x;
		}

		private unsafe void basterdised_rice_decompress(int output_size, ref int pos, ref predictor_t predictor_info, ref int[] predicterror_buffer, int readsamplesize)
		{
			fixed (predictor_t* pr = &predictor_info)
			fixed (int* output_buffer = &predicterror_buffer[0])
			fixed (byte* buff = &_framesBuffer[0])
			{
				uint history = setinfo_rice_initialhistory;
				int rice_kmodifier = setinfo_rice_kmodifier;
				int rice_historymult = pr->ricemodifier * setinfo_rice_historymult / 4;
				int sign_modifier = 0;

				for (int output_count = 0; output_count < output_size; output_count++)
				{
					int x = sign_modifier + decode_scalar(buff, ref pos, 31 - count_leading_zeroes((history >> 9) + 3), rice_kmodifier, readsamplesize);
					output_buffer[output_count] = (x >> 1) ^ - (x & 1);
					sign_modifier = 0;

					/* now update the history */
					history = (uint)(history + (x * rice_historymult)
							 - ((history * rice_historymult) >> 9));

					if (x > 0xffff)
						history = 0xffff;

					/* special case: there may be compressed blocks of 0 */
					if ((history < 128) && (output_count + 1 < output_size))
					{
						int k = 7 - (31 - count_leading_zeroes(history)) + (((int)history + 16) >> 6);
						int block_size = decode_scalar(buff, ref pos, k, rice_kmodifier, 16);
						if (block_size > 0)
						{
							if (output_count + 1 + block_size > output_size)
								throw new Exception("buffer overflow: " + output_size.ToString() + " vs " + (output_count + 1 + block_size).ToString());
								//block_size = (int) output_size - output_count - 1;
							for (int p = 0; p < block_size; p++)
								output_buffer[output_count + 1 + p] = 0;
							output_count += block_size;
						}
						sign_modifier = block_size > 0xffff ? 0 : 1;
						history = 0;
					}
				}
			}
		}

		private static int extend_sign32(int val, int bits)
		{
			return (val << (32 - bits)) >> (32 - bits);
		}

		private static short sign_only(int v)
		{
			return (short)(1 - ((v >> 30) & 2));
		}

		private unsafe void predictor_decompress_fir_adapt(int output_size, ref predictor_t predictor_info, ref int[] error_buffer, ref int[] buffer_out, int readsamplesize)
		{
			int i;

			fixed (predictor_t* pr = &predictor_info)
			fixed (int* buf_out = &buffer_out[0], buf_err = &error_buffer[0])
			{
				if (pr->predictor_coef_num == 0)
				{
					for (i = 0; i < output_size; i++)
						buf_out[i] = buf_err[i];
					return;
				}

				int sample = 0;

				if (pr->predictor_coef_num == 0x1f)
				{ /* 11111 - max value of predictor_coef_num */
					/* second-best case scenario for fir decompression,
					 * error describes a small difference from the previous sample only
					 */
					if (output_size <= 1)
						return;
					for (i = 0; i < output_size; i++)
					{
						sample = extend_sign32(sample + buf_err[i], readsamplesize);
						buf_out[i] = sample;
					}
					return;
				}

				if (output_size <= predictor_info.predictor_coef_num || pr->predictor_coef_num < 0)
					throw new Exception("invalid output size");

				/* read warm-up samples */
				for (i = 0; i <= predictor_info.predictor_coef_num; i++)
				{
					sample = extend_sign32(sample + buf_err[i], readsamplesize);
					buf_out[i] = sample;
				}

				/* general case */
				int* buf_pos = buf_out;
				int predictor_coef_table_sum = pr->predictor_coef_table_sum;
				for (i = (int)pr->predictor_coef_num + 1; i < output_size; i++)
				{
					int j;
					int sum = 0;
					int outval;
					int error_val = buf_err[i];
					int sample_val = *(buf_pos++);

					for (j = 0; j < pr->predictor_coef_num; j++)
						sum += buf_pos[j] * pr->predictor_coef_table[j];
					sum -= predictor_coef_table_sum * sample_val;
					outval = (1 << (pr->prediction_quantitization - 1)) + sum;
					outval >>= pr->prediction_quantitization;
					outval += sample_val + error_val;

					buf_pos[pr->predictor_coef_num] = extend_sign32(outval, readsamplesize);

					if (error_val != 0)
					{
						short error_sign = sign_only(error_val);
						for (j = 0; j < pr->predictor_coef_num; j++)
						{
							int val = sample_val - buf_pos[j];
							if (val == 0) 
								continue;
							short sign = sign_only(error_sign * val);
							pr->predictor_coef_table[j] -= sign;
							predictor_coef_table_sum -= sign;
							val *= sign; /* absolute value with same sign as error */
							error_val -= (val >> pr->prediction_quantitization) * (j + 1);
							if (error_val * error_sign <= 0)
								break;
						}
					}
				}
				pr->predictor_coef_table_sum = predictor_coef_table_sum;
			}
		}

		internal unsafe void deinterlace(int[,] samplesBuffer, int offset, int sampleCount)
		{
			if (sampleCount <= 0 || sampleCount > _samplesInBuffer)
				return;

			int i;
			fixed (int* buf_a = &_outputsamples_buffer_a[_samplesBufferOffset], buf_b = &_outputsamples_buffer_b[_samplesBufferOffset])
			fixed (int* buf_s = &samplesBuffer[offset, 0])
			{
				/* weighted interlacing */
				if (_interlacing_leftweight != 0)
				{
					for (i = 0; i < sampleCount; i++)
					{
						int midright = buf_a[i];
						int diff = buf_b[i];

						midright -= (diff * _interlacing_leftweight) >> _interlacing_shift;

						buf_s[i * 2] = midright + diff;
						buf_s[i * 2 + 1] = midright;

#if DEBUG
						if (buf_s[i * 2] >= (1 << pcm.BitsPerSample) || buf_s[i * 2] < -(1 << pcm.BitsPerSample) ||
							buf_s[i * 2 + 1] >= (1 << pcm.BitsPerSample) || buf_s[i * 2 + 1] < -(1 << pcm.BitsPerSample)
							)
							throw new Exception("overflow in ALAC decoder");
#endif
					}
					return;
				}

				/* otherwise basic interlacing took place */
				AudioSamples.Interlace(buf_s, buf_a, buf_b, sampleCount);
			}
		}

		internal int DecodeFrame(byte[] buffer, int pos, int len)
		{
			Array.Copy(buffer, pos, _framesBuffer, 0, len);
			decodeFrame(len);
			return len; // pos
		}

		private void decodeFrame(int sampleSize)
		{
			_bitaccumulator = 0;
			int pos = 0;

			int channels = (int) readbits(_framesBuffer, ref pos, 3);
			if (channels != 1)
				throw new Exception("Not stereo");

			readbits(_framesBuffer, ref pos, 4);
			readbits(_framesBuffer, ref pos, 12); /* unknown, skip 12 bits */
			bool hassize = 0 != readbits(_framesBuffer, ref pos, 1); /* the output sample size is stored soon */
			int wasted_bytes = (int) readbits(_framesBuffer, ref pos, 2); /* unknown ? */
			bool isnotcompressed = 0 != readbits(_framesBuffer, ref pos, 1); /* whether the frame is compressed */
			int outputSamples = hassize ? (int)readbits(_framesBuffer, ref pos, 32) : setinfo_max_samples_per_frame;

			int readsamplesize = pcm.BitsPerSample - (wasted_bytes * 8) + pcm.ChannelCount - 1;
			if (!isnotcompressed)
			{
				/* compressed */

				_interlacing_shift = (byte)readbits(_framesBuffer, ref pos, 8);
				_interlacing_leftweight = (byte)readbits(_framesBuffer, ref pos, 8);

				if (wasted_bytes != 0)
					throw new Exception("FIXME: unimplemented, unhandling of wasted_bytes");

				readPredictor(ref pos, ref predictor_info_a);
				readPredictor(ref pos, ref predictor_info_b);
				basterdised_rice_decompress(outputSamples, ref pos, ref predictor_info_a, ref _predicterror_buffer_a, readsamplesize);
				if (predictor_info_a.prediction_type == 0)
					predictor_decompress_fir_adapt(outputSamples, ref predictor_info_a, ref _predicterror_buffer_a, ref _outputsamples_buffer_a, readsamplesize);
				else
					throw new Exception("FIXME: unhandled prediction type.");
				basterdised_rice_decompress(outputSamples, ref pos, ref predictor_info_b, ref _predicterror_buffer_b, readsamplesize);
				if (predictor_info_b.prediction_type == 0)
					predictor_decompress_fir_adapt(outputSamples, ref predictor_info_b, ref _predicterror_buffer_b, ref _outputsamples_buffer_b, readsamplesize);
				else
					throw new Exception("FIXME: unhandled prediction type.");
			}
			else
			{
				/* not compressed, easy case */
				int bps = pcm.BitsPerSample;
				for (int i = 0; i < outputSamples; i++)
				{
					_outputsamples_buffer_a[i] = extend_sign32((int)readbits(_framesBuffer, ref pos, bps), bps);
					_outputsamples_buffer_b[i] = extend_sign32((int)readbits(_framesBuffer, ref pos, bps), bps);
				}
				/* wasted_bytes = 0; */
				_interlacing_shift = 0;
				_interlacing_leftweight = 0;
			}

			if (readbits(_framesBuffer, ref pos, 3) != 7)
				throw new Exception("Invalid frame.");

			_samplesInBuffer = outputSamples;
		}

		/* 'mvhd' movie header atom */
		private void qtmovie_read_chunk_mvhd(string path, uint length, object param)
		{
			UInt32 size_remaining = length;

			/* version */
			stream_read_uint8();
			size_remaining -= 1;
			stream_read_uint8();
			stream_read_uint8();
			stream_read_uint8();
			size_remaining -= 3;

			stream_read_uint32(); /* creation time */
			size_remaining -= 4;
			stream_read_uint32(); /* modification time */
			size_remaining -= 4;

			stream_read_uint32(); /* time scale */
			size_remaining -= 4;
			_sampleCount = stream_read_uint32(); /* duration */
			size_remaining -= 4;

			stream_read_uint32(); /* preferred scale */
			size_remaining -= 4;

			stream_read_uint16(); /* preferred volume */
			size_remaining -= 2;

			stream_skip(10); /* reserved */
			size_remaining -= 10;
			stream_skip(36); /* display matrix */
			size_remaining -= 36;

			stream_read_uint32(); size_remaining -= 4; /* preview time */
			stream_read_uint32(); size_remaining -= 4; /* preview duration */
			stream_read_uint32(); size_remaining -= 4; /* poster time */
			stream_read_uint32(); size_remaining -= 4; /* selection time */
			stream_read_uint32(); size_remaining -= 4; /* selection duration */
			stream_read_uint32(); size_remaining -= 4; /* current time */
			stream_read_uint32(); size_remaining -= 4; /* next track ID */

			if (size_remaining > 0)
			{
				throw new Exception("ehm, size remianing?");
				// stream_skip(size_remaining);
			}
		}

		private void qtmovie_read_chunk_stsd(string path, uint length, object param)
		{
			uint i;
			UInt32 numentries;
			UInt32 size_remaining = length;

			/* version */
			stream_read_uint8();
			size_remaining -= 1;
			/* flags */
			stream_read_uint8();
			stream_read_uint8();
			stream_read_uint8();
			size_remaining -= 3;

			numentries = stream_read_uint32();
			size_remaining -= 4;

			if (numentries != 1)
				throw new Exception("only expecting one entry in sample description atom!");

			for (i = 0; i < numentries; i++)
			{
				UInt32 entry_size;
				UInt16 version;

				UInt32 entry_remaining;

				entry_size = stream_read_uint32();
				UInt32 format = stream_read_uint32();
				entry_remaining = entry_size;
				entry_remaining -= 8;

				/* sound info: */

				stream_skip(6); /* reserved */
				entry_remaining -= 6;

				version = stream_read_uint16();
				if (version != 1)
					throw new Exception("unknown version!?");
				entry_remaining -= 2;

				/* revision level */
				stream_read_uint16();
				/* vendor */
				stream_read_uint32();
				entry_remaining -= 6;

				/* EH?? spec doesn't say theres an extra 16 bits here.. but there is! */
				stream_read_uint16();
				entry_remaining -= 2;

				int _channelCount = (int)stream_read_uint16();

				int _bitsPerSample = stream_read_uint16();
				entry_remaining -= 4;

				/* compression id */
				stream_read_uint16();
				/* packet size */
				stream_read_uint16();
				entry_remaining -= 4;

				/* sample rate - 32bit fixed point = 16bit?? */
				int _sampleRate = stream_read_uint16();
				entry_remaining -= 2;

				/* skip 2 */
				stream_skip(2);
				entry_remaining -= 2;

				pcm = new AudioPCMConfig(_bitsPerSample, _channelCount, _sampleRate);

				/* remaining is codec data */

				//#if 0
				//        qtmovie->res->codecdata_len = stream_read_uint32();
				//        if (qtmovie->res->codecdata_len != entry_remaining)
				//            fprintf(stderr, "perhaps not? %i vs %i\n",
				//                    qtmovie->res->codecdata_len, entry_remaining);
				//        entry_remaining -= 4;
				//        stream_read_uint32(); /* 'alac' */
				//        entry_remaining -= 4;

				//        qtmovie->res->codecdata = malloc(qtmovie->res->codecdata_len - 8);

				//        stream_read(qtmovie->stream,
				//                entry_remaining,
				//                qtmovie->res->codecdata);
				//        entry_remaining = 0;

				//#else
				/* 12 = audio format atom, 8 = padding */
				uint _codecDataLen = entry_remaining + 12 + 8;
				_codecData = new byte[_codecDataLen];
				/* audio format atom */
				_codecData[0] = 0; _codecData[1] = 0; _codecData[2] = 0; _codecData[3] = 0x0C;
				_codecData[4] = (byte)'f'; _codecData[5] = (byte)'r'; _codecData[6] = (byte)'m'; _codecData[7] = (byte)'a';
				_codecData[8] = (byte)'a'; _codecData[9] = (byte)'l'; _codecData[10] = (byte)'a'; _codecData[11] = (byte)'c';
				_IO.Read(_codecData, 12, (int)entry_remaining);
				entry_remaining -= entry_remaining;

				//#endif
				if (entry_remaining > 0)
					stream_skip(entry_remaining);

				_formatRead = true;
				if (format != FCC_ALAC)
					throw new Exception("Expecting ALAC data format");
			}
		}

		private void qtmovie_read_chunk_stts(string path, uint length, object param)
		{
			uint i;
			UInt32 numentries;
			UInt32 size_remaining = length;

			/* version */
			stream_read_uint8();
			size_remaining -= 1;
			/* flags */
			stream_read_uint8();
			stream_read_uint8();
			stream_read_uint8();
			size_remaining -= 3;

			numentries = stream_read_uint32();
			size_remaining -= 4;

			_time_to_sample_count = new int[numentries];
			_time_to_sample_duration = new int[numentries];

			for (i = 0; i < numentries; i++)
			{
				_time_to_sample_count[i] = (int)stream_read_uint32();
				_time_to_sample_duration[i] = (int)stream_read_uint32();
				size_remaining -= 8;
			}

			if (size_remaining > 0)
			{
				throw new Exception("ehm, size remianing?");
				// stream_skip(size_remaining);
			}
		}

		private void qtmovie_read_chunk_stsz(string path, uint length, object param)
		{
			uint i;
			UInt32 numentries;
			UInt32 size_remaining = length;

			/* version */
			stream_read_uint8();
			size_remaining -= 1;
			/* flags */
			stream_read_uint8();
			stream_read_uint8();
			stream_read_uint8();
			size_remaining -= 3;

			/* default sample size */
			if (stream_read_uint32() != 0)
			{
				throw new Exception("i was expecting variable samples sizes\n");
				//stream_read_uint32();
				//size_remaining -= 4;
				//return;
			}
			size_remaining -= 4;

			numentries = stream_read_uint32();
			size_remaining -= 4;

			_sample_byte_size = new int[numentries];

			for (i = 0; i < numentries; i++)
			{
				_sample_byte_size[i] = (int)stream_read_uint32();
				size_remaining -= 4;
			}

			if (size_remaining > 0)
			{
				throw new Exception("ehm, size remianing?\n");
				//stream_skip(qtmovie->stream, size_remaining);
			}
		}

		/* media handler inside mdia */
		private void qtmovie_read_chunk_hdlr(string path, uint length, object param)
		{
			UInt32 comptype, compsubtype;
			UInt32 size_remaining = length;

			/* version */
			stream_read_uint8();
			size_remaining -= 1;
			/* flags */
			stream_read_uint8();
			stream_read_uint8();
			stream_read_uint8();
			size_remaining -= 3;

			/* component type */
			comptype = stream_read_uint32();
			compsubtype = stream_read_uint32();
			size_remaining -= 8;

			/* component manufacturer */
			stream_read_uint32();
			size_remaining -= 4;

			/* flags */
			stream_read_uint32();
			stream_read_uint32();
			size_remaining -= 8;

			/* name */

			// Had do disable: some files have 'SoundHandler' here without preceding length;

			//UInt32 strlen = stream_read_uint8();
			//byte[] str = new byte[strlen];
			//_IO.Read(str, 0, (int)strlen);
			//size_remaining -= 1 + strlen;

			if (size_remaining > 0)
				stream_skip(size_remaining);
		}

		private void read_chunk_ftyp(UInt32 chunk_len)
		{
			UInt32 type = stream_read_uint32();
			if (type != FCC_M4A)
				throw new Exception("not M4A file.");
			UInt32 minor_ver = stream_read_uint32();
			stream_skip(chunk_len - 16);
			//UInt32 size_remaining = chunk_len - 16; /* FIXME: can't hardcode 16, size may be 64bit */
			/* compatible brands */
			//while (size_remaining > 0)
			//{
			//    /* unused */
			//    /*fourcc_t cbrand =*/
			//    stream_read_uint32();
			//    size_remaining -= 4;
			//}
		}

		private void read_chunk_mdat(UInt32 chunk_len, bool skip_mdat)
		{
			UInt32 size_remaining = chunk_len - 8; /* FIXME WRONG */
			if (size_remaining == 0)
				return;
			if (skip_mdat)
			{
				_saved_mdat_pos = _IO.Position;
				stream_skip(size_remaining);
			}
			//#if 0
			//    qtmovie->res->mdat = malloc(size_remaining);

			//    stream_read(qtmovie->stream, size_remaining, qtmovie->res->mdat);
			//#endif
		}

		private delegate void qtmovie_read_atom (string path, uint length, object param);

		private void qtmovie_read_meta_name(string path, uint length, object param)
		{
			uint language = stream_read_uint32();
			_meta_name = new ASCIIEncoding().GetString(stream_read_bytes((int)length - 4), 0, (int)length - 4);
		}

		private void qtmovie_read_meta_mean(string path, uint length, object param)
		{
			uint language = stream_read_uint32();
			_meta_mean = new ASCIIEncoding().GetString(stream_read_bytes((int)length - 4), 0, (int)length - 4);
		}

		private void qtmovie_read_meta_data(string path, uint length, object param)
		{
			uint tag_format = stream_read_uint32();
			uint language = stream_read_uint32();
			int str_size = (int)length - 8;
			if (str_size <= 0) return;
			if (tag_format != 1) throw new Exception(path + ": not a string");
			_meta_data = new UTF8Encoding().GetString(stream_read_bytes(str_size), 0, str_size);
		}

		private void qtmovie_read_meta_freeform(string path, uint length, object param)
		{
			_meta_data = _meta_name = _meta_mean = null;
			qtmovie_read_lst(path, length, param);
			if (_meta_data == null || _meta_name == null || _meta_mean == null)
				throw new Exception(path + " doesn't contain data, name or mean");
			_tags.Add(_meta_name, _meta_data);
		}

		private void qtmovie_read_meta_string(string path, uint length, object param)
		{
			_meta_data = null;
			qtmovie_read_lst(path, length, param);
			if (_meta_data == null)
				throw new Exception(path + " doesn't contain data");
			_tags.Add((string)param, _meta_data);
		}

		private void qtmovie_read_meta_binary(string path, uint length, object param)
		{
			uint tag_format = stream_read_uint32();
			uint language = stream_read_uint32();
			int str_size = (int)length - 8;
			if (str_size <= 0) return;
			if (tag_format != 0) throw new Exception(path + " not a binary");
			byte[] value = new byte[str_size];
			if (_IO.Read(value, 0, str_size) != str_size)
				throw new Exception("Decoding failed.");
			if (path.EndsWith(".trkn.data"))
			{
				if (str_size >= 4)
					_tags.Add("TRACKNUMBER", value[3].ToString());
				if (str_size >= 6)
					_tags.Add("TOTALTRACKS", value[5].ToString());
			}
			else if (path.EndsWith(".disk.data"))
			{
				if (str_size >= 4)
					_tags.Add("DISCNUMBER", value[3].ToString());
				if (str_size >= 6)
					_tags.Add("TOTALDISCS", value[5].ToString());
			}
		}

		private void qtmovie_read_nul(string path, uint length, object param)
		{
			stream_skip(length);
		}
		
		private void qtmovie_read_lst(string path, uint length, object param)
		{
			uint size_remaining = length;

			if (param != null && param is uint)
			{
				size_remaining -= (uint) param;
				stream_skip((uint)param);
			}

			StringBuilder chunk_path = new StringBuilder(path);
			chunk_path.Append('.');
			while (size_remaining > 0)
			{
				uint sub_chunk_len = stream_read_uint32();
				if (sub_chunk_len <= 1 || sub_chunk_len > size_remaining)
					throw new Exception("strange size for chunk inside "+path+".");
				stream_read_bytes(4);
				for (int c = 0; c < 4; c++)
					chunk_path.Append((char)_buff[c]);

				string chunk_path_str = chunk_path.ToString();
				qtmovie_read_atom handler;
				if (_qtmovie_parsers.TryGetValue(chunk_path_str, out handler))
					handler(chunk_path_str, sub_chunk_len - 8, _qtmovie_parser_params[chunk_path_str]);
				else
					stream_skip(sub_chunk_len - 8);
				chunk_path.Length -= 4;
				size_remaining -= sub_chunk_len;
			}
		}

		private void qtmovie_add_any_parser (string path, qtmovie_read_atom handler, object param)
		{
			_qtmovie_parsers.Add(path, handler);
			_qtmovie_parser_params.Add(path, param);
		}

		private void qtmovie_add_lst_parser(string path, object param)
		{
			qtmovie_add_any_parser(path, new qtmovie_read_atom(qtmovie_read_lst), param);
		}

		private void qtmovie_add_nul_parser(string path)
		{
			qtmovie_add_any_parser(path, new qtmovie_read_atom(qtmovie_read_nul), null);
		}

		private void qtmovie_add_tag_parser(string path, string flacName)
		{
			qtmovie_add_any_parser(path, new qtmovie_read_atom(qtmovie_read_meta_string), flacName);
			qtmovie_add_any_parser(path + ".data", new qtmovie_read_atom(qtmovie_read_meta_data), null);
		}

		private void qtmovie_add_tag_parser(string path)
		{
			qtmovie_add_lst_parser(path, null);
			qtmovie_add_any_parser(path + ".data", new qtmovie_read_atom(qtmovie_read_meta_binary), null);
		}
		
		private void qtmovie_read()
		{
			bool found_moov = false;
			bool found_mdat = false;

			_qtmovie_parsers = new Dictionary<string, qtmovie_read_atom>();
			_qtmovie_parser_params = new Dictionary<string, object>();

			qtmovie_add_lst_parser("top.moov", null);
			qtmovie_add_any_parser("top.moov.mvhd", new qtmovie_read_atom(qtmovie_read_chunk_mvhd), null);
			qtmovie_add_nul_parser("top.moov.elst");
			qtmovie_add_nul_parser("top.moov.iods");
			qtmovie_add_lst_parser("top.moov.trak", null); /* 'trak' - a movie track */
			qtmovie_add_nul_parser("top.moov.trak.tkhd");
			qtmovie_add_nul_parser("top.moov.trak.edts");
			qtmovie_add_lst_parser("top.moov.trak.mdia", null);
			qtmovie_add_any_parser("top.moov.trak.mdia.hdlr", new qtmovie_read_atom(qtmovie_read_chunk_hdlr), null);
			qtmovie_add_lst_parser("top.moov.trak.mdia.minf", null); 
			qtmovie_add_nul_parser("top.moov.trak.mdia.minf.smhd"); // required, unused
			qtmovie_add_nul_parser("top.moov.trak.mdia.minf.dinf"); // required, unused
			qtmovie_add_lst_parser("top.moov.trak.mdia.minf.stbl", null); // SAMPLE TABLE, required
			qtmovie_add_any_parser("top.moov.trak.mdia.minf.stbl.stsd", new qtmovie_read_atom(qtmovie_read_chunk_stsd), null); // _codecData
			qtmovie_add_any_parser("top.moov.trak.mdia.minf.stbl.stts", new qtmovie_read_atom(qtmovie_read_chunk_stts), null); // _time_to_sample_*
			qtmovie_add_any_parser("top.moov.trak.mdia.minf.stbl.stsz", new qtmovie_read_atom(qtmovie_read_chunk_stsz), null); // _sample_byte_size
			qtmovie_add_nul_parser("top.moov.trak.mdia.minf.stbl.stsc"); /* skip these, no indexing for us! */
			qtmovie_add_nul_parser("top.moov.trak.mdia.minf.stbl.stco"); /* skip these, no indexing for us! */
			qtmovie_add_nul_parser("top.moov.udta");
			//qtmovie_add_lst_parser("top.moov.udta", null);
			//qtmovie_add_lst_parser("top.moov.udta.meta", (uint)4);
			//qtmovie_add_lst_parser("top.moov.udta.meta.ilst", null);
			//qtmovie_add_tag_parser("top.moov.udta.meta.ilst.©nam", "TITLE");
			//qtmovie_add_tag_parser("top.moov.udta.meta.ilst.©ART", "ARTIST");
			//qtmovie_add_tag_parser("top.moov.udta.meta.ilst.©wrt", "COMPOSER");
			//qtmovie_add_tag_parser("top.moov.udta.meta.ilst.©alb", "ALBUM");
			//qtmovie_add_tag_parser("top.moov.udta.meta.ilst.©day", "DATE");
			//qtmovie_add_tag_parser("top.moov.udta.meta.ilst.©gen", "GENRE");
			//qtmovie_add_tag_parser("top.moov.udta.meta.ilst.disk");
			//qtmovie_add_tag_parser("top.moov.udta.meta.ilst.trkn");
			//qtmovie_add_any_parser("top.moov.udta.meta.ilst.----", new qtmovie_read_atom(qtmovie_read_meta_freeform), null);
			//qtmovie_add_any_parser("top.moov.udta.meta.ilst.----.mean", new qtmovie_read_atom(qtmovie_read_meta_mean), null);
			//qtmovie_add_any_parser("top.moov.udta.meta.ilst.----.name", new qtmovie_read_atom(qtmovie_read_meta_name), null);
			//qtmovie_add_any_parser("top.moov.udta.meta.ilst.----.data", new qtmovie_read_atom(qtmovie_read_meta_data), null);

			while (true)
			{
				UInt32 chunk_len = stream_read_uint32();
				if (chunk_len == 1)
					throw new Exception("need 64bit support.");
				UInt32 chunk_id = stream_read_uint32();
				switch (chunk_id)
				{
					case FCC_FTYP:
						read_chunk_ftyp(chunk_len);
						break;
					case FCC_MOOV:
						qtmovie_read_lst("top.moov", chunk_len - 8, null);
						if (found_mdat)
						{
							_IO.Position = _saved_mdat_pos;
							return;
						}
						found_moov = true;
						break;
					/* if we hit mdat before we've found moov, record the position
					 * and move on. We can then come back to mdat later.
					 * This presumes the stream supports seeking backwards.
					 */
					case FCC_MDAT:
						read_chunk_mdat(chunk_len, !found_moov);
						if (found_moov)
							return;
						found_mdat = true;
						break;

					/*  these following atoms can be skipped !!!! */
					case FCC_FREE:
						stream_skip(chunk_len - 8); /* FIXME not 8 */
						break;
					default:
						throw new Exception(String.Format("(top) unknown chunk id: {0}.", chunk_id));
				}
			}
		}

		string _path;
		Stream _IO;

		byte[] _codecData;
		int[] _time_to_sample_count, _time_to_sample_duration, _sample_byte_size;
		long _saved_mdat_pos;
		bool _formatRead;
		int _bitaccumulator;
		int setinfo_max_samples_per_frame;
		byte setinfo_rice_initialhistory;
		byte setinfo_rice_kmodifier;
		byte setinfo_rice_historymult;

		int[] _predicterror_buffer_a,
			_predicterror_buffer_b, 
			_outputsamples_buffer_a, 
			_outputsamples_buffer_b;
		predictor_t predictor_info_a;
		predictor_t predictor_info_b;

		NameValueCollection _tags;
		int _samplesInBuffer, _samplesBufferOffset;
		byte[] _framesBuffer;
		byte[] _buff;
		byte _interlacing_shift;
		byte _interlacing_leftweight;
		AudioPCMConfig pcm;
		long _sampleCount;
		long _sampleOffset;
		long _iSample;

		Dictionary<string, qtmovie_read_atom> _qtmovie_parsers;
		Dictionary<string, object> _qtmovie_parser_params;
		string _meta_data, _meta_name, _meta_mean;

		unsafe struct predictor_t
		{
			public fixed short predictor_coef_table[32];
			public int predictor_coef_table_sum;
			public int predictor_coef_num;
			public int prediction_type;
			public int prediction_quantitization;
			public int ricemodifier;
		}

		const UInt32 FCC_FTYP = ('f' << 24) + ('t' << 16) + ('y' << 8) + ('p' << 0);
		const UInt32 FCC_MOOV = ('m' << 24) + ('o' << 16) + ('o' << 8) + ('v' << 0);
		const UInt32 FCC_MDAT = ('m' << 24) + ('d' << 16) + ('a' << 8) + ('t' << 0);
		const UInt32 FCC_FREE = ('f' << 24) + ('r' << 16) + ('e' << 8) + ('e' << 0);
		const UInt32 FCC_M4A = ('M' << 24) + ('4' << 16) + ('A' << 8) + (' ' << 0);
		const UInt32 FCC_ALAC = ('a' << 24) + ('l' << 16) + ('a' << 8) + ('c' << 0);
	}
}
	