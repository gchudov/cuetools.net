using System;
using System.Text;
using System.IO;
using System.Collections.Generic;
using System.Collections.Specialized;
using AudioCodecsDotNet;

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

namespace ALACDotNet
{
	public class ALACReader : IAudioSource
	{
		public ALACReader(string path)
		{
			_path = path;
			m_spIO = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
			_buff = new byte[4];
			_tags = new NameValueCollection();
			qtmovie_read();
			if (!_formatRead || _bitsPerSample != 16 || _channelCount != 2 || _sampleRate != 44100)
				throw new Exception("Invalid ALAC file.");
			_saved_mdat_pos = m_spIO.Position;
		}

		public uint Read(byte[] buff, uint sampleCount)
		{
			if (_predicterror_buffer_a == null)
			{
				setinfo_max_samples_per_frame = read_uint32(_codecData, 24);
				byte setinfo_7a = read_uint8(_codecData, 28);
				byte setinfo_sample_size = read_uint8(_codecData, 29);
				setinfo_rice_historymult = read_uint8(_codecData, 30);
				setinfo_rice_initialhistory = read_uint8(_codecData, 31);
				setinfo_rice_kmodifier = read_uint8(_codecData, 32);
				byte setinfo_7f = read_uint8(_codecData, 33);
				ushort setinfo_80 = read_uint16(_codecData, 34);
				uint setinfo_82 = read_uint32(_codecData, 38);
				uint setinfo_86 = read_uint32(_codecData, 42);
				uint setinfo_8a_rate = read_uint32(_codecData, 44);

				_predicterror_buffer_a = new int[setinfo_max_samples_per_frame];
				_predicterror_buffer_b = new int[setinfo_max_samples_per_frame];
				_outputsamples_buffer_a = new int[setinfo_max_samples_per_frame];
				_outputsamples_buffer_b = new int[setinfo_max_samples_per_frame];

				_samplesInBuffer = 0;
				_samplesBuffer = new byte[setinfo_max_samples_per_frame*4];
				_framesBuffer = new byte[65536];
			}

			ulong offset = 0;

			while (_samplesInBuffer < sampleCount)
			{
				if (_samplesInBuffer > 0)
				{
					Array.Copy(_samplesBuffer, (long) _samplesBufferOffset * 4, buff, (long)offset * 4, (long)_samplesInBuffer * 4);
					sampleCount -= (uint) _samplesInBuffer;
					offset += _samplesInBuffer;
					_samplesInBuffer = 0;
					_samplesBufferOffset = 0;
				}

				ulong sampleDuration;
				uint sampleSize;
				if ((int) _iSample >= _sample_byte_size.Length)
					return (uint)offset;
				get_sample_info(_iSample, out sampleDuration, out sampleSize);
				m_spIO.Read(_framesBuffer, 0, (int) sampleSize);
				decodeFrame(sampleDuration, sampleSize);
				if (sampleDuration != _samplesInBuffer)
					throw new Exception("sample count mismatch");
				_samplesInBuffer -= _samplesBufferOffset;
				_sampleOffset += _samplesInBuffer;
				_iSample++;
			}

			Array.Copy(_samplesBuffer, (long) _samplesBufferOffset * 4, buff, (long)offset * 4, (long)sampleCount * 4);
			_samplesInBuffer -= sampleCount;
			_samplesBufferOffset += sampleCount;
			if (_samplesInBuffer == 0)
				_samplesBufferOffset = 0;
			return (uint) offset + sampleCount;
		}

		public void Close()
		{
			m_spIO.Close();
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
				return _sampleCount - _sampleOffset + _samplesInBuffer;
			}
		}

		public ulong Position
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
				ulong durOffs = 0;
				ulong sampleDuration;
				long fileOffs = 0;
				uint sampleSize;
				do
				{
					if (durOffs == value)
					{
						m_spIO.Position = _saved_mdat_pos + fileOffs;
						return;
					}
					get_sample_info(_iSample, out sampleDuration, out sampleSize);
					durOffs += sampleDuration;
					fileOffs += sampleSize;
					_iSample++;
				} while (durOffs <= value);
				m_spIO.Position = _saved_mdat_pos + fileOffs - sampleSize;
				_samplesBufferOffset = sampleDuration - durOffs + value;
				_iSample--;
			}
		}

		public int BitsPerSample
		{
			get
			{
				return _bitsPerSample;
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

		public NameValueCollection Tags
		{
			get
			{
				return _tags;
			}
			set
			{
				_tags = value;
			}
		}

		public string Path 
		{
			get 
			{ 
				return _path; 
			} 
		}

		private void get_sample_info(ulong iSample, out ulong sampleDuration, out uint sampleSize)
		{
			// if (iSample >= _sample_byte_size.Length)
			uint duration_index_accum = 0;
			uint duration_cur_index = 0;
			while (_time_to_sample_count[duration_cur_index] + duration_index_accum <= iSample)
			{
				duration_index_accum += _time_to_sample_count[duration_cur_index];
				duration_cur_index ++;
			}
			sampleDuration = _time_to_sample_duration[duration_cur_index];
			sampleSize = _sample_byte_size[iSample];
		}

		private uint read_uint32(byte [] buff, int pos)
		{
			return (uint)((buff[pos] << 24) + (buff[pos+1] << 16) + (buff[pos+2] << 8) + (buff[pos+3] << 0));
		}

		private uint stream_read_uint32()
		{
			if (m_spIO.Read(_buff, 0, 4) != 4)
				throw new Exception("Decoding failed.");
			return read_uint32 (_buff, 0);
		}

		private ushort read_uint16(byte[] buff, int pos)
		{
			return (ushort)((buff[pos] << 8) + buff[pos + 1]);
		}

		private ushort stream_read_uint16()
		{
			if (m_spIO.Read(_buff, 0, 2) != 2)
				throw new Exception("Decoding failed.");
			return read_uint16 (_buff, 0);
		}

		private byte read_uint8(byte[] buff, int pos)
		{
			return buff[pos];
		}

		private byte stream_read_uint8()
		{
			if (m_spIO.Read(_buff, 0, 1) != 1)
				throw new Exception("Decoding failed.");
			return read_uint8 (_buff, 0);
		}

		private void stream_skip (UInt32 skip)
		{
			m_spIO.Position += skip;
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
		private unsafe uint peekbits_16(byte* buff, int pos, int bits)
		{
			uint result = (((uint)buff[pos]) << 16) | (((uint)buff[pos + 1]) << 8) | ((uint)buff[pos + 2]);
			result <<= _bitaccumulator;
			result &= 0x00ffffff;
			result >>= 24 - bits;
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

		private static uint SIGN_EXTENDED32(uint val, int bits) 
		{
			return ((val << (32 - bits)) >> (32 - bits));
		}

		private void unreadbits(ref int pos, int bits)
		{
			int new_accumulator = (_bitaccumulator - bits);
			pos += (new_accumulator >> 3);

			_bitaccumulator = (new_accumulator & 7);
			if (_bitaccumulator < 0)
				_bitaccumulator *= -1;
		}

		private static int count_leading_zeros(uint input, int first)
		{
			int zeroes = 1;
			int check = first;
			while (check > 0)
			{
				uint shifted_input = input >> check;
				if (shifted_input == 0) 
					zeroes += check; 
				else
					input = shifted_input;
				check >>= 1;
			}
			return zeroes - (int)input;
		}

		private static int count_leading_zeros(uint input)
		{
			int zeroes = 1;
			int check = 16;
			while (check > 0)
			{
				uint shifted_input = input >> check;
				if (shifted_input == 0)
					zeroes += check;
				else
					input = shifted_input;
				check >>= 1;
			}
			return zeroes - (int)input;
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
				for (int i = 0; i < pr->predictor_coef_num; i++)
				{
					pr->predictor_coef_table[i] = (short)readbits(_framesBuffer, ref pos, 16);
				}
			}
		}

		private unsafe int decode_scalar(byte * buff, ref int pos, int k, int limit, int readsamplesize)
		{
			int x = 0;
			//while (x <= 8 && readbit(_framesBuffer, ref pos) != 0)
			//    x++;
			uint next = peekbits_16(buff, pos, 9);
			for (x = 0; (next & 0x100) != 0; x++)
				next <<= 1;
			skipbits(ref pos, x >= 9 ? 9 : x + 1);
			if (x > 8) /* RICE THRESHOLD */
				return (int) readbits(buff, ref pos, readsamplesize);
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

		private unsafe void basterdised_rice_decompress(uint output_size, ref int pos, ref predictor_t predictor_info, ref int[] predicterror_buffer, int readsamplesize)
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
					int x = 0;
					int x_modified;
					int final_val;

					x = decode_scalar(buff, ref pos, 31 - count_leading_zeros((history >> 9) + 3), rice_kmodifier, readsamplesize);

					x_modified = sign_modifier + x;
					final_val = (x_modified + 1) / 2;
					if ((x_modified & 1) != 0) final_val *= -1;

					output_buffer[output_count] = final_val;

					sign_modifier = 0;

					/* now update the history */
					history = (uint)(history + (x_modified * rice_historymult)
							 - ((history * rice_historymult) >> 9));

					if (x_modified > 0xffff)
						history = 0xffff;

					/* special case: there may be compressed blocks of 0 */
					if ((history < 128) && (output_count + 1 < output_size))
					{
						int k = 7 - (31 - count_leading_zeros(history)) + (((int)history + 16) >> 6);
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

		private static int sign_only(int v)
		{
			return v == 0 ? 0 : v > 0 ? 1 : -1;
		}

		private unsafe void predictor_decompress_fir_adapt(uint output_size, ref predictor_t predictor_info, ref int[] error_buffer, ref int[] buffer_out, int readsamplesize)
		{
			int i;

			/* first sample always copies */
			buffer_out[0] = error_buffer[0];

			if (predictor_info.predictor_coef_num == 0)
			{
				if (output_size <= 1)
					return;
				for (i = 1; i < output_size; i++)
					buffer_out[i] = error_buffer[i];
				return;
			}

			if (predictor_info.predictor_coef_num == 0x1f)
			{ /* 11111 - max value of predictor_coef_num */
				/* second-best case scenario for fir decompression,
				 * error describes a small difference from the previous sample only
				 */
				if (output_size <= 1)
					return;
				for (i = 0; i < output_size - 1; i++)
				{
					int prev_value;
					int error_value;

					prev_value = buffer_out[i];
					error_value = error_buffer[i + 1];
					buffer_out[i + 1] =
						extend_sign32((prev_value + error_value), readsamplesize);
				}
				return;
			}

			/* read warm-up samples */
			if (predictor_info.predictor_coef_num > 0)
				for (i = 0; i < predictor_info.predictor_coef_num; i++)
				{
					int val;

					val = buffer_out[i] + error_buffer[i + 1];
					val = extend_sign32(val, readsamplesize);
					buffer_out[i + 1] = val;
				}

			//#if 0
			//            /* 4 and 8 are very common cases (the only ones i've seen). these
			//             * should be unrolled and optimized
			//             */
			//            if (predictor_coef_num == 4) {
			//                /* FIXME: optimized general case */
			//                return;
			//            }

			//            if (predictor_coef_table == 8) {
			//                /* FIXME: optimized general case */
			//                return;
			//            }
			//#endif

			/* general case */
			if (predictor_info.predictor_coef_num > 0)
			fixed (predictor_t* pr = &predictor_info)
			fixed (int * buf_out = &buffer_out[0], buf_err = &error_buffer[0])
			{
				int pos = 0;

				for (i = (int) predictor_info.predictor_coef_num + 1; i < output_size; i++)
				{
					int j;
					int sum = 0;
					int outval;
					int error_val = buf_err[i];

					for (j = 0; j < predictor_info.predictor_coef_num; j++)
					{
						sum += (buf_out[pos + predictor_info.predictor_coef_num - j] - buf_out[pos]) *
							   pr->predictor_coef_table[j];
					}

					outval = (1 << (predictor_info.prediction_quantitization - 1)) + sum;
					outval = outval >> predictor_info.prediction_quantitization;
					outval = outval + buf_out[pos] + error_val;
					outval = extend_sign32(outval, readsamplesize);

					buf_out[pos + pr->predictor_coef_num + 1] = outval;

					if (error_val > 0)
					{
						int predictor_num = pr->predictor_coef_num - 1;

						while (predictor_num >= 0 && error_val > 0)
						{
							int val = buf_out[pos] - buf_out[pos + predictor_info.predictor_coef_num - predictor_num];
							short sign = (short) sign_only(val);

							pr->predictor_coef_table[predictor_num] -= sign;

							val *= sign; /* absolute value */

							error_val -= ((val >> predictor_info.prediction_quantitization) *
										  (predictor_info.predictor_coef_num - predictor_num));

							predictor_num--;
						}
					}
					else if (error_val < 0)
					{
						int predictor_num = predictor_info.predictor_coef_num - 1;

						while (predictor_num >= 0 && error_val < 0)
						{
							int val = buf_out[pos] - buf_out[pos + predictor_info.predictor_coef_num - predictor_num];
							short sign = (short) sign_only(- val);

							pr->predictor_coef_table[predictor_num] -= sign;

							val *= sign; /* neg value */

							error_val -= ((val >> predictor_info.prediction_quantitization) *
										  (predictor_info.predictor_coef_num - predictor_num));

							predictor_num--;
						}
					}

					pos++;
				}
			}
		}

		private unsafe void deinterlace_16(uint numsamples, byte interlacing_shift, byte interlacing_leftweight)
		{
			int i;

			if (numsamples <= 0)
				return;

			fixed (int* buf_a = &_outputsamples_buffer_a[0], buf_b = _outputsamples_buffer_b)
			fixed (byte * buf_s = &_samplesBuffer[0])
			{
				/* weighted interlacing */
				if (interlacing_leftweight != 0)
				{
					for (i = 0; i < numsamples; i++)
					{
						int a = buf_a[i];
						int b = buf_b[i];

						a -= (b * interlacing_leftweight) >> interlacing_shift;
						b += a;

						buf_s[i * 4] = (byte)(b & 0xff);
						buf_s[i * 4 + 1] = (byte)((b >> 8) & 0xff);
						buf_s[i * 4 + 2] = (byte)(a & 0xff);
						buf_s[i * 4 + 3] = (byte)((a >> 8) & 0xff);
					}
					return;
				}

				/* otherwise basic interlacing took place */
				for (i = 0; i < numsamples; i++)
				{
					int a = buf_a[i];
					int b = buf_b[i];
					buf_s[i * 4] = (byte)(a & 0xff);
					buf_s[i * 4 + 1] = (byte)((a >> 8) & 0xff);
					buf_s[i * 4 + 2] = (byte)(b & 0xff);
					buf_s[i * 4 + 3] = (byte)((b >> 8) & 0xff);
				}
			}
		}

		private void decodeFrame(ulong sampleDuration, uint sampleSize)
		{
			_bitaccumulator = 0;
			int pos = 0;

			int channels = (int) readbits(_framesBuffer, ref pos, 3);
			if (channels != 1)
				throw new Exception("Not stereo");

			byte interlacing_shift;
			byte interlacing_leftweight;
			readbits(_framesBuffer, ref pos, 4);
			readbits(_framesBuffer, ref pos, 12); /* unknown, skip 12 bits */
			bool hassize = 0 != readbits(_framesBuffer, ref pos, 1); /* the output sample size is stored soon */
			int wasted_bytes = (int) readbits(_framesBuffer, ref pos, 2); /* unknown ? */
			bool isnotcompressed = 0 != readbits(_framesBuffer, ref pos, 1); /* whether the frame is compressed */
			uint outputSamples = hassize ? readbits(_framesBuffer, ref pos, 32) : setinfo_max_samples_per_frame;

			int readsamplesize = _bitsPerSample - (wasted_bytes * 8) + channels;
			if (!isnotcompressed)
			{
				/* compressed */

				interlacing_shift = (byte)readbits(_framesBuffer, ref pos, 8);
				interlacing_leftweight = (byte)readbits(_framesBuffer, ref pos, 8);

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
				if (_bitsPerSample != 16)
					throw new Exception("Not 16 bit");
				for (int i = 0; i < outputSamples; i++)
				{
					_outputsamples_buffer_a[i] = extend_sign32((int)readbits(_framesBuffer, ref pos, _bitsPerSample), _bitsPerSample);
					_outputsamples_buffer_b[i] = extend_sign32((int)readbits(_framesBuffer, ref pos, _bitsPerSample), _bitsPerSample);
				}
				/* wasted_bytes = 0; */
				interlacing_shift = 0;
				interlacing_leftweight = 0;
			}

			if (_bitsPerSample != 16)
				throw new Exception("Not 16 bit");

			deinterlace_16(outputSamples, interlacing_shift, interlacing_leftweight);

			_samplesInBuffer = outputSamples;
		}

		private void skip_chunk(UInt32 chunk_len)
		{
			stream_skip(chunk_len - 8); /* FIXME WRONG */
		}

		/* 'mvhd' movie header atom */
		private void read_chunk_mvhd(UInt32 chunk_len)
		{
			UInt32 size_remaining = chunk_len - 8; /* FIXME WRONG */

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

		private void read_chunk_tkhd(UInt32 chunk_len)
		{
			/* don't need anything from here atm, skip */
			skip_chunk(chunk_len);
		}

		private void read_chunk_mdhd(UInt32 chunk_len)
		{
			/* don't need anything from here atm, skip */
			skip_chunk(chunk_len);
		}

		/* 'iods' */
		private void read_chunk_iods(UInt32 chunk_len)
		{
			/* don't need anything from here atm, skip */
			skip_chunk(chunk_len);
		}

		private void read_chunk_edts(UInt32 chunk_len)
		{
			/* don't need anything from here atm, skip */
			skip_chunk(chunk_len);
		}

		private void read_chunk_elst(UInt32 chunk_len)
		{
			/* don't need anything from here atm, skip */
			skip_chunk(chunk_len);
		}

		private void read_chunk_stsd(UInt32 chunk_len)
		{
			uint i;
			UInt32 numentries;
			UInt32 size_remaining = chunk_len - 8; /* FIXME WRONG */

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

				_channelCount = (int)stream_read_uint16();

				_bitsPerSample = stream_read_uint16();
				entry_remaining -= 4;

				/* compression id */
				stream_read_uint16();
				/* packet size */
				stream_read_uint16();
				entry_remaining -= 4;

				/* sample rate - 32bit fixed point = 16bit?? */
				_sampleRate = stream_read_uint16();
				entry_remaining -= 2;

				/* skip 2 */
				stream_skip(2);
				entry_remaining -= 2;

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
				m_spIO.Read(_codecData, 12, (int)entry_remaining);
				entry_remaining -= entry_remaining;

				//#endif
				if (entry_remaining > 0)
					stream_skip(entry_remaining);

				_formatRead = true;
				if (format != FCC_ALAC)
					throw new Exception("Expecting ALAC data format");
			}
		}

		private void read_chunk_stts(UInt32 chunk_len)
		{
			uint i;
			UInt32 numentries;
			UInt32 size_remaining = chunk_len - 8; /* FIXME WRONG */

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

			_time_to_sample_count = new UInt32[numentries];
			_time_to_sample_duration = new UInt32[numentries];

			for (i = 0; i < numentries; i++)
			{
				_time_to_sample_count[i] = stream_read_uint32();
				_time_to_sample_duration[i] = stream_read_uint32();
				size_remaining -= 8;
			}

			if (size_remaining > 0)
			{
				throw new Exception("ehm, size remianing?");
				// stream_skip(size_remaining);
			}
		}

		private void read_chunk_stsz(UInt32 chunk_len)
		{
			uint i;
			UInt32 numentries;
			UInt32 size_remaining = chunk_len - 8; /* FIXME WRONG */

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

			_sample_byte_size = new uint[numentries];

			for (i = 0; i < numentries; i++)
			{
				_sample_byte_size[i] = stream_read_uint32();
				size_remaining -= 4;
			}

			if (size_remaining > 0)
			{
				throw new Exception("ehm, size remianing?\n");
				//stream_skip(qtmovie->stream, size_remaining);
			}
		}

		private void read_chunk_stbl(UInt32 chunk_len)
		{
			UInt32 size_remaining = chunk_len - 8; /* FIXME WRONG */
			while (size_remaining > 0)
			{
				UInt32 sub_chunk_len = stream_read_uint32();
				if (sub_chunk_len <= 1 || sub_chunk_len > size_remaining)
					throw new Exception("strange size for chunk inside stbl.");

				UInt32 sub_chunk_id = stream_read_uint32();
				switch (sub_chunk_id)
				{
					case FCC_STSD:
						read_chunk_stsd(sub_chunk_len);
						break;
					case FCC_STTS:
						read_chunk_stts(sub_chunk_len);
						break;
					case FCC_STSZ:
						read_chunk_stsz(sub_chunk_len);
						break;
					case FCC_STSC:
					case FCC_STCO:
						/* skip these, no indexing for us! */
						stream_skip(sub_chunk_len - 8);
						break;
					default:
						throw new Exception(String.Format("(trak) unknown chunk id: {0}.", sub_chunk_id));
				}
				size_remaining -= sub_chunk_len;
			}
		}

		private void read_chunk_minf(UInt32 chunk_len)
		{
			UInt32 dinf_size, stbl_size;
			UInt32 size_remaining = chunk_len - 8; /* FIXME WRONG */

			/**** SOUND HEADER CHUNK ****/
			if (stream_read_uint32() != 16)
				throw new Exception("unexpected size in media info\n");
			if (stream_read_uint32() != FCC_SMHD)
				throw new Exception("not a sound header! can't handle this.\n");
			/* now skip the rest */
			stream_skip(16 - 8);
			size_remaining -= 16;
			/****/

			/**** DINF CHUNK ****/
			dinf_size = stream_read_uint32();
			if (stream_read_uint32() != FCC_DINF)
				throw new Exception("expected dinf, didn't get it.");
			/* skip it */
			stream_skip(dinf_size - 8);
			size_remaining -= dinf_size;
			/****/

			/**** SAMPLE TABLE ****/
			stbl_size = stream_read_uint32();
			if (stream_read_uint32() != FCC_STBL)
				throw new Exception("expected stbl, didn't get it.");
			read_chunk_stbl(stbl_size);
			size_remaining -= stbl_size;

			if (size_remaining > 0)
			{
				throw new Exception("oops\n");
				//stream_skip(size_remaining);
			}
		}

		/* media handler inside mdia */
		private void read_chunk_hdlr(UInt32 chunk_len)
		{
			UInt32 comptype, compsubtype;
			UInt32 size_remaining = chunk_len - 8; /* FIXME WRONG */

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
			UInt32 strlen = stream_read_uint8();
			byte[] str = new byte[strlen];
			m_spIO.Read(str, 0, (int)strlen);
			size_remaining -= 1 + strlen;

			if (size_remaining > 0)
				stream_skip(size_remaining);
		}


		private void read_chunk_mdia(UInt32 chunk_len)
		{
			UInt32 size_remaining = chunk_len - 8; /* FIXME WRONG */
			while (size_remaining > 0)
			{
				UInt32 sub_chunk_len = stream_read_uint32();
				if (sub_chunk_len <= 1 || sub_chunk_len > size_remaining)
					throw new Exception("strange size for chunk inside mdia.");

				UInt32 sub_chunk_id = stream_read_uint32();
				switch (sub_chunk_id)
				{
					case FCC_MDHD:
						read_chunk_mdhd(sub_chunk_len);
						break;
					case FCC_HDLR:
						read_chunk_hdlr(sub_chunk_len);
						break;
					case FCC_MINF:
						read_chunk_minf(sub_chunk_len);
						break;
					default:
						throw new Exception(String.Format("(mdia) unknown chunk id: {0}.", sub_chunk_id));
				}

				size_remaining -= sub_chunk_len;
			}
		}

		/* 'trak' - a movie track - contains other atoms */
		private void read_chunk_trak(UInt32 chunk_len)
		{
			UInt32 size_remaining = chunk_len - 8; /* FIXME WRONG */
			while (size_remaining > 0)
			{
				UInt32 sub_chunk_len = stream_read_uint32(); ;
				if (sub_chunk_len <= 1 || sub_chunk_len > size_remaining)
					throw new Exception("strange size for chunk inside trak.");

				UInt32 sub_chunk_id = stream_read_uint32();
				switch (sub_chunk_id)
				{
					case FCC_TKHD:
						read_chunk_tkhd(sub_chunk_len);
						break;
					case FCC_MDIA:
						read_chunk_mdia(sub_chunk_len);
						break;
					case FCC_EDTS:
						read_chunk_edts(sub_chunk_len);
						break;
					default:
						throw new Exception(String.Format("(trak) unknown chunk id: {0}.", sub_chunk_id));
				}

				size_remaining -= sub_chunk_len;
			}
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
				_saved_mdat_pos = m_spIO.Position;
				stream_skip(size_remaining);
			}
			//#if 0
			//    qtmovie->res->mdat = malloc(size_remaining);

			//    stream_read(qtmovie->stream, size_remaining, qtmovie->res->mdat);
			//#endif
		}

		private void read_chunk_moov(UInt32 chunk_len)
		{
			UInt32 size_remaining = chunk_len - 8; /* FIXME WRONG */
			while (size_remaining > 0)
			{
				UInt32 sub_chunk_len = stream_read_uint32();
				if (sub_chunk_len <= 1 || sub_chunk_len > size_remaining)
					throw new Exception("strange size for chunk inside moov.");
				UInt32 sub_chunk_id = stream_read_uint32();
				switch (sub_chunk_id)
				{
					case FCC_MVHD:
						read_chunk_mvhd(sub_chunk_len);
						break;
					case FCC_TRAK:
						read_chunk_trak(sub_chunk_len);
						break;
					case FCC_UDTA:
						qtmovie_read_lst("top.moov", sub_chunk_len-8, null);
						break;
					case FCC_ELST:
						read_chunk_elst(sub_chunk_len);
						break;
					case FCC_IODS:
						read_chunk_iods(sub_chunk_len);
						break;
					default:
						throw new Exception(String.Format("(moov) unknown chunk id: {0}.", sub_chunk_id));
				}
				size_remaining -= sub_chunk_len;
			}
		}

		private delegate void qtmovie_read_atom (string path, uint length, object param);

		private void qtmovie_read_meta_name(string path, uint length, object param)
		{
			uint language = stream_read_uint32();
			length -= 4;
			byte[] value = new byte[length];
			if (m_spIO.Read(value, 0, (int) length) != length)
				throw new Exception(path + ": decoding failed.");
			_meta_name = new ASCIIEncoding().GetString(value);
		}

		private void qtmovie_read_meta_mean(string path, uint length, object param)
		{
			uint language = stream_read_uint32();
			length -= 4;
			byte[] value = new byte[length];
			if (m_spIO.Read(value, 0, (int) length) != length)
				throw new Exception(path + ": decoding failed.");
			_meta_mean = new ASCIIEncoding().GetString(value);
		}

		private void qtmovie_read_meta_data(string path, uint length, object param)
		{
			uint tag_format = stream_read_uint32();
			uint language = stream_read_uint32();
			int str_size = (int)length - 8;
			if (str_size <= 0) return;
			if (tag_format != 1) throw new Exception(path + ": not a string");
			byte[] value = new byte[str_size];
			if (m_spIO.Read(value, 0, str_size) != str_size)
				throw new Exception(path + ": decoding failed.");
			_meta_data = new UTF8Encoding().GetString(value);
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
			if (m_spIO.Read(value, 0, str_size) != str_size)
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
				if (m_spIO.Read(_buff, 0, 4) != 4)
					throw new Exception("Decoding failed.");
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

		private void qtmovie_add_parser (string path, qtmovie_read_atom handler, object param)
		{
			_qtmovie_parsers.Add(path, handler);
			_qtmovie_parser_params.Add(path, param);
		}

		private void qtmovie_add_lst_parser(string path, object param)
		{
			qtmovie_add_parser(path, new qtmovie_read_atom(qtmovie_read_lst), param);
		}

		private void qtmovie_add_tag_parser(string path, string flacName)
		{
			qtmovie_add_parser(path, new qtmovie_read_atom(qtmovie_read_meta_string), flacName);
			qtmovie_add_parser(path + ".data", new qtmovie_read_atom(qtmovie_read_meta_data), null);
		}

		private void qtmovie_add_tag_parser(string path)
		{
			qtmovie_add_lst_parser(path, null);
			qtmovie_add_parser(path + ".data", new qtmovie_read_atom(qtmovie_read_meta_binary), null);
		}
		
		private void qtmovie_read()
		{
			bool found_moov = false;
			bool found_mdat = false;

			_qtmovie_parsers = new Dictionary<string, qtmovie_read_atom>();
			_qtmovie_parser_params = new Dictionary<string, object>();

			qtmovie_add_lst_parser("top.moov.meta", (uint)4);
			qtmovie_add_lst_parser("top.moov.meta.ilst", null);
			qtmovie_add_tag_parser("top.moov.meta.ilst.©nam", "TITLE");
			qtmovie_add_tag_parser("top.moov.meta.ilst.©ART", "ARTIST");
			qtmovie_add_tag_parser("top.moov.meta.ilst.©wrt", "COMPOSER");
			qtmovie_add_tag_parser("top.moov.meta.ilst.©alb", "ALBUM");
			qtmovie_add_tag_parser("top.moov.meta.ilst.©day", "DATE");
			qtmovie_add_tag_parser("top.moov.meta.ilst.©gen", "GENRE");
			qtmovie_add_tag_parser("top.moov.meta.ilst.disk");
			qtmovie_add_tag_parser("top.moov.meta.ilst.trkn");
			qtmovie_add_parser("top.moov.meta.ilst.----", new qtmovie_read_atom(qtmovie_read_meta_freeform), null);
			qtmovie_add_parser("top.moov.meta.ilst.----.mean", new qtmovie_read_atom(qtmovie_read_meta_mean), null);
			qtmovie_add_parser("top.moov.meta.ilst.----.name", new qtmovie_read_atom(qtmovie_read_meta_name), null);
			qtmovie_add_parser("top.moov.meta.ilst.----.data", new qtmovie_read_atom(qtmovie_read_meta_data), null);

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
						read_chunk_moov(chunk_len);
						if (found_mdat)
						{
							m_spIO.Position = _saved_mdat_pos;
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
		FileStream m_spIO;

		byte[] _codecData;
		uint[] _time_to_sample_count, _time_to_sample_duration, _sample_byte_size;
		long _saved_mdat_pos;
		bool _formatRead;
		int _bitaccumulator;
		uint setinfo_max_samples_per_frame;
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
		ulong _samplesInBuffer;
		byte[] _samplesBuffer;
		byte[] _framesBuffer;
		byte[] _buff;
		int _sampleRate;
		int _channelCount;
		int _bitsPerSample;
		ulong _sampleCount;
		ulong _sampleOffset;
		ulong _samplesBufferOffset;
		ulong _iSample;

		Dictionary<string, qtmovie_read_atom> _qtmovie_parsers;
		Dictionary<string, object> _qtmovie_parser_params;
		string _meta_data, _meta_name, _meta_mean;

		unsafe struct predictor_t
		{
			public fixed short predictor_coef_table[32];
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
		const UInt32 FCC_MVHD = ('m' << 24) + ('v' << 16) + ('h' << 8) + ('d' << 0);
		const UInt32 FCC_TRAK = ('t' << 24) + ('r' << 16) + ('a' << 8) + ('k' << 0);
		const UInt32 FCC_UDTA = ('u' << 24) + ('d' << 16) + ('t' << 8) + ('a' << 0);
		const UInt32 FCC_ELST = ('e' << 24) + ('l' << 16) + ('s' << 8) + ('t' << 0);
		const UInt32 FCC_IODS = ('i' << 24) + ('o' << 16) + ('d' << 8) + ('s' << 0);
		const UInt32 FCC_TKHD = ('t' << 24) + ('k' << 16) + ('h' << 8) + ('d' << 0);
		const UInt32 FCC_MDIA = ('m' << 24) + ('d' << 16) + ('i' << 8) + ('a' << 0);
		const UInt32 FCC_EDTS = ('e' << 24) + ('d' << 16) + ('t' << 8) + ('s' << 0);
		const UInt32 FCC_MDHD = ('m' << 24) + ('d' << 16) + ('h' << 8) + ('d' << 0);
		const UInt32 FCC_HDLR = ('h' << 24) + ('d' << 16) + ('l' << 8) + ('r' << 0);
		const UInt32 FCC_MINF = ('m' << 24) + ('i' << 16) + ('n' << 8) + ('f' << 0);
		const UInt32 FCC_SMHD = ('s' << 24) + ('m' << 16) + ('h' << 8) + ('d' << 0);
		const UInt32 FCC_DINF = ('d' << 24) + ('i' << 16) + ('n' << 8) + ('f' << 0);
		const UInt32 FCC_STBL = ('s' << 24) + ('t' << 16) + ('b' << 8) + ('l' << 0);
		const UInt32 FCC_STSD = ('s' << 24) + ('t' << 16) + ('s' << 8) + ('d' << 0);
		const UInt32 FCC_STTS = ('s' << 24) + ('t' << 16) + ('t' << 8) + ('s' << 0);
		const UInt32 FCC_STSZ = ('s' << 24) + ('t' << 16) + ('s' << 8) + ('z' << 0);
		const UInt32 FCC_STSC = ('s' << 24) + ('t' << 16) + ('s' << 8) + ('c' << 0);
		const UInt32 FCC_STCO = ('s' << 24) + ('t' << 16) + ('c' << 8) + ('o' << 0);
		const UInt32 FCC_ALAC = ('a' << 24) + ('l' << 16) + ('a' << 8) + ('c' << 0);
	}
}
	