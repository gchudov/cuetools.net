/**
 * CUETools.Flake: pure managed FLAC audio encoder
 * Copyright (c) 2009 Gregory S. Chudov
 * Based on Flake encoder, http://flake-enc.sourceforge.net/
 * Copyright (c) 2006-2009 Justin Ruggles
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
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace CUETools.Codecs.FLAKE
{
	public class FlakeReader: IAudioSource
	{
		int[] samplesBuffer;
		int[] residualBuffer;

		byte[] _framesBuffer;
		int _framesBufferLength = 0, _framesBufferOffset = 0;
		long first_frame_offset;

		SeekPoint[] seek_table;

		Crc8 crc8;
		Crc16 crc16;
		int channels;
		uint bits_per_sample;
		int sample_rate = 44100;

		uint min_block_size = 0;
		uint max_block_size = 0;
		uint min_frame_size = 0;
		uint max_frame_size = 0;

		uint _samplesInBuffer, _samplesBufferOffset;
		ulong _sampleCount = 0;
		ulong _sampleOffset = 0;

		string _path;
		Stream _IO;

		public int[] Samples
		{
			get
			{
				return samplesBuffer;
			}
		}

		public FlakeReader(string path, Stream IO)
		{
			_path = path;
			_IO = IO != null ? IO : new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 0x10000);

			crc8 = new Crc8();
			crc16 = new Crc16();
			
			_framesBuffer = new byte[0x20000];
			decode_metadata();

			//max_frame_size = 16 + ((Flake.MAX_BLOCKSIZE * (int)(bits_per_sample * channels + 1) + 7) >> 3);
			if (((int)max_frame_size * (int)bits_per_sample * channels * 2 >> 3) > _framesBuffer.Length)
			{
				byte[] temp = _framesBuffer;
				_framesBuffer = new byte[((int)max_frame_size * (int)bits_per_sample * channels * 2 >> 3)];
				if (_framesBufferLength > 0)
					Array.Copy(temp, _framesBufferOffset, _framesBuffer, 0, _framesBufferLength);
				_framesBufferOffset = 0;
			}
			_samplesInBuffer = 0;

			if (bits_per_sample != 16 || channels != 2 || sample_rate != 44100)
				throw new Exception("invalid flac file");

			samplesBuffer = new int[Flake.MAX_BLOCKSIZE * channels];
			residualBuffer = new int[Flake.MAX_BLOCKSIZE * channels];
		}

		public FlakeReader(int _channels, uint _bits_per_sample)
		{
			crc8 = new Crc8();
			crc16 = new Crc16();

			channels = _channels;
			bits_per_sample = _bits_per_sample;
			samplesBuffer = new int[Flake.MAX_BLOCKSIZE * channels];
			residualBuffer = new int[Flake.MAX_BLOCKSIZE * channels];
		}

		public void Close()
		{
			_IO.Close();
		}

		public int[,] Read(int[,] buff)
		{
			return AudioSamples.Read(this, buff);
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
				return Length - Position;
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
				if (value > _sampleCount)
					throw new Exception("seeking past end of stream");
				if (value < Position || value > _sampleOffset)
				{
					if (seek_table != null && _IO.CanSeek)
					{
						int best_st = -1;
						for (int st = 0; st < seek_table.Length; st++)
						{
							if (seek_table[st].number <= value &&
								(best_st == -1 || seek_table[st].number > seek_table[best_st].number))
								best_st = st;
						}
						if (best_st != -1)
						{
							_framesBufferLength = 0;
							_samplesInBuffer = 0;
							_IO.Position = (long)seek_table[best_st].offset + first_frame_offset;
							_sampleOffset = seek_table[best_st].number;
						}
					}
					if (value < Position)
						throw new Exception("cannot seek backwards without seek table");
				}
				while (value > _sampleOffset)
				{
					_samplesInBuffer = 0;
					_samplesBufferOffset = 0;

					fill_frames_buffer();
					if (_framesBufferLength == 0)
						throw new Exception("seek failed");

					int bytesDecoded = DecodeFrame(_framesBuffer, _framesBufferOffset, _framesBufferLength);
					_framesBufferLength -= bytesDecoded;
					_framesBufferOffset += bytesDecoded;

					_sampleOffset += _samplesInBuffer;
				};
				uint diff = _samplesInBuffer - (uint)(_sampleOffset - value);
				_samplesInBuffer -= diff;
				_samplesBufferOffset += diff;
			}
		}

		public int BitsPerSample
		{
			get
			{
				return (int)bits_per_sample;
			}
		}

		public int ChannelCount
		{
			get
			{
				return channels;
			}
		}

		public int SampleRate
		{
			get
			{
				return sample_rate;
			}
		}

		public string Path
		{
			get
			{
				return _path;
			}
		}

		unsafe void interlace(int [,] buff, int offset, int count)
		{
			if (channels == 2)
			{
				fixed (int* res = &buff[offset, 0], src = &samplesBuffer[_samplesBufferOffset])
					Flake.interlace(res, src, src + Flake.MAX_BLOCKSIZE, count);
			}
			else
			{
				for (int ch = 0; ch < channels; ch++)
					fixed (int* res = &buff[offset, ch], src = &samplesBuffer[_samplesBufferOffset + ch * Flake.MAX_BLOCKSIZE])
					{
						int* psrc = src;
						for (int i = 0; i < count; i++)
							res[i + i] = *(psrc++);
					}
			}
		}

		public uint Read(int[,] buff, uint sampleCount)
		{
			uint offset = 0;

			while (_samplesInBuffer < sampleCount)
			{
				if (_samplesInBuffer > 0)
				{
					interlace(buff, (int)offset, (int)_samplesInBuffer);
					sampleCount -= (uint)_samplesInBuffer;
					offset += _samplesInBuffer;
					_samplesInBuffer = 0;
					_samplesBufferOffset = 0;
				}
				
				fill_frames_buffer();

				if (_framesBufferLength == 0)
					return offset;

				int bytesDecoded = DecodeFrame(_framesBuffer, _framesBufferOffset, _framesBufferLength);
				_framesBufferLength -= bytesDecoded;
				_framesBufferOffset += bytesDecoded;

				_samplesInBuffer -= _samplesBufferOffset; // can be set by Seek, otherwise zero
				_sampleOffset += _samplesInBuffer;
			}

			interlace(buff, (int)offset, (int)sampleCount);
			_samplesInBuffer -= sampleCount;
			_samplesBufferOffset += sampleCount;
			if (_samplesInBuffer == 0)
				_samplesBufferOffset = 0;
			return (uint)offset + sampleCount;
		}

		unsafe void fill_frames_buffer()
		{
			if (_framesBufferLength == 0)
				_framesBufferOffset = 0;
			else if (_framesBufferLength < _framesBuffer.Length / 2 && _framesBufferOffset >= _framesBuffer.Length / 2)
			{
				fixed (byte* buff = _framesBuffer)
					Flake.memcpy(buff, buff + _framesBufferOffset, _framesBufferLength);
				_framesBufferOffset = 0;
			}
			while (_framesBufferLength < _framesBuffer.Length / 2)
			{
				int read = _IO.Read(_framesBuffer, _framesBufferOffset + _framesBufferLength, _framesBuffer.Length - _framesBufferOffset - _framesBufferLength);
				_framesBufferLength += read;
				if (read == 0)
					break;
			}
		}

		unsafe void decode_frame_header(BitReader bitreader, FlacFrame* frame)
		{
			int header_start = bitreader.Position;

			if (bitreader.readbits(15) != 0x7FFC)
				throw new Exception("invalid frame");
			uint vbs = bitreader.readbit();
			frame->bs_code0 = (int) bitreader.readbits(4);
			uint sr_code0 = bitreader.readbits(4);
			frame->ch_mode = (ChannelMode)bitreader.readbits(4);
			uint bps_code = bitreader.readbits(3);
			if (Flake.flac_bitdepths[bps_code] != bits_per_sample)
				throw new Exception("unsupported bps coding");
			uint t1 = bitreader.readbit(); // == 0?????
			if (t1 != 0)
				throw new Exception("unsupported frame coding");
			frame->frame_count = bitreader.read_utf8();

			// custom block size
			if (frame->bs_code0 == 6)
			{
				frame->bs_code1 = (int)bitreader.readbits(8);
				frame->blocksize = frame->bs_code1 + 1;
			}
			else if (frame->bs_code0 == 7)
			{
				frame->bs_code1 = (int)bitreader.readbits(16);
				frame->blocksize = frame->bs_code1 + 1;
			}
			else
				frame->blocksize = Flake.flac_blocksizes[frame->bs_code0];

			// custom sample rate
			if (sr_code0 < 4 || sr_code0 > 11)
			{
				// sr_code0 == 12 -> sr == bitreader.readbits(8) * 1000;
				// sr_code0 == 13 -> sr == bitreader.readbits(16);
				// sr_code0 == 14 -> sr == bitreader.readbits(16) * 10;
				throw new Exception("invalid sample rate mode");
			}

			int frame_channels = (int)frame->ch_mode + 1;
			if (frame_channels > 11)
				throw new Exception("invalid channel mode");
			if (frame_channels == 2 || frame_channels > 8) // Mid/Left/Right Side Stereo
				frame_channels = 2;
			else
				frame->ch_mode = ChannelMode.NotStereo;
			if (frame_channels != channels)
				throw new Exception("invalid channel mode");

			// CRC-8 of frame header
			byte crc = crc8.ComputeChecksum(bitreader.Buffer, header_start, bitreader.Position - header_start);
			frame->crc8 = (byte)bitreader.readbits(8);
			if (frame->crc8 != crc)
				throw new Exception("header crc mismatch");
		}

		unsafe void decode_subframe_constant(BitReader bitreader, FlacFrame* frame, int ch)
		{
			int obits = (int)frame->subframes[ch].obits;
			frame->subframes[ch].best.residual[0] = bitreader.readbits_signed(obits);
		}

		unsafe void decode_subframe_verbatim(BitReader bitreader, FlacFrame* frame, int ch)
		{
			int obits = (int)frame->subframes[ch].obits;
			for (int i = 0; i < frame->blocksize; i++)
				frame->subframes[ch].best.residual[i] = bitreader.readbits_signed(obits);
		}

		unsafe void decode_residual(BitReader bitreader, FlacFrame* frame, int ch)
		{
			// rice-encoded block
			uint coding_method = bitreader.readbits(2); // ????? == 0
			if (coding_method != 0 && coding_method != 1) // if 1, then parameter length == 5 bits instead of 4
				throw new Exception("unsupported residual coding");
			// partition order
			frame->subframes[ch].best.rc.porder = (int)bitreader.readbits(4);
			if (frame->subframes[ch].best.rc.porder > 8)
				throw new Exception("invalid partition order");
			int psize = frame->blocksize >> frame->subframes[ch].best.rc.porder;
			int res_cnt = psize - frame->subframes[ch].best.order;

			int rice_len = 4 + (int)coding_method;
			// residual
			int j = frame->subframes[ch].best.order;
			int* r = frame->subframes[ch].best.residual + j;
			for (int p = 0; p < (1 << frame->subframes[ch].best.rc.porder); p++)
			{
				if (p == 1) res_cnt = psize;
				int n = Math.Min(res_cnt, frame->blocksize - j);

				int k = frame->subframes[ch].best.rc.rparams[p] = (int)bitreader.readbits(rice_len);
				if (k == (1 << rice_len) - 1)
				{
					k = frame->subframes[ch].best.rc.esc_bps[p] = (int)bitreader.readbits(5);
					for (int i = n; i > 0; i--)
						*(r++) = bitreader.readbits_signed((int)k);
				}
				else
				{
					bitreader.read_rice_block(n, (int)k, r);
					r += n;
				}
				j += n;
			}
		}

		unsafe void decode_subframe_fixed(BitReader bitreader, FlacFrame* frame, int ch)
		{
			// warm-up samples
			int obits = (int)frame->subframes[ch].obits;
			for (int i = 0; i < frame->subframes[ch].best.order; i++)
				frame->subframes[ch].best.residual[i] = bitreader.readbits_signed(obits);

			// residual
			decode_residual(bitreader, frame, ch);
		}

		unsafe void decode_subframe_lpc(BitReader bitreader, FlacFrame* frame, int ch)
		{
			// warm-up samples
			int obits = (int)frame->subframes[ch].obits;
			for (int i = 0; i < frame->subframes[ch].best.order; i++)
				frame->subframes[ch].best.residual[i] = bitreader.readbits_signed(obits);

			// LPC coefficients
			frame->subframes[ch].best.cbits = (int)bitreader.readbits(4) + 1; // lpc_precision
			frame->subframes[ch].best.shift = bitreader.readbits_signed(5);
			if (frame->subframes[ch].best.shift < 0)
				throw new Exception("negative shift");
			for (int i = 0; i < frame->subframes[ch].best.order; i++)
				frame->subframes[ch].best.coefs[i] = bitreader.readbits_signed(frame->subframes[ch].best.cbits);

			// residual
			decode_residual(bitreader, frame, ch);
		}

		unsafe void decode_subframes(BitReader bitreader, FlacFrame* frame)
		{
			fixed (int *r = residualBuffer, s = samplesBuffer)
			for (int ch = 0; ch < channels; ch++)
			{
				// subframe header
				uint t1 = bitreader.readbit(); // ?????? == 0
				if (t1 != 0)
					throw new Exception("unsupported subframe coding");
				int type_code = (int)bitreader.readbits(6);
				frame->subframes[ch].wbits = bitreader.readbit();
				if (frame->subframes[ch].wbits != 0)
					frame->subframes[ch].wbits += bitreader.read_unary();

				frame->subframes[ch].obits = bits_per_sample - frame->subframes[ch].wbits;
				switch (frame->ch_mode)
				{
					case ChannelMode.MidSide: frame->subframes[ch].obits += (uint)ch; break;
					case ChannelMode.LeftSide: frame->subframes[ch].obits += (uint)ch; break;
					case ChannelMode.RightSide: frame->subframes[ch].obits += 1 - (uint)ch; break;
				}

				frame->subframes[ch].best.type = (SubframeType)type_code;
				frame->subframes[ch].best.order = 0;

				if ((type_code & (uint)SubframeType.LPC) != 0)
				{
					frame->subframes[ch].best.order = (type_code - (int)SubframeType.LPC) + 1;
					frame->subframes[ch].best.type = SubframeType.LPC;
				}
				else if ((type_code & (uint)SubframeType.Fixed) != 0)
				{
					frame->subframes[ch].best.order = (type_code - (int)SubframeType.Fixed);
					frame->subframes[ch].best.type = SubframeType.Fixed;
				}

				frame->subframes[ch].best.residual = r + ch * Flake.MAX_BLOCKSIZE;
				frame->subframes[ch].samples = s + ch * Flake.MAX_BLOCKSIZE;

				// subframe
				switch (frame->subframes[ch].best.type)
				{
					case SubframeType.Constant:
						decode_subframe_constant(bitreader, frame, ch);
						break;
					case SubframeType.Verbatim:
						decode_subframe_verbatim(bitreader, frame, ch);
						break;
					case SubframeType.Fixed:
						decode_subframe_fixed(bitreader, frame, ch);
						break;
					case SubframeType.LPC:
						decode_subframe_lpc(bitreader, frame, ch);
						break;
					default:
						throw new Exception("invalid subframe type");
				}
			}
		}

		unsafe void restore_samples_fixed(FlacFrame* frame, int ch)
		{
			FlacSubframeInfo* sub = frame->subframes + ch;

			Flake.memcpy(sub->samples, sub->best.residual, sub->best.order);
			int* data = sub->samples + sub->best.order;
			int* residual = sub->best.residual + sub->best.order;
			int data_len = frame->blocksize - sub->best.order;
			int s0, s1, s2;
			switch (sub->best.order)
			{
				case 0:
					Flake.memcpy(data, residual, data_len);
					break;
				case 1:
					s1 = data[-1];
					for (int i = data_len; i > 0; i--)
					{
						s1 += *(residual++);
						*(data++) = s1;
					}
					//data[i] = residual[i] + data[i - 1];
					break;
				case 2:
					s2 = data[-2];
					s1 = data[-1];
					for (int i = data_len; i > 0; i--)
					{
						s0 = *(residual++) + (s1 << 1) - s2;
						*(data++) = s0;
						s2 = s1;
						s1 = s0;
					}
					//data[i] = residual[i] + data[i - 1] * 2  - data[i - 2];
					break;
				case 3:
					for (int i = 0; i < data_len; i++)
						data[i] = residual[i] + (((data[i - 1] - data[i - 2]) << 1) + (data[i - 1] - data[i - 2])) + data[i - 3];
					break;
				case 4:
					for (int i = 0; i < data_len; i++)
						data[i] = residual[i] + ((data[i - 1] + data[i - 3]) << 2) - ((data[i - 2] << 2) + (data[i - 2] << 1)) - data[i - 4];
					break;
			}
		}

		unsafe void restore_samples_lpc(FlacFrame* frame, int ch)
		{
			FlacSubframeInfo* sub = frame->subframes + ch;
			ulong csum = 0;
			for (int i = sub->best.order; i > 0; i--)
				csum += (ulong)Math.Abs(sub->best.coefs[i - 1]);
			if ((csum << (int)sub->obits) >= 1UL << 32)
				lpc.decode_residual_long(sub->best.residual, sub->samples, frame->blocksize, sub->best.order, sub->best.coefs, sub->best.shift);
			else
				lpc.decode_residual(sub->best.residual, sub->samples, frame->blocksize, sub->best.order, sub->best.coefs, sub->best.shift);
		}

		unsafe void restore_samples(FlacFrame* frame)
		{
			for (int ch = 0; ch < channels; ch++)
			{
				switch (frame->subframes[ch].best.type)
				{
					case SubframeType.Constant:
						Flake.memset(frame->subframes[ch].samples, frame->subframes[ch].best.residual[0], frame->blocksize);
						break;
					case SubframeType.Verbatim:
						Flake.memcpy(frame->subframes[ch].samples, frame->subframes[ch].best.residual, frame->blocksize);
						break;
					case SubframeType.Fixed:
						restore_samples_fixed(frame, ch);
						break;
					case SubframeType.LPC:
						restore_samples_lpc(frame, ch);
						break;
				}
				if (frame->subframes[ch].wbits != 0)
				{
					int* s = frame->subframes[ch].samples;
					int x = (int) frame->subframes[ch].wbits;
					for (int i = frame->blocksize; i > 0; i--)
						*(s++) <<= x;
				}
			}
			if (frame->ch_mode != ChannelMode.NotStereo)
			{
				int* l = frame->subframes[0].samples;
				int* r = frame->subframes[1].samples;
				switch (frame->ch_mode)
				{
					case ChannelMode.LeftRight:
						break;
					case ChannelMode.MidSide:
						for (int i = frame->blocksize; i > 0; i--)
						{
							int mid = *l;
							int side = *r;
							mid <<= 1;
							mid |= (side & 1); /* i.e. if 'side' is odd... */
							*(l++) = (mid + side) >> 1;
							*(r++) = (mid - side) >> 1;
						}
						break;
					case ChannelMode.LeftSide:
						for (int i = frame->blocksize; i > 0; i--)
						{
							int _l = *(l++), _r = *r;
							*(r++) = _l - _r;
						}
						break;
					case ChannelMode.RightSide:
						for (int i = frame->blocksize; i > 0; i--)
							*(l++) += *(r++);
						break;
				}
			}
		}

		public unsafe int DecodeFrame(byte[] buffer, int pos, int len)
		{
			fixed (byte* buf = buffer)
			{
				BitReader bitreader = new BitReader(buf, pos, len);
				FlacFrame frame;
				FlacSubframeInfo* subframes = stackalloc FlacSubframeInfo[channels];
				frame.subframes = subframes;
				decode_frame_header(bitreader, &frame);
				decode_subframes(bitreader, &frame);
				bitreader.flush();
				ushort crc = crc16.ComputeChecksum(bitreader.Buffer + pos, bitreader.Position - pos);
				if (crc != bitreader.readbits(16))
					throw new Exception("frame crc mismatch");
				restore_samples(&frame);
				_samplesInBuffer = (uint)frame.blocksize;
				return bitreader.Position - pos;
			}
		}


		bool skip_bytes(int bytes)
		{
			for (int j = 0; j < bytes; j++)
				if (0 == _IO.Read(_framesBuffer, 0, 1))
					return false;
			return true;
		}

		unsafe void decode_metadata()
		{
			byte x;
			int i, id;
			bool first = true;
			byte[] FLAC__STREAM_SYNC_STRING = new byte[] { (byte)'f', (byte)'L', (byte)'a', (byte)'C' };
			byte[] ID3V2_TAG_ = new byte[] { (byte)'I', (byte)'D', (byte)'3' };

			for (i = id = 0; i < 4; )
			{
				if (_IO.Read(_framesBuffer, 0, 1) == 0)
					throw new Exception("FLAC stream not found");
				x = _framesBuffer[0];
				if (x == FLAC__STREAM_SYNC_STRING[i])
				{
					first = true;
					i++;
					id = 0;
					continue;
				}
				if (id < 3 && x == ID3V2_TAG_[id])
				{
					id++;
					i = 0;
					if (id == 3)
					{
						if (!skip_bytes(3))
							throw new Exception("FLAC stream not found");
						int skip = 0;
						for (int j = 0; j < 4; j++)
						{
							if (0 == _IO.Read(_framesBuffer, 0, 1))
								throw new Exception("FLAC stream not found");
							skip <<= 7;
							skip |= ((int)_framesBuffer[0] & 0x7f);
						}
						if (!skip_bytes(skip))
							throw new Exception("FLAC stream not found");
					}
					continue;
				}
				id = 0;
				if (x == 0xff) /* MAGIC NUMBER for the first 8 frame sync bits */
				{
					do
					{
						if (_IO.Read(_framesBuffer, 0, 1) == 0)
							throw new Exception("FLAC stream not found");
						x = _framesBuffer[0];
					} while (x == 0xff);
					if (x >> 2 == 0x3e) /* MAGIC NUMBER for the last 6 sync bits */
					{
						//_IO.Position -= 2;
						// state = frame
						throw new Exception("headerless file unsupported");
					}
				}
				throw new Exception("FLAC stream not found");
			}

			do
			{
				fill_frames_buffer();
				fixed (byte* buf = _framesBuffer)
				{
					BitReader bitreader = new BitReader(buf, _framesBufferOffset, _framesBufferLength - _framesBufferOffset);
					bool is_last = bitreader.readbit() != 0;
					MetadataType type = (MetadataType)bitreader.readbits(7);
					int len = (int)bitreader.readbits(24);

					if (type == MetadataType.FLAC__METADATA_TYPE_STREAMINFO)
					{
						const int FLAC__STREAM_METADATA_STREAMINFO_MIN_BLOCK_SIZE_LEN = 16; /* bits */
						const int FLAC__STREAM_METADATA_STREAMINFO_MAX_BLOCK_SIZE_LEN = 16; /* bits */
						const int FLAC__STREAM_METADATA_STREAMINFO_MIN_FRAME_SIZE_LEN = 24; /* bits */
						const int FLAC__STREAM_METADATA_STREAMINFO_MAX_FRAME_SIZE_LEN = 24; /* bits */
						const int FLAC__STREAM_METADATA_STREAMINFO_SAMPLE_RATE_LEN = 20; /* bits */
						const int FLAC__STREAM_METADATA_STREAMINFO_CHANNELS_LEN = 3; /* bits */
						const int FLAC__STREAM_METADATA_STREAMINFO_BITS_PER_SAMPLE_LEN = 5; /* bits */
						const int FLAC__STREAM_METADATA_STREAMINFO_TOTAL_SAMPLES_LEN = 36; /* bits */
						const int FLAC__STREAM_METADATA_STREAMINFO_MD5SUM_LEN = 128; /* bits */

						min_block_size = bitreader.readbits(FLAC__STREAM_METADATA_STREAMINFO_MIN_BLOCK_SIZE_LEN);
						max_block_size = bitreader.readbits(FLAC__STREAM_METADATA_STREAMINFO_MAX_BLOCK_SIZE_LEN);
						min_frame_size = bitreader.readbits(FLAC__STREAM_METADATA_STREAMINFO_MIN_FRAME_SIZE_LEN);
						max_frame_size = bitreader.readbits(FLAC__STREAM_METADATA_STREAMINFO_MAX_FRAME_SIZE_LEN);
						sample_rate = (int)bitreader.readbits(FLAC__STREAM_METADATA_STREAMINFO_SAMPLE_RATE_LEN);
						channels = 1 + (int)bitreader.readbits(FLAC__STREAM_METADATA_STREAMINFO_CHANNELS_LEN);
						bits_per_sample = 1 + bitreader.readbits(FLAC__STREAM_METADATA_STREAMINFO_BITS_PER_SAMPLE_LEN);
						_sampleCount = bitreader.readbits64(FLAC__STREAM_METADATA_STREAMINFO_TOTAL_SAMPLES_LEN);
						bitreader.skipbits(FLAC__STREAM_METADATA_STREAMINFO_MD5SUM_LEN);
					}
					else if (type == MetadataType.FLAC__METADATA_TYPE_SEEKTABLE)
					{
						int num_entries = len / 18;
						seek_table = new SeekPoint[num_entries];
						for (int e = 0; e < num_entries; e++)
						{
							seek_table[e].number = bitreader.readbits64(Flake.FLAC__STREAM_METADATA_SEEKPOINT_SAMPLE_NUMBER_LEN);
							seek_table[e].offset = bitreader.readbits64(Flake.FLAC__STREAM_METADATA_SEEKPOINT_STREAM_OFFSET_LEN);
							seek_table[e].framesize = bitreader.readbits24(Flake.FLAC__STREAM_METADATA_SEEKPOINT_FRAME_SAMPLES_LEN);
						}
					}
					if (_framesBufferLength < 4 + len)
					{
						_IO.Position += 4 + len - _framesBufferLength;
						_framesBufferLength = 0;
					}
					else
					{
						_framesBufferLength -= 4 + len;
						_framesBufferOffset += 4 + len;
					}
					if (is_last)
						break;
				}
			} while (true);
			first_frame_offset = _IO.Position - _framesBufferLength;
		}
	}
}
