using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace CUETools.Codecs.FLAKE
{
	public class FlakeReader: IAudioSource
	{
		int[] flac_blocksizes;
		int[] flac_bitdepths;

		int[] samplesBuffer;
		int[] residualBuffer;

		byte[] _framesBuffer;
		int _framesBufferLength = 0, _framesBufferOffset = 0;

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
			_IO = IO != null ? IO : new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);

			flac_bitdepths = new int[8] { 0, 8, 12, 0, 16, 20, 24, 0 };
			flac_blocksizes = new int[15] { 0, 192, 576, 1152, 2304, 4608, 0, 0, 256, 512, 1024, 2048, 4096, 8192, 16384 };

			crc8 = new Crc8();
			crc16 = new Crc16();
			
			_framesBuffer = new byte[0x10000];
			decode_metadata();

			//max_frame_size = 16 + ((Flake.MAX_BLOCKSIZE * (int)(bits_per_sample * channels + 1) + 7) >> 3);
			if (max_frame_size * 2 > _framesBuffer.Length)
			{
				byte[] temp = _framesBuffer;
				_framesBuffer = new byte[max_frame_size * 2];
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
			flac_bitdepths = new int[8] { 0, 8, 12, 0, 16, 20, 24, 0 };
			flac_blocksizes = new int[15] { 0, 192, 576, 1152, 2304, 4608, 0, 0, 256, 512, 1024, 2048, 4096, 8192, 16384 };

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
				throw new Exception("seeking not yet supported");
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

		void interlace(int [,] buff, int offset, int count)
		{
			for (int ch = 0; ch < channels ; ch++)
				for (int i = 0; i < count; i++)
					buff[offset + i, ch] = samplesBuffer[_samplesBufferOffset + i + ch * Flake.MAX_BLOCKSIZE];
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

		void fill_frames_buffer()
		{
			if (_framesBufferLength == 0)
				_framesBufferOffset = 0;
			else if (_framesBufferLength < _framesBuffer.Length / 2 && _framesBufferOffset >= _framesBuffer.Length / 2)
			{
				Array.Copy(_framesBuffer, _framesBufferOffset, _framesBuffer, 0, _framesBufferLength);
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

			if (bitreader.readbits(16) != 0xFFF8)
				throw new Exception("invalid frame");

			frame->bs_code0 = (int) bitreader.readbits(4);
			uint sr_code0 = bitreader.readbits(4);
			frame->ch_mode = (ChannelMode)bitreader.readbits(4);
			uint bps_code = bitreader.readbits(3);
			if (flac_bitdepths[bps_code] != bits_per_sample)
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
				frame->blocksize = flac_blocksizes[frame->bs_code0];

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
			frame->subframes[ch].residual[0] = bitreader.readbits_signed(obits);
		}

		unsafe void decode_subframe_verbatim(BitReader bitreader, FlacFrame* frame, int ch)
		{
			int obits = (int)frame->subframes[ch].obits;
			for (int i = 0; i < frame->blocksize; i++)
				frame->subframes[ch].residual[i] = bitreader.readbits_signed(obits);
		}

		unsafe void decode_residual(BitReader bitreader, FlacFrame* frame, int ch)
		{
			// rice-encoded block
			uint coding_method = bitreader.readbits(2); // ????? == 0
			if (coding_method != 0) // if 1, then parameter length == 5 bits instead of 4
				throw new Exception("unsupported residual coding");
			// partition order
			frame->subframes[ch].rc.porder = (int)bitreader.readbits(4);
			if (frame->subframes[ch].rc.porder > 8)
				throw new Exception("invalid partition order");
			int psize = frame->blocksize >> frame->subframes[ch].rc.porder;
			int res_cnt = psize - frame->subframes[ch].order;

			// residual
			int j = frame->subframes[ch].order;
			int* r = frame->subframes[ch].residual + j;
			for (int p = 0; p < (1 << frame->subframes[ch].rc.porder); p++)
			{
				uint k = frame->subframes[ch].rc.rparams[p] = bitreader.readbits(4);
				if (p == 1) res_cnt = psize;
				int n = Math.Min(res_cnt, frame->blocksize - j);
				if (k == 0)
					for (int i = n; i > 0; i--)
						*(r++) = bitreader.read_unary_signed();
				else if (k <= 8)
					for (int i = n; i > 0; i--)
						*(r++) = bitreader.read_rice_signed8((int)k);
				else
					for (int i = n; i > 0; i--)
						*(r++) = bitreader.read_rice_signed((int)k);
				j += n;
			}
		}

		unsafe void decode_subframe_fixed(BitReader bitreader, FlacFrame* frame, int ch)
		{
			// warm-up samples
			int obits = (int)frame->subframes[ch].obits;
			for (int i = 0; i < frame->subframes[ch].order; i++)
				frame->subframes[ch].residual[i] = bitreader.readbits_signed(obits);

			// residual
			decode_residual(bitreader, frame, ch);
		}

		unsafe void decode_subframe_lpc(BitReader bitreader, FlacFrame* frame, int ch)
		{
			// warm-up samples
			int obits = (int)frame->subframes[ch].obits;
			for (int i = 0; i < frame->subframes[ch].order; i++)
				frame->subframes[ch].residual[i] = bitreader.readbits_signed(obits);

			// LPC coefficients
			frame->subframes[ch].cbits = (int)bitreader.readbits(4) + 1; // lpc_precision
			frame->subframes[ch].shift = bitreader.readbits_signed(5);
			for (int i = 0; i < frame->subframes[ch].order; i++)
				frame->subframes[ch].coefs[i] = bitreader.readbits_signed(frame->subframes[ch].cbits);

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

				frame->subframes[ch].type = (SubframeType)type_code;
				frame->subframes[ch].order = 0;

				if ((type_code & (uint)SubframeType.LPC) != 0)
				{
					frame->subframes[ch].order = (type_code - (int)SubframeType.LPC) + 1;
					frame->subframes[ch].type = SubframeType.LPC;
				}
				else if ((type_code & (uint)SubframeType.Fixed) != 0)
				{
					frame->subframes[ch].order = (type_code - (int)SubframeType.Fixed);
					frame->subframes[ch].type = SubframeType.Fixed;
				}

				frame->subframes[ch].residual = r + ch * Flake.MAX_BLOCKSIZE;
				frame->subframes[ch].samples = s + ch * Flake.MAX_BLOCKSIZE;

				// subframe
				switch (frame->subframes[ch].type)
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
			FlacSubframe* sub = frame->subframes + ch;

			Flake.memcpy(sub->samples, sub->residual, sub->order);
			int* data = sub->samples + sub->order;
			int* residual = sub->residual + sub->order;
			int data_len = frame->blocksize - sub->order;
			switch (sub->order)
			{
				case 0:
					Flake.memcpy(data, residual, data_len);
					break;
				case 1:
					for (int i = 0; i < data_len; i++)
						data[i] = residual[i] + data[i - 1];
					break;
				case 2:
					for (int i = 0; i < data_len; i++)
						data[i] = residual[i] + (data[i - 1] << 1) - data[i - 2];
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
			FlacSubframe* sub = frame->subframes + ch;

			if ((ulong)sub->order * ((1UL << (int)sub->obits) - 1) * ((1U << sub->cbits) - 1) >= (1UL << 32))
				lpc.decode_residual_long(sub->residual, sub->samples, frame->blocksize, sub->order, sub->coefs, sub->shift);
			else
				lpc.decode_residual(sub->residual, sub->samples, frame->blocksize, sub->order, sub->coefs, sub->shift);
		}

		unsafe void restore_samples(FlacFrame* frame)
		{
			for (int ch = 0; ch < channels; ch++)
			{
				switch (frame->subframes[ch].type)
				{
					case SubframeType.Constant:
						Flake.memset(frame->subframes[ch].samples, frame->subframes[ch].residual[0], frame->blocksize);
						break;
					case SubframeType.Verbatim:
						Flake.memcpy(frame->subframes[ch].samples, frame->subframes[ch].residual, frame->blocksize);
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
					int x = (int) frame->subframes[ch].wbits;
					for (int i = 0; i < frame->blocksize; i++)
						frame->subframes[ch].samples[i] <<= x;
				}
			}
			switch (frame->ch_mode)
			{
				case ChannelMode.NotStereo:
					break;
				case ChannelMode.LeftRight:
					break;
				case ChannelMode.MidSide:
					for (int i = 0; i < frame->blocksize; i++)
					{
						int mid = frame->subframes[0].samples[i];
						int side = frame->subframes[1].samples[i];
						mid <<= 1;
						mid |= (side & 1); /* i.e. if 'side' is odd... */
						frame->subframes[0].samples[i] = (mid + side) >> 1;
						frame->subframes[1].samples[i] = (mid - side) >> 1;
					}
					break;
				case ChannelMode.LeftSide:
					for (int i = 0; i < frame->blocksize; i++)
						frame->subframes[1].samples[i] = frame->subframes[0].samples[i] - frame->subframes[1].samples[i];
					break;
				case ChannelMode.RightSide:
					for (int i = 0; i < frame->blocksize; i++)
						frame->subframes[0].samples[i] += frame->subframes[1].samples[i];
					break;
			}
		}

		public unsafe int DecodeFrame(byte[] buffer, int pos, int len)
		{
			BitReader bitreader = new BitReader(buffer, pos, len);
			FlacFrame frame;
			FlacSubframe* subframes = stackalloc FlacSubframe[channels];
			frame.subframes = subframes;
			decode_frame_header(bitreader, &frame);
			decode_subframes(bitreader, &frame);
			bitreader.flush();
			ushort crc = crc16.ComputeChecksum(bitreader.Buffer, pos, bitreader.Position - pos);
			if (crc != bitreader.readbits(16))
				throw new Exception("frame crc mismatch");
			restore_samples(&frame);
			restore_samples(&frame);
			_samplesInBuffer = (uint)frame.blocksize;
			return bitreader.Position - pos;
		}


		bool skip_bytes(int bytes)
		{
			for (int j = 0; j < bytes; j++)
				if (0 == _IO.Read(_framesBuffer, 0, 1))
					return false;
			return true;
		}

		void decode_metadata()
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
				long pos = _IO.Position;
				fill_frames_buffer();
				BitReader bitreader = new BitReader(_framesBuffer, _framesBufferOffset, _framesBufferLength - _framesBufferOffset);
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
					sample_rate = (int) bitreader.readbits(FLAC__STREAM_METADATA_STREAMINFO_SAMPLE_RATE_LEN);
					channels = 1 + (int) bitreader.readbits(FLAC__STREAM_METADATA_STREAMINFO_CHANNELS_LEN);
					bits_per_sample = 1 + bitreader.readbits(FLAC__STREAM_METADATA_STREAMINFO_BITS_PER_SAMPLE_LEN);
					_sampleCount = (((ulong)bitreader.readbits(4)) << 4) + (ulong)bitreader.readbits(32);
					bitreader.skipbits(FLAC__STREAM_METADATA_STREAMINFO_MD5SUM_LEN);
				}
				if (_framesBufferLength < 4 + len)
				{
					_IO.Position = pos + 4 + len;
					_framesBufferLength = 0;
				}
				else
				{
					_framesBufferLength -= 4 + len;
					_framesBufferOffset += 4 + len;
				}
				if (is_last)
					break;
			} while (true);
		}
	}
}
