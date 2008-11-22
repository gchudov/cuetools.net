// ****************************************************************************
// 
// Copyright (C) 2007, 2008 Nick Currie, Copyleft.
// Contact: lossywav <at> yahoo <dot> com
// Added noise WAV bit reduction method by David Robinson;
// Noise shaping coefficients by Sebastian Gesemann;
// C# version by Gregory S. Chudov <gchudov <at> gmail <dot> com>;
// 
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// 
// ****************************************************************************

using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Text;
using AudioCodecsDotNet;

namespace LossyWAVDotNet
{
	public class LossyWAVWriter : IAudioDest
	{
		#region Public Methods

		public const string version_string = "1.1.1#";

		public LossyWAVWriter(IAudioDest audioDest, IAudioDest lwcdfDest, int bitsPerSample, int channelCount, int sampleRate, double quality)
		{
			_audioDest = audioDest;
			_lwcdfDest = lwcdfDest;
			channels = channelCount;
			samplerate = sampleRate;
			bitspersample = bitsPerSample;

			if (_audioDest != null && _audioDest.BitsPerSample > bitsPerSample)
				throw new Exception("audio parameters mismatch");
			if (_lwcdfDest != null && _lwcdfDest.BitsPerSample != bitsPerSample)
				throw new Exception("audio parameters mismatch");

			int quality_integer = (int)Math.Floor(quality);

			fft_analysis_string = new string[4] { "0100010", "0110010", "0111010", "0111110" };
			bool[] quality_auto_fft32_on = { false, false, false, true, true, true, true, true, true, true, true };
			double[] quality_noise_threshold_shifts = { 20, 16, 9, 6, 3, 0, -2.4, -4.8, -7.2, -9.6, -12 };
			double[] quality_signal_to_noise_ratio = { -18, -22, -23.5, -23.5, -23.5, -25, -28, -31, -34, -37, -40 };
			double[] quality_dynamic_minimum_bits_to_keep = { 2.5, 2.75, 3.00, 3.25, 3.50, 3.75, 4.0, 4.25, 4.5, 4.75, 5.00 };
			double[] quality_maximum_clips_per_channel = { 3, 3, 3, 3, 2, 1, 0, 0, 0, 0, 0 };

			this_analysis_number = 2;
			impulse = quality_auto_fft32_on[quality_integer];
			linkchannels = false;
			noise_threshold_shift = Math.Round(interpolate_param(quality_noise_threshold_shifts, quality) * 1000) / 1000;
			snr_value = Math.Round(interpolate_param(quality_signal_to_noise_ratio, quality) * 1000) / 1000;
			dynamic_minimum_bits_to_keep = Math.Round(interpolate_param(quality_dynamic_minimum_bits_to_keep, quality) * 1000) / 1000;
			maximum_clips_per_channel = (int)Math.Round(interpolate_param(quality_maximum_clips_per_channel, quality));
			scaling_factor = 1.0;
			shaping_factor = Math.Min(1, quality / 10);
			shaping_is_on = shaping_factor > 0;
		}

		public void Close()
		{
			if (next_codec_block_size > 0)
			{
				shift_codec_blocks();  
				if (samplesInBuffer > 0)
					Array.Copy(sampleBuffer, 0, rotating_blocks_ptr[3], 0, samplesInBuffer * channels);
				next_codec_block_size = samplesInBuffer;
				process_this_codec_block();
				if (next_codec_block_size > 0)
				{
					samplesWritten += next_codec_block_size;
					shift_codec_blocks();
					process_this_codec_block();
				}
			}
			if (_audioDest != null) _audioDest.Close();
			if (_lwcdfDest != null) _lwcdfDest.Close();
		}

		public void Write(int[,] buff, uint sampleCount)
		{
			if (!initialized)
				Initialize();

			long pos = 0;
			while (sampleCount + samplesInBuffer > codec_block_size)
			{
				shift_codec_blocks(); // next_codec_block_size is now zero
				if (samplesInBuffer > 0)
					Array.Copy(sampleBuffer, 0, rotating_blocks_ptr[3], 0, samplesInBuffer * channels);
				Array.Copy(buff, pos * channels, rotating_blocks_ptr[3], samplesInBuffer * channels, (codec_block_size - samplesInBuffer) * channels);
				next_codec_block_size = codec_block_size;
				pos += codec_block_size - samplesInBuffer;
				sampleCount -= codec_block_size - (uint)samplesInBuffer;
				samplesInBuffer = 0;
				if (samplesWritten > 0)
					process_this_codec_block();
				samplesWritten += next_codec_block_size;
			}
			if (sampleCount > 0)
			{
				Array.Copy(buff, pos * channels, sampleBuffer, samplesInBuffer * channels, sampleCount * channels);
				samplesInBuffer += (int) sampleCount;
			}
		}

		public bool SetTags(NameValueCollection tags)
		{
			if (_audioDest != null) _audioDest.SetTags(tags);
			if (_lwcdfDest != null) _lwcdfDest.SetTags(tags);
			return true;
		}

		public string Path { get { return _audioDest.Path; } }

		public void Delete()
		{
			if (_audioDest != null) _audioDest.Delete();
			if (_lwcdfDest != null) _lwcdfDest.Delete();
		}

		public long FinalSampleCount
		{
			set
			{
				if (_audioDest != null) _audioDest.FinalSampleCount = value;
				if (_lwcdfDest != null) _lwcdfDest.FinalSampleCount = value;
			}
		}

		public long BlockSize
		{
			set { }
		}

		public int BitsPerSample
		{
			get { return bitspersample; }
		}

		public int OverallBitsRemoved
		{
			get { return overall_bits_removed; }
		}

		public int BlocksProcessed
		{
			get { return blocks_processed; }
		}

		public int SamplesProcessed
		{
			get { return (blocks_processed - 1) * codec_block_size + this_codec_block_size; }
		}

		public int Analysis
		{
			get { return this_analysis_number; }
			set { this_analysis_number = value; }
		}

		public bool Impulse
		{
			get { return impulse; }
			set { impulse = value; }
		}

		public bool LinkChannels
		{
			get { return linkchannels; }
			set
			{
				throw new Exception("Unsupported");
				//linkchannels = value; 
			}
		}

		public double ScalingFactor
		{
			get { return scaling_factor; }
			set
			{
				if (scaling_factor <= 0 || scaling_factor > 8)
					throw new Exception("Invalid scaling_factor");
				scaling_factor = value;
			}
		}

		public double ShapingFactor
		{
			get { return shaping_factor; }
			set
			{
				if (shaping_factor < 0 || shaping_factor > 1)
					throw new Exception("Invalid shaping_factor");
				shaping_factor = value;
				shaping_is_on = shaping_factor > 0;
			}
		}

		public float FrequencyLimit
		{
			get { return frequency_limit; }
			set
			{
				if (frequency_limit <= 14470.3125F || frequency_limit > 48000F)
					throw new Exception("Invalid frequency_limit");
				frequency_limit = value;
			}
		}
		#endregion

		#region Private Methods

		static double fastlog2(double x)
		{
			return Math.Log(x, 2);
		}

		static double fastsqr(double x)
		{
			return x * x;
		}

		static double interpolate_param(double[] param, double quality)
		{
			if (quality >= quality_presets)
				return param[quality_presets];

			int quality_integer = (int)Math.Floor(quality);
			double quality_fraction = quality - quality_integer;
			return quality_fraction * (param[quality_integer + 1] - param[quality_integer]) + param[quality_integer];
		}

		//const double LN2 = 0.69314718055994530941; 
		const double lg2x20 = 6.020599913279623904274778;
		const int precalc_analyses = 6;
		const int spread_freqs = 8;
		const int codec_block_size = 512;
		const int maxblocksize = 4096;
		const int maxchannels = 8;
		const int MaxFFTBitLength = 13;
		const int shaping_length = 4;
		const short static_minimum_bits_to_keep = 6;
		const double SP = 1.0;
		const double SW = 0.7071067811865475;
		const int skewing_amplitude = 36;
		const int threshold_index_spread = 64;
		const int threshold_index_spread_max = threshold_index_spread * 256;
		const int quality_presets = 10;

		void Initialize()
		{
			fft_bit_length = new short[precalc_analyses] { 5, 6, 7, 8, 9, 10 };
			frequency_limits = new float[spread_freqs + 1] { 20F, 1378.125F, 3445.3125F, 5512.5F, 8268.75F, 10335.9375F, 12403.125F, 14470.3125F, 16000F };//537.5F);
			frequency_limits[spread_freqs] = frequency_limit;

			float[,] reference_threshold = {
				{ 0.855049F,1.647705F,2.586367F,3.570348F,4.566311F,5.565302F, 6.565059F, 7.565002F, 8.564982F, 9.564947F,10.564983F,11.564966F,12.564990F,13.564963F,14.564963F,15.564953F,16.564938F,17.564932F,18.564946F,19.564977F,20.564993F,21.564943F,22.564971F,23.564972F,24.564971F,25.564942F,26.564973F,27.564952F,28.564934F,29.564964F,30.565000F,31.564957F},
				{ 1.365595F,2.158121F,3.096140F,4.079761F,5.075650F,6.074656F, 7.074348F, 8.074301F, 9.074286F,10.074265F,11.074260F,12.074282F,13.074297F,14.074298F,15.074254F,16.074310F,17.074259F,18.074278F,19.074298F,20.074277F,21.074247F,22.074304F,23.074269F,24.074309F,25.074261F,26.074301F,27.074316F,28.074259F,29.074281F,30.074272F,31.074257F,32.074265F},
				{ 1.870837F,2.663372F,3.600974F,4.584524F,5.580366F,6.579303F, 7.579018F, 8.578990F, 9.578999F,10.578921F,11.578967F,12.578961F,13.578965F,14.578950F,15.578952F,16.578964F,17.578957F,18.578957F,19.578943F,20.578949F,21.578953F,22.578919F,23.578966F,24.578940F,25.578934F,26.578952F,27.578921F,28.578943F,29.578934F,30.578939F,31.578957F,32.578939F},
				{ 2.373478F,3.165937F,4.103403F,5.086839F,6.082678F,7.081625F, 8.081372F, 9.081305F,10.081304F,11.081288F,12.081279F,13.081282F,14.081269F,15.081277F,16.081299F,17.081315F,18.081269F,19.081260F,20.081285F,21.081299F,22.081287F,23.081292F,24.081301F,25.081300F,26.081281F,27.081274F,28.081268F,29.081285F,30.081311F,31.081289F,32.081297F,33.081284F},
				{ 2.874786F,3.667300F,4.604600F,5.588036F,6.583839F,7.582784F, 8.582545F, 9.582496F,10.582470F,11.582446F,12.582467F,13.582453F,14.582474F,15.582478F,16.582438F,17.582437F,18.582458F,19.582484F,20.582439F,21.582456F,22.582444F,23.582433F,24.582435F,25.582454F,26.582425F,27.582443F,28.582443F,29.582451F,30.582450F,31.582509F,32.582402F,33.582447F},
				{ 3.375442F,4.167948F,5.105193F,6.088653F,7.084443F,8.083383F, 9.083136F,10.083083F,11.083088F,12.083042F,13.083045F,14.083021F,15.083045F,16.083047F,17.083032F,18.083045F,19.083051F,20.083031F,21.083038F,22.083041F,23.083036F,24.083033F,25.083059F,26.083044F,27.083043F,28.083031F,29.083039F,30.083038F,31.083027F,32.083062F,33.083057F,34.083017F},
				{ 3.875760F,4.668262F,5.605555F,6.588945F,7.584770F,8.583679F, 9.583467F,10.583333F,11.583352F,12.583357F,13.583353F,14.583342F,15.583322F,16.583328F,17.583336F,18.583304F,19.583327F,20.583344F,21.583317F,22.583326F,23.583307F,24.583321F,25.583300F,26.583365F,27.583308F,28.583330F,29.583333F,30.583336F,31.583339F,32.583320F,33.583304F,34.583342F},
				{ 4.375942F,5.168419F,6.105638F,7.089092F,8.084885F,9.083830F,10.083552F,11.083465F,12.083491F,13.083485F,14.083446F,15.083493F,16.083486F,17.083496F,18.083485F,19.083490F,20.083477F,21.083474F,22.083482F,23.083483F,24.083463F,25.083500F,26.083483F,27.083467F,28.083480F,29.083465F,30.083478F,31.083495F,32.083484F,33.083481F,34.083507F,35.083478F},
				{ 4.876022F,5.668487F,6.605748F,7.589148F,8.584943F,9.583902F,10.583678F,11.583589F,12.583571F,13.583568F,14.583551F,15.583574F,16.583541F,17.583544F,18.583555F,19.583554F,20.583574F,21.583577F,22.583531F,23.583560F,24.583548F,25.583570F,26.583561F,27.583581F,28.583552F,29.583540F,30.583564F,31.583530F,32.583546F,33.583585F,34.583557F,35.583567F}
			};

			if (samplerate > 46050)
			{
				shaping_a = /* order_4_48000_a */ new double[4] { +0.90300, +0.01160, -0.58530, -0.25710 };
				shaping_b = /* order_4_48000_b */ new double[4] { -2.23740, +0.73390, +0.12510, +0.60330 };
			}
			else
			{
				shaping_a = /* order_4_44100_a */ new double[4] { +1.05870, +0.06760, -0.60540, -0.27380 };
				shaping_b = /* order_4_44100_b */ new double[4] { -2.20610, +0.47070, +0.25340, +0.62130 };
			}

			fifo = new double[shaping_length + maxblocksize];

			window_function = new float[1 << (MaxFFTBitLength + 1)];
			short bits_in_block_size = (short)Math.Floor(fastlog2(codec_block_size));
			for (int i = 0; i < precalc_analyses; i++)
				fft_bit_length[i] += (short)(bits_in_block_size - 9);
			if (frequency_limits[spread_freqs] > samplerate / 2)
				frequency_limits[spread_freqs] = samplerate / 2;
			fill_fft_lookup_block = new int[maxblocksize * 4];
			fill_fft_lookup_offset = new int[maxblocksize * 4];
			for (int i = 0; i < codec_block_size * 4; i++)
			{
				fill_fft_lookup_block[i] = i / codec_block_size;
				fill_fft_lookup_offset[i] = i % codec_block_size;
			}
			saved_fft_results = new fft_results_rec[maxchannels, precalc_analyses];
			for (int i = 0; i < maxchannels; i++)
				for (int j = 0; j < precalc_analyses; j++)
					saved_fft_results[i, j].start = -1;
			clipped_samples = 0;
			this_max_sample = (1 << (bitspersample - 1)) - 1;
			this_min_sample = 0 - (1 << (bitspersample - 1));
			for (int this_fft_bit_length = 1; this_fft_bit_length <= MaxFFTBitLength; this_fft_bit_length++)
			{
				int this_fft_length = 1 << this_fft_bit_length;
				// Generate window_function lookup table for each fft_length
				for (int i = 0; i < this_fft_length; i++)
					window_function[this_fft_length + i] = (float)
						(0.5 * (1 - Math.Cos(((i + 0.5) * 2 * Math.PI / this_fft_length))) * scaling_factor);
			}

			double sf = 1.0;
			for (int i = 0; i < shaping_length; i++)
			{
				sf *= shaping_factor;
				shaping_a[i] *= sf;
				shaping_b[i] *= sf;
			}
			static_maximum_bits_to_remove = (short)(bitspersample - static_minimum_bits_to_keep);
			double lfb = Math.Log10(frequency_limits[0]); // 20Hz lower limit for skewing;
			double mfb = Math.Log10(frequency_limits[2]); // 3445.3125Hz upper limit for skewing;
			double dfb = mfb - lfb;						  // skewing range;

			fft_a = new double[2 << MaxFFTBitLength];
			for (int i = 0; i < 1 << MaxFFTBitLength; i++)
			{
				fft_a[2 * i] = Math.Cos(-i * Math.PI / (1 << MaxFFTBitLength));
				fft_a[2 * i + 1] = Math.Sin(-i * Math.PI / (1 << MaxFFTBitLength));
			}

			reversedbits = new int[1 << MaxFFTBitLength];
			reversedbits[0] = 0;
			int rb1 = 1 << (MaxFFTBitLength - 1);
			reversedbits[1] = rb1;
			for (int i = 1; i < 1 << (MaxFFTBitLength - 1); i++)
			{
				int rb2n = reversedbits[i] >> 1;
				reversedbits[i << 1] = rb2n;
				reversedbits[(i << 1) + 1] = rb2n | rb1;
			}

			fft_array = new double[1 << (MaxFFTBitLength + 1)];
			fft_result = new double[1 << MaxFFTBitLength];
			rotating_blocks_ptr = new int[4][,];
			sampleBuffer = new int[codec_block_size, channels];
			channel_recs = new channel_rec[channels];
			analysis_recs = new analyzis_rec[precalc_analyses];

			for (int analysis_number = 0; analysis_number < precalc_analyses; analysis_number++)
			{
				int this_fft_bit_length = fft_bit_length[analysis_number];
				//analyzis_rec analyzis = analysis_recs[analysis_number];

				analysis_recs[analysis_number].spreading_averages_int = new int[1 << (MaxFFTBitLength)];
				analysis_recs[analysis_number].spreading_averages_rem = new float[1 << (MaxFFTBitLength)];
				analysis_recs[analysis_number].spreading_averages_rec = new float[1 << (MaxFFTBitLength)];
				analysis_recs[analysis_number].skewing_function = new float[1 << (MaxFFTBitLength)];
				analysis_recs[analysis_number].threshold_index = new byte[threshold_index_spread_max];

				analysis_recs[analysis_number].end_overlap_length = Math.Min(1 << (this_fft_bit_length - 1), codec_block_size);
				analysis_recs[analysis_number].actual_analysis_blocks_start = -analysis_recs[analysis_number].end_overlap_length;
				int total_overlap_length = codec_block_size + analysis_recs[analysis_number].end_overlap_length * 2 - (1 << this_fft_bit_length);
				analysis_recs[analysis_number].fft_underlap_length = 1 << (this_fft_bit_length - 1);
				analysis_recs[analysis_number].analysis_blocks = total_overlap_length >> (this_fft_bit_length - 1);
				//(int) Math.Max(0, Math.Floor(total_overlap_length / analysis_recs[analysis_number].fft_underlap_length));
				analysis_recs[analysis_number].fft_underlap_length = total_overlap_length * 1.0 / Math.Max(1, analysis_recs[analysis_number].analysis_blocks);

				// Calculate actual analysis_time values for fft_lengths
				analysis_recs[analysis_number].bin_width = samplerate * 1.0 / (1 << this_fft_bit_length);
				analysis_recs[analysis_number].bin_time = 1.0 / analysis_recs[analysis_number].bin_width;

				// Calculate which FFT bin corresponds to the low frequency limit
				analysis_recs[analysis_number].lo_bins =
					(int)Math.Max(1, Math.Round(frequency_limits[0] * analysis_recs[analysis_number].bin_time) - 1);
				analysis_recs[analysis_number].hi_bins =
					(int)Math.Max(1, Math.Round(frequency_limits[spread_freqs] * analysis_recs[analysis_number].bin_time) - 1);
				if (analysis_recs[analysis_number].hi_bins > (1 << (this_fft_bit_length - 1)) - 1)
					throw new Exception("frequency too high");
				double f_lfb = analysis_recs[analysis_number].lo_bins * analysis_recs[analysis_number].bin_width;
				double f_hfb = analysis_recs[analysis_number].hi_bins * analysis_recs[analysis_number].bin_width;
				double f_dfb = f_hfb - f_lfb;
				for (int i = analysis_recs[analysis_number].lo_bins; i <= analysis_recs[analysis_number].hi_bins; i++)
				{
					double f_tfb = i * analysis_recs[analysis_number].bin_width;
					double sp = 1 + Math.Pow(Math.Min(f_dfb, f_tfb - f_lfb) / f_dfb, SP) * SW;
					int sp_int = (int)Math.Floor(sp - 1);
					double sp_rem = Math.Max(0, sp - sp_int - (1 - (sp_int & 1))) / 2.0;
					analysis_recs[analysis_number].spreading_averages_int[i] = sp_int;
					analysis_recs[analysis_number].spreading_averages_rem[i] = (float)sp_rem;
					analysis_recs[analysis_number].spreading_averages_rec[i] = (float)(1 / sp);
				}
				analysis_recs[analysis_number].max_bins = analysis_recs[analysis_number].hi_bins
					- (analysis_recs[analysis_number].spreading_averages_int[analysis_recs[analysis_number].hi_bins] & 0x7FFFFFFE);
				analysis_recs[analysis_number].num_bins = analysis_recs[analysis_number].max_bins - analysis_recs[analysis_number].lo_bins + 1;
				analysis_recs[analysis_number].skewing_function[0] = 0F;
				for (int i = 1; i <= analysis_recs[analysis_number].hi_bins + 1; i++)
				{
					double sa_tfb = Math.Log10(i * analysis_recs[analysis_number].bin_width);
					if (sa_tfb < mfb)
						analysis_recs[analysis_number].skewing_function[i] = (float)Math.Pow(10, (Math.Pow(Math.Sin(Math.PI / 2 * Math.Max(0, sa_tfb - lfb) / dfb), 0.75) - 1) * skewing_amplitude / 20);
					else
						analysis_recs[analysis_number].skewing_function[i] = 1F;
				}
				int last_filled = 0;
				for (byte sa_bit = 0; sa_bit < 32; sa_bit++)
				{
					double this_reference_threshold = (reference_threshold[fft_bit_length[analysis_number] - 5, sa_bit]) * threshold_index_spread * lg2x20;
					while (last_filled < this_reference_threshold)
						analysis_recs[analysis_number].threshold_index[last_filled++] = sa_bit;
				}
				while (last_filled < threshold_index_spread_max)
					analysis_recs[analysis_number].threshold_index[last_filled++] = 32; // ?? 31?
			} // calculating for each analysis_number
			for (int i = 0; i < 4; i++)
				rotating_blocks_ptr[i] = new int[codec_block_size, channels];
			btrd_codec_block = new int[codec_block_size, channels];
			corr_codec_block = new int[codec_block_size, channels];
			blocks_processed = 0;
			overall_bits_removed = 0;
			overall_bits_lost = 0;
			initialized = true;

			const uint fccFact = 0x74636166;
			string datestamp = DateTime.Now.ToString();
			string parameter_string = "--standard "; // !!!!!
			string factString = "lossyWAV " + version_string + " @ " + datestamp + ", " + parameter_string + "\r\n\0";
			if (_audioDest != null && _audioDest is WAVWriter) ((WAVWriter)_audioDest).WriteChunk(fccFact, new ASCIIEncoding().GetBytes(factString));
			if (_lwcdfDest != null && _lwcdfDest is WAVWriter) ((WAVWriter)_lwcdfDest).WriteChunk(fccFact, new ASCIIEncoding().GetBytes(factString));
			if (_audioDest != null) _audioDest.BlockSize = codec_block_size;
			if (_lwcdfDest != null) _lwcdfDest.BlockSize = codec_block_size * 2;
		}

		double fill_fft_input(int actual_analysis_block_start, int this_fft_length, int channel)
		{
			double this_fft_fill_rms = 0;
			int ff_n = 2 * codec_block_size + actual_analysis_block_start;
			for (int ff_i = 0; ff_i < this_fft_length; ff_i++)
			{
				int ff_j = ff_i + ff_n;
				double ff_m = rotating_blocks_ptr[fill_fft_lookup_block[ff_j]][fill_fft_lookup_offset[ff_j], channel];
				fft_array[ff_i] = ff_m * window_function[this_fft_length + ff_i];
				this_fft_fill_rms += ff_m * ff_m;
			}
			return Math.Sqrt(this_fft_fill_rms / this_fft_length);
		}

		double spread_complex(ref fft_results_rec this_fft_result, int analysis_number)
		{
			double sc_y = 0.0, sc_x;
			int sc_i = analysis_recs[analysis_number].lo_bins - 1;
			fft_result[sc_i] = Math.Sqrt(fastsqr(fft_array[sc_i * 2]) + fastsqr(fft_array[sc_i * 2 + 1])) * analysis_recs[analysis_number].skewing_function[sc_i];
			for (sc_i = analysis_recs[analysis_number].lo_bins; sc_i <= analysis_recs[analysis_number].hi_bins; sc_i++)
			{
				sc_x = Math.Sqrt(fastsqr(fft_array[sc_i * 2]) + fastsqr(fft_array[sc_i * 2 + 1])) * analysis_recs[analysis_number].skewing_function[sc_i];
				sc_y += sc_x;
				fft_result[sc_i] = sc_x;
			}
			sc_i = analysis_recs[analysis_number].hi_bins + 1;
			sc_x = Math.Sqrt(fastsqr(fft_array[sc_i * 2]) + fastsqr(fft_array[sc_i * 2 + 1])) * analysis_recs[analysis_number].skewing_function[sc_i];
			//sc_y += sc_x;
			fft_result[sc_i] = sc_x;

			double snr_value_exp = Math.Pow(2, snr_value / lg2x20);
			double noise_threshold_shift_exp = Math.Pow(2, noise_threshold_shift / lg2x20);

			this_fft_result.savebin = (float)(sc_y / analysis_recs[analysis_number].num_bins * snr_value_exp);
			sc_y = 5623413251903.490803949510; // MaxDb { 10^(255/20)   }
			for (sc_i = analysis_recs[analysis_number].lo_bins; sc_i <= analysis_recs[analysis_number].max_bins; sc_i++)
			{
				double sc_z = ((fft_result[sc_i - 1] + fft_result[sc_i + 1]) * analysis_recs[analysis_number].spreading_averages_rem[sc_i] + fft_result[sc_i]) * analysis_recs[analysis_number].spreading_averages_rec[sc_i];
				sc_y = Math.Min(sc_y, sc_z);
			}

			this_fft_result.sminbin = (float)(sc_y * noise_threshold_shift_exp);
			return lg2x20 * fastlog2(Math.Min(this_fft_result.savebin, this_fft_result.sminbin));
		}

		void remove_bits(int channel, short bits_to_remove_from_this_channel)
		{
			short min_bits_to_remove = 0;
			if (_audioDest != null && _audioDest.BitsPerSample < bitspersample)
				min_bits_to_remove = (short) (bitspersample - _audioDest.BitsPerSample);
			if (bits_to_remove_from_this_channel < min_bits_to_remove)
				bits_to_remove_from_this_channel = min_bits_to_remove;

			channel_recs[channel].bits_to_remove = bits_to_remove_from_this_channel;
			channel_recs[channel].bits_lost = 0;
			channel_recs[channel].clipped_samples = 0;

			while (bits_to_remove_from_this_channel >= 0)
			{
				short this_channel_clips = 0;
				int max_sample = (this_max_sample >> bits_to_remove_from_this_channel) << bits_to_remove_from_this_channel;
				int min_sample = (this_min_sample >> bits_to_remove_from_this_channel) << bits_to_remove_from_this_channel;
				if (shaping_is_on && bits_to_remove_from_this_channel > 0)
				{
					for (int i = 0; i < this_codec_block_size; i++)
					{
						int sample = rotating_blocks_ptr[2][i, channel];
						double wanted_temp =
							sample * scaling_factor / (1 << bits_to_remove_from_this_channel) +
							fifo[i + 3] * shaping_b[0] +
							fifo[i + 2] * shaping_b[1] +
							fifo[i + 1] * shaping_b[2] +
							fifo[i] * shaping_b[3];
						int output_temp = (int)Math.Round(wanted_temp);
						fifo[i + 4] = output_temp - wanted_temp -
							fifo[i + 3] * shaping_a[0] -
							fifo[i + 2] * shaping_a[1] -
							fifo[i + 1] * shaping_a[2] -
							fifo[i] * shaping_a[3];
						int new_sample = output_temp << bits_to_remove_from_this_channel;
						if (new_sample > max_sample)
						{
							new_sample = max_sample;
							this_channel_clips++;
						}
						else if (new_sample < min_sample)
						{
							new_sample = min_sample;
							this_channel_clips++;
						}
						btrd_codec_block[i, channel] = new_sample;
						corr_codec_block[i, channel] = sample - (int)Math.Round(new_sample / scaling_factor);
					}
				}
				else
				{
					for (int i = 0; i < this_codec_block_size; i++)
					{
						int sample = rotating_blocks_ptr[2][i, channel];
						int new_sample = ((int)Math.Round(sample * scaling_factor / (1 << bits_to_remove_from_this_channel)))
							<< bits_to_remove_from_this_channel;
						if (new_sample > max_sample)
						{
							new_sample = max_sample;
							this_channel_clips++;
						}
						else if (new_sample < min_sample)
						{
							new_sample = min_sample;
							this_channel_clips++;
						}
						btrd_codec_block[i, channel] = new_sample;
						corr_codec_block[i, channel] = sample - (int)Math.Round(new_sample / scaling_factor);
					}
				}
				channel_recs[channel].clipped_samples = this_channel_clips;
				if (this_channel_clips <= maximum_clips_per_channel)
					break;
				if (bits_to_remove_from_this_channel <= min_bits_to_remove)
					break;
				bits_to_remove_from_this_channel--;
				channel_recs[channel].bits_lost++;
				channel_recs[channel].bits_to_remove--;
			}
		}

		void process_this_codec_block()
		{
			short codec_block_dependent_bits_to_remove = (short)bitspersample;

			double min_codec_block_channel_rms = channel_recs[0].this_codec_block_rms;
			for (int channel = 0; channel < channels; channel++)
				min_codec_block_channel_rms = Math.Min(min_codec_block_channel_rms, channel_recs[channel].this_codec_block_rms);

			for (int channel = 0; channel < channels; channel++)
			{
				// if (linkchannels)...
				channel_recs[channel].this_codec_block_bits = channel_recs[channel].this_codec_block_rms;
			}
			for (int channel = 0; channel < channels; channel++)
			{
				fft_results_rec min_fft_result;

				min_fft_result.savebin = 999;
				min_fft_result.sminbin = 999;

				channel_recs[channel].maximum_bits_to_remove = Math.Max((short)0, (short)Math.Floor(channel_recs[channel].this_codec_block_bits - dynamic_minimum_bits_to_keep));
				channel_recs[channel].maximum_bits_to_remove = Math.Min(channel_recs[channel].maximum_bits_to_remove, static_maximum_bits_to_remove);

				if (channel_recs[channel].this_codec_block_bits == 0)
				{
					min_fft_result.start = -9999;
					min_fft_result.btr = static_maximum_bits_to_remove;
					min_fft_result.savebin = 0;
					min_fft_result.nminbin = 0; // ?? sminbin?
					min_fft_result.sminbin = -1; // ?? nminbin?
					min_fft_result.analysis = -1;
					for (int analysis_number = 0; analysis_number < precalc_analyses; analysis_number++)
						saved_fft_results[channel, analysis_number] = min_fft_result;
					min_fft_result.btr = 0;
				}
				else
				{
					fft_results_rec this_fft_result;

					this_fft_result.savebin = 0; // initializing structure to keep compiler happy. 
					this_fft_result.nminbin = 0; // No idea why it wasn't initialized in delphi code,
					this_fft_result.sminbin = -1; // and no idea if i initialized it right!!!
					this_fft_result.analysis = -1;
					this_fft_result.start = -9999;

					this_fft_result.btr = channel_recs[channel].maximum_bits_to_remove;
					min_fft_result.btr = this_fft_result.btr;
					for (short analysis_number = 0; analysis_number < precalc_analyses; analysis_number++)
					{
						if (((analysis_number == 0 && impulse) || fft_analysis_string[this_analysis_number - 2][analysis_number] == '1')
							&& (1 << fft_bit_length[analysis_number] <= codec_block_size * 2))
						{
							short this_fft_bit_length = fft_bit_length[analysis_number];
							int this_fft_length = 1 << this_fft_bit_length;

							this_fft_result.analysis = analysis_number;
							this_fft_result.nminbin = -1;
							int neg_codec_block_start = -(last_codec_block_size + prev_codec_block_size);
							int pos_codec_block_start = this_codec_block_size + next_codec_block_size - this_fft_length;
							int this_actual_analysis_blocks_start = analysis_recs[analysis_number].actual_analysis_blocks_start;
							double this_fft_underlap_length = analysis_recs[analysis_number].fft_underlap_length;
							for (int analysis_block_number = 0; analysis_block_number <= analysis_recs[analysis_number].analysis_blocks; analysis_block_number++)
							{
								int actual_analysis_block_start = (int)Math.Floor(this_actual_analysis_blocks_start + analysis_block_number * this_fft_underlap_length);
								actual_analysis_block_start = Math.Max(actual_analysis_block_start, neg_codec_block_start);
								actual_analysis_block_start = Math.Min(actual_analysis_block_start, pos_codec_block_start);
								if (analysis_block_number == 0)
								{
									if (actual_analysis_block_start == saved_fft_results[channel, analysis_number].start)
									{
										this_fft_result = saved_fft_results[channel, analysis_number];
										if (this_fft_result.btr < min_fft_result.btr || min_fft_result.btr == -1)
											min_fft_result = this_fft_result;
									}
								}
								else
								{
									if (fill_fft_input(actual_analysis_block_start, this_fft_length, channel) > 0)
									{
										FFT_DReal(this_fft_bit_length - 1);
										double spread = spread_complex(ref this_fft_result, analysis_number);
										this_fft_result.btr = analysis_recs[analysis_number].threshold_index[(int)Math.Floor(Math.Max(0, spread) * threshold_index_spread)];
										if (this_fft_result.btr < min_fft_result.btr || min_fft_result.btr == -1)
											min_fft_result = this_fft_result;
									}
									if (analysis_block_number == analysis_recs[analysis_number].analysis_blocks)
									{
										this_fft_result.start = (short)(actual_analysis_block_start - codec_block_size);
										saved_fft_results[channel, analysis_number] = this_fft_result;
									}
								}
							}
						}
					}
					if (min_fft_result.sminbin > min_fft_result.savebin)
						fft_average = fft_average + 1;
					else
						fft_minimum = fft_minimum + 1;
				}
				min_fft_result.btr = Math.Max(min_fft_result.btr, (short)0);
				remove_bits(channel, min_fft_result.btr);
				codec_block_dependent_bits_to_remove = Math.Min(codec_block_dependent_bits_to_remove, channel_recs[channel].bits_to_remove);
			}

			for (int channel = 0; channel < channels; channel++)
			{
				// if (linkchannels)
				overall_bits_removed += channel_recs[channel].bits_to_remove;
				overall_bits_lost += channel_recs[channel].bits_lost;
				clipped_samples += channel_recs[channel].clipped_samples;
			}

			if (_audioDest != null)
			{
				if (_audioDest.BitsPerSample < bitspersample)
				{
					int sh = bitspersample - _audioDest.BitsPerSample;
					for (int i = 0; i < this_codec_block_size; i++)
						for (int c = 0; c < channels; c++)
							btrd_codec_block[i, c] >>= sh;
				}
				_audioDest.Write(btrd_codec_block, (uint)this_codec_block_size);
			}
			if (_lwcdfDest != null) _lwcdfDest.Write(corr_codec_block, (uint)this_codec_block_size);
		}

		void shift_codec_blocks()
		{
			int[,] sc_p = rotating_blocks_ptr[0];
			rotating_blocks_ptr[0] = rotating_blocks_ptr[1];
			rotating_blocks_ptr[1] = rotating_blocks_ptr[2];
			rotating_blocks_ptr[2] = rotating_blocks_ptr[3];
			rotating_blocks_ptr[3] = sc_p;

			prev_codec_block_size = last_codec_block_size;
			last_codec_block_size = this_codec_block_size;
			this_codec_block_size = next_codec_block_size;
			next_codec_block_size = 0;

			if (this_codec_block_size > 0)
			{
				blocks_processed++;
				for (int channel = 0; channel < channels; channel++)
				{
					double x = 0;
					for (int i = 0; i < this_codec_block_size; i++)
					{
						double s = rotating_blocks_ptr[2][i, channel];
						x += s * s;
					}
					channel_recs[channel].this_codec_block_rms = fastlog2(x / this_codec_block_size) / 2;
				}
			}
		}

		unsafe static void shuffle_in_place_dcomplex(double* fft_ptr, int* reversedbits_ptr, int bits)
		{
			for (int i = 0; i < 1 << bits; i++)
			{
				int j = reversedbits_ptr[i] >> (MaxFFTBitLength - bits);
				if (j > i)
				{
					double re = fft_ptr[2 * i];
					double im = fft_ptr[2 * i + 1];
					fft_ptr[2 * i] = fft_ptr[2 * j];
					fft_ptr[2 * i + 1] = fft_ptr[2 * j + 1];
					fft_ptr[2 * j] = re;
					fft_ptr[2 * j + 1] = im;
				}
			}
		}

		unsafe static void FFT_DComplex(double* fft_ptr, double* fft_a_ptr, int* reversedbits_ptr, int bits)
		{
			shuffle_in_place_dcomplex(fft_ptr, reversedbits_ptr, bits);

			for (int blockbitlen = 0; blockbitlen < bits; blockbitlen++)
			{
				int blockend = 1 << blockbitlen;
				int blocksize = 1 << (blockbitlen + 1);
				int i = 0;
				do
				{
					int j = 2 * i;
					for (int n = 0; n < blockend; n++)
					{
						int k = j + 2 * blockend;
						int a_index = n << (MaxFFTBitLength - blockbitlen + 1);
						double ARe = fft_a_ptr[a_index];
						double AIm = fft_a_ptr[a_index + 1];
						double KRe = fft_ptr[k];
						double KIm = fft_ptr[k + 1];
						double TRe = ARe * KRe - AIm * KIm;
						double TIm = ARe * KIm + AIm * KRe;
						fft_ptr[k] = fft_ptr[j] - TRe;
						fft_ptr[k + 1] = fft_ptr[j + 1] - TIm;
						fft_ptr[j] += TRe;
						fft_ptr[j + 1] += TIm;
						j += 2;
					}
					i += blocksize;
				} while (i < 1 << bits);
			}
		}

		unsafe void FFT_DReal(int bits)
		{
			fixed (double* fft_ptr = fft_array)
			fixed (double* fft_a_ptr = fft_a)
			fixed (int* reversedbits_ptr = reversedbits)
			{
				FFT_DComplex(fft_ptr, fft_a_ptr, reversedbits_ptr, bits);

				fft_ptr[(1 << bits) * 2] = fft_ptr[0] - fft_ptr[1];
				fft_ptr[(1 << bits) * 2 + 1] = 0;
				fft_ptr[0] = fft_ptr[0] + fft_ptr[1];
				fft_ptr[1] = 0;
				for (int j = 1; j < 1 << (bits - 1); j++)
				{
					int k = (1 << bits) - j;
					double ARe = fft_a_ptr[j << (MaxFFTBitLength - bits + 1)];
					double AIm = fft_a_ptr[(j << (MaxFFTBitLength - bits + 1)) + 1];
					double XRe = fft_ptr[j * 2];
					double XIm = fft_ptr[j * 2 + 1];
					double TRe = ARe * (XIm + fft_ptr[k * 2 + 1]) + AIm * (XRe - fft_ptr[k * 2]);
					double TIm = ARe * (XRe - fft_ptr[k * 2]) - AIm * (XIm + fft_ptr[k * 2 + 1]);
					fft_ptr[j * 2] = (XRe + fft_ptr[k * 2] + TRe) / 2;
					fft_ptr[j * 2 + 1] = (XIm - fft_ptr[k * 2 + 1] - TIm) / 2;
					fft_ptr[k * 2] = (XRe + fft_ptr[k * 2] - TRe) / 2;
					fft_ptr[k * 2 + 1] = -(XIm - fft_ptr[k * 2 + 1] + TIm) / 2;
				}
			}
		}
		#endregion

		#region Private fields
		IAudioDest _audioDest, _lwcdfDest;
		int channels, samplerate, bitspersample;
		short[] fft_bit_length;
		float[] frequency_limits;
		int[] fill_fft_lookup_block;
		int[] fill_fft_lookup_offset;
		fft_results_rec[,] saved_fft_results;
		float[] window_function;
		int clipped_samples;
		int this_max_sample, this_min_sample;
		double[] shaping_a;
		double[] shaping_b;
		double scaling_factor;
		double shaping_factor;
		short static_maximum_bits_to_remove;
		analyzis_rec[] analysis_recs;
		channel_rec[] channel_recs;
		int blocks_processed, overall_bits_removed, overall_bits_lost;
		int[][,] rotating_blocks_ptr;
		int[,] btrd_codec_block, corr_codec_block;
		int last_codec_block_size;
		int prev_codec_block_size;
		int this_codec_block_size;
		int next_codec_block_size;
		int [,] sampleBuffer;
		int samplesInBuffer, samplesWritten;
		int fft_minimum = 0, fft_average = 0;
		double[] fft_array;
		double[] fft_result;
		double[] fft_a;
		int[] reversedbits;
		string[] fft_analysis_string;
		double[] fifo;
		bool initialized = false;

		// settings
		int this_analysis_number;
		bool impulse;
		bool linkchannels;
		bool shaping_is_on;
		double snr_value;
		double noise_threshold_shift;
		double dynamic_minimum_bits_to_keep;
		int maximum_clips_per_channel;
		float frequency_limit = 16000F;
		#endregion
	}

	#region Private data types
	struct fft_results_rec
	{
		public float sminbin, savebin;
		public short btr, start, analysis, nminbin;
	}

	struct analyzis_rec
	{
		public double fft_underlap_length;
		public int end_overlap_length, central_block_start, half_number_of_blocks, actual_analysis_blocks_start, analysis_blocks;
		public double bin_width, bin_time;
		public int lo_bins, hi_bins, max_bins, num_bins;
		public int[] spreading_averages_int;
		public float[] spreading_averages_rem;
		public float[] spreading_averages_rec;
		public float[] skewing_function;
		public byte[] threshold_index;
	}

	struct channel_rec
	{
		public double this_codec_block_rms;
		public double this_codec_block_bits;
		public short maximum_bits_to_remove;
		public short bits_to_remove;
		public short bits_lost;
		public short clipped_samples;
	}

	public class LossyWAVReader : IAudioSource
	{
		public LossyWAVReader(IAudioSource audioSource, IAudioSource lwcdfSource)
		{
			_audioSource = audioSource;
			_lwcdfSource = lwcdfSource;

			if (_audioSource.Length != _lwcdfSource.Length)
				throw new Exception("Data not same length");
			if (_audioSource.BitsPerSample != _lwcdfSource.BitsPerSample
				|| _audioSource.ChannelCount != _lwcdfSource.ChannelCount
				|| _audioSource.SampleRate != _lwcdfSource.SampleRate)
				throw new Exception("FMT Data mismatch");

			scaling_factor = 1.0; // !!!! Need to read 'fact' chunks or tags here
		}

		public uint Read(int[,] buff, uint sampleCount)
		{
			if (sampleBuffer == null || sampleBuffer.Length < sampleCount)
				sampleBuffer = new int[sampleCount, _audioSource.ChannelCount];
			sampleCount = _audioSource.Read(buff, sampleCount);
			if (sampleCount != _lwcdfSource.Read(sampleBuffer, sampleCount))
				throw new Exception("size mismatch");
			for (uint i = 0; i < sampleCount; i++)
				for (int c = 0; c < _audioSource.ChannelCount; c++)
					buff[i,c] = (int)Math.Round(buff[i, c] / scaling_factor + sampleBuffer[i, c]);
			return sampleCount;
		}

		public ulong Length
		{
			get
			{
				return _audioSource.Length;
			}
		}

		public ulong Position
		{
			get
			{
				return _audioSource.Position;
			}
			set
			{
				_audioSource.Position = value;
				_lwcdfSource.Position = value;
			}
		}

		public ulong Remaining
		{
			get
			{
				return _audioSource.Remaining;
			}
		}

		public int BitsPerSample
		{
			get
			{
				return _audioSource.BitsPerSample;
			}
		}

		public int ChannelCount
		{
			get
			{
				return _audioSource.ChannelCount;
			}
		}

		public int SampleRate
		{
			get
			{
				return _audioSource.SampleRate;
			}
		}

		public NameValueCollection Tags
		{
			get
			{
				return _audioSource.Tags;
			}
			set
			{
				_audioSource.Tags = value;
			}
		}

		public string Path
		{
			get
			{
				return _audioSource.Path;
			}
		}


		public void Close()
		{
			_audioSource.Close();
			_lwcdfSource.Close();
		}

		IAudioSource _audioSource, _lwcdfSource;
		double scaling_factor;
		int[,] sampleBuffer;
	}

	#endregion
}
