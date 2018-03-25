using System;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.HDCD
{
    public class HDCDDotNet : IAudioDest, IAudioFilter, IFormattable
	{
		public HDCDDotNet (int channels, int sample_rate, int output_bps, bool decode)
		{
			_decoder = IntPtr.Zero;
#if !MONO
			if (decode)
				_audioBuffer = new AudioBuffer(new AudioPCMConfig(output_bps, channels, 44100), 256);
			_decoder = HDCDDLL.hdcd_decoder_new();
			_channelCount = channels;
			_bitsPerSample = output_bps;
			if (_decoder == IntPtr.Zero)
				throw new Exception("Failed to initialize HDCD decoder.");
			bool b = true;
			b &= HDCDDLL.hdcd_decoder_set_num_channels(_decoder, (short) _channelCount);
			b &= HDCDDLL.hdcd_decoder_set_sample_rate(_decoder, sample_rate);
			b &= HDCDDLL.hdcd_decoder_set_input_bps(_decoder, 16);
			b &= HDCDDLL.hdcd_decoder_set_output_bps(_decoder, (short)_bitsPerSample);
			if (!b)
				throw new Exception("Failed to set up HDCD _decoder parameters.");
			_decoderCallback = decode ? new HDCDDLL.hdcd_decoder_write_callback(DecoderCallback) : null;
			_gch = GCHandle.Alloc(this);
			hdcd_decoder_init_status status = HDCDDLL.hdcd_decoder_init(_decoder, IntPtr.Zero, _decoderCallback, (IntPtr) _gch);
			switch (status)
			{
				case hdcd_decoder_init_status.HDCD_DECODER_INIT_STATUS_OK:
					break;
				case hdcd_decoder_init_status.HDCD_DECODER_INIT_STATUS_MEMORY_ALOCATION_ERROR:
					throw new Exception("Memory allocation error.");
				case hdcd_decoder_init_status.HDCD_DECODER_INIT_STATUS_INVALID_NUM_CHANNELS:
					throw new Exception("Invalid number of channels.");
				case hdcd_decoder_init_status.HDCD_DECODER_INIT_STATUS_INVALID_SAMPLE_RATE:
					throw new Exception("Invalid sample rate.");
				default:
					throw new Exception("Unknown error(" + status.ToString() + ").");
			}
#else
			throw new Exception("HDCD unsupported.");
#endif
		}

		public bool Detected
		{
			get
			{
#if !MONO
				return HDCDDLL.hdcd_decoder_detected_hdcd(_decoder);
#else
				throw new Exception("HDCD unsupported.");
#endif
			}
		}

		public long FinalSampleCount
		{
			set { throw new Exception("unsupported"); }
		}

		public string Path
		{
			get { throw new Exception("unsupported"); }
		}

        public IAudioEncoderSettings Settings
		{
			get { throw new Exception("unsupported"); }
			set { throw new Exception("unsupported"); }
		}

		public string ToString(string format, IFormatProvider formatProvider)
		{
			if (format == "f")
			{
				hdcd_decoder_statistics stats;
				GetStatistics(out stats);
				return string.Format(formatProvider, "peak extend: {0}, transient filter: {1}, gain: {2}",
					(stats.enabled_peak_extend ? (stats.disabled_peak_extend ? "some" : "yes") : "none"),
					(stats.enabled_transient_filter ? (stats.disabled_transient_filter ? "some" : "yes") : "none"),
					stats.min_gain_adjustment == stats.max_gain_adjustment ?
					(stats.min_gain_adjustment == 1.0 ? "none" : String.Format("{0:0.0}dB", (Math.Log10(stats.min_gain_adjustment) * 20))) :
					String.Format(formatProvider, "{0:0.0}dB..{1:0.0}dB", (Math.Log10(stats.min_gain_adjustment) * 20), (Math.Log10(stats.max_gain_adjustment) * 20))
					);
			}
			else
			{
				return Detected ? "HDCD detected" : "";
			}
		}

		public bool Decoding
		{
			get
			{
				return _decoderCallback != null;
			}
		}

		public int Channels
		{
			get
			{
				return _channelCount;
			}
		}

		public int BitsPerSample
		{
			get
			{
				return _bitsPerSample;
			}
		}

		public void Reset()
		{
#if !MONO
			if (!HDCDDLL.hdcd_decoder_reset(_decoder))
#endif
				throw new Exception("error resetting decoder.");
		}

		public void GetStatistics(out hdcd_decoder_statistics stats)
		{
#if !MONO
			IntPtr _statsPtr = HDCDDLL.hdcd_decoder_get_statistics(_decoder);
#else
			IntPtr _statsPtr = IntPtr.Zero;
#endif
			if (_statsPtr == IntPtr.Zero)
				throw new Exception("HDCD statistics error.");
			stats = (hdcd_decoder_statistics) Marshal.PtrToStructure(_statsPtr, typeof(hdcd_decoder_statistics));
		}

		public void Write(AudioBuffer buff)
		{
#if !MONO
			if (!HDCDDLL.hdcd_decoder_process_buffer_interleaved(_decoder, buff.Samples, buff.Length))
				throw new Exception("HDCD processing error.");
#endif
		}

		public void Flush ()
		{
#if !MONO
			if (!HDCDDLL.hdcd_decoder_flush_buffer(_decoder))
#endif
				throw new Exception("error flushing buffer.");
		}

		public void Close()
		{
#if !MONO
			if (_decoder != IntPtr.Zero)
                HDCDDLL.hdcd_decoder_delete(_decoder);
			_decoder = IntPtr.Zero;
			if (_gch.IsAllocated)
				_gch.Free();
#endif
		}

		public void Delete()
		{
			Close();
		}

		public IAudioDest AudioDest
		{
			get
			{
				return _audioDest;
			}
			set
			{
				//if (hdcd_decoder_get_state(_decoder) == hdcd_decoder_state.HDCD_DECODER_STATE_DIRTY) 
				//    Flush(); /* Flushing is currently buggy! Doesn't work twice, and can't continue after flush! */
				_audioDest = value;
			}
		}

		private AudioBuffer _audioBuffer;
		private IntPtr _decoder;
		//private int[,] _inSampleBuffer;
		private int[,] _outSampleBuffer;
		private int _channelCount, _bitsPerSample;
		HDCDDLL.hdcd_decoder_write_callback _decoderCallback;
		IAudioDest _audioDest;
		GCHandle _gch;

		~HDCDDotNet()
		{
			Close();
		}

		private unsafe bool Output(IntPtr buffer, int samples)
		{
			if (AudioDest == null)
				return true;

			if (_outSampleBuffer == null || _outSampleBuffer.GetLength(0) < samples)
				_outSampleBuffer = new int[samples, _channelCount];

			int loopCount = samples * _channelCount;
			int* pInSamples = (int*)buffer;
			fixed (int* pOutSamplesFixed = &_outSampleBuffer[0, 0])
			{
				int* pOutSamples = pOutSamplesFixed;
				for (int i = 0; i < loopCount; i++)
					*(pOutSamples++) = *(pInSamples++);
			}
			_audioBuffer.Prepare(_outSampleBuffer, samples);
			AudioDest.Write(_audioBuffer);
			return true;
		}

		private static unsafe bool DecoderCallback(IntPtr decoder, IntPtr buffer, int samples, IntPtr client_data)
		{
			GCHandle gch = (GCHandle)client_data;
			HDCDDotNet hdcd = (HDCDDotNet)gch.Target;
			return hdcd.Output(buffer, samples);
		}
	}
}
