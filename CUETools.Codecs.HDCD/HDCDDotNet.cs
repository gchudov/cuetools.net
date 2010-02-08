using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
using CUETools.Codecs;

namespace HDCDDotNet
{
	/** \brief Statistics for decoding. */
	[StructLayout(LayoutKind.Sequential)]
	public struct hdcd_decoder_statistics
	{
		public UInt32 num_packets;
		/**<Total number of samples processed. */
		public bool enabled_peak_extend;
		/**< True if peak extend was enabled during decoding. */
		public bool disabled_peak_extend;
		/**< True if peak extend was disabled during decoding. */
		public double min_gain_adjustment;
		/**< Minimum dynamic gain used during decoding. */
		public double max_gain_adjustment;
		/**< Maximum dynamic gain used during decoding. */
		public bool enabled_transient_filter;
		/**< True if the transient filter was enabled during decoding. */
		public bool disabled_transient_filter;
		/**< True if the transient filter was disabled during decoding. */
	};

	public class HDCDDotNet : IAudioDest, IAudioFilter, IFormattable
	{
		public HDCDDotNet (int channels, int sample_rate, int output_bps, bool decode)
		{
			_decoder = IntPtr.Zero;
#if !MONO
			if (decode)
				_audioBuffer = new AudioBuffer(new AudioPCMConfig(output_bps, channels, 44100), 256);
			_decoder = hdcd_decoder_new();
			_channelCount = channels;
			_bitsPerSample = output_bps;
			if (_decoder == IntPtr.Zero)
				throw new Exception("Failed to initialize HDCD decoder.");
			bool b = true;
			b &= hdcd_decoder_set_num_channels(_decoder, (short) _channelCount);
			b &= hdcd_decoder_set_sample_rate(_decoder, sample_rate);
			b &= hdcd_decoder_set_input_bps(_decoder, 16);
			b &= hdcd_decoder_set_output_bps(_decoder, (short)_bitsPerSample);
			if (!b)
				throw new Exception("Failed to set up HDCD _decoder parameters.");
			_decoderCallback = decode ? new hdcd_decoder_write_callback(DecoderCallback) : null;
			_gch = GCHandle.Alloc(this);
			hdcd_decoder_init_status status = hdcd_decoder_init(_decoder, IntPtr.Zero, _decoderCallback, (IntPtr) _gch);
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
				return hdcd_decoder_detected_hdcd(_decoder);
#else
				throw new Exception("HDCD unsupported.");
#endif
			}
		}

		public AudioPCMConfig PCM
		{
			get { return AudioPCMConfig.RedBook; }
		}

		public long FinalSampleCount
		{
			set { throw new Exception("unsupported"); }
		}

		public long BlockSize
		{
			set { throw new Exception("unsupported"); }
		}

		public string Path
		{
			get { throw new Exception("unsupported"); }
		}

		public string Options
		{
			set { throw new Exception("unsupported"); }
		}

		public int CompressionLevel
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
			if (!hdcd_decoder_reset(_decoder))
#endif
				throw new Exception("error resetting decoder.");
		}

		public void GetStatistics(out hdcd_decoder_statistics stats)
		{
#if !MONO
			IntPtr _statsPtr = hdcd_decoder_get_statistics(_decoder);
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
			if (!hdcd_decoder_process_buffer_interleaved(_decoder, buff.Samples, buff.Length))
				throw new Exception("HDCD processing error.");
#endif
		}

		public void Flush ()
		{
#if !MONO
			if (!hdcd_decoder_flush_buffer(_decoder))
#endif
				throw new Exception("error flushing buffer.");
		}

		public void Close()
		{
#if !MONO
			if (_decoder != IntPtr.Zero)
				hdcd_decoder_delete(_decoder);
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

		/** \brief Return values from hdcd_decoder_init. */
		private enum hdcd_decoder_init_status
		{
			HDCD_DECODER_INIT_STATUS_OK = 0,
			/**< Initialization was successful. */
			HDCD_DECODER_INIT_STATUS_INVALID_STATE,
			/**< The _decoder was already initialised. */
			HDCD_DECODER_INIT_STATUS_MEMORY_ALOCATION_ERROR,
			/**< Initialization failed due to a memory allocation error. */
			HDCD_DECODER_INIT_STATUS_INVALID_NUM_CHANNELS,
			/**< Initialization failed because the configured number of channels was invalid. */
			HDCD_DECODER_INIT_STATUS_INVALID_SAMPLE_RATE,
			/**< Initialization failed because the configured sample rate was invalid. */
			HDCD_DECODER_INIT_STATUS_INVALID_INPUT_BPS,
			/**< Initialization failed because the configured input bits per sample was invalid. */
			HDCD_DECODER_INIT_STATUS_INVALID_OUTPUT_BPS
			/**< Initialization failed because the configured output bits per sample was invalid. */
		}

		/** \brief State values for a decoder.
		 *
		 * The decoder's state can be obtained by calling hdcd_decoder_get_state().
		 */
		private enum hdcd_decoder_state
		{
			HDCD_DECODER_STATE_UNINITIALISED = 1,
			/**< The decoder is uninitialised. */
			HDCD_DECODER_STATE_READY,
			/**< The decoder is initialised and ready to process data. */
			HDCD_DECODER_STATE_DIRTY,
			/**< The decoder has processed data, but has not yet been flushed. */
			HDCD_DECODER_STATE_FLUSHED,
			/**< The decoder has been flushed. */
			HDCD_DECODER_STATE_WRITE_ERROR,
			/**< An error was returned by the write callback. */
			HDCD_DECODER_STATE_MEMORY_ALOCATION_ERROR
			/**< Processing failed due to a memory allocation error. */
		};

		private AudioBuffer _audioBuffer;
		private IntPtr _decoder;
		private int[,] _inSampleBuffer;
		private int[,] _outSampleBuffer;
		private int _channelCount, _bitsPerSample;
		hdcd_decoder_write_callback _decoderCallback;
		IAudioDest _audioDest;
		GCHandle _gch;

		~HDCDDotNet()
		{
			Close();
		}

		private delegate bool hdcd_decoder_write_callback(IntPtr decoder, IntPtr buffer, int samples, IntPtr client_data);

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

#if !MONO
		[DllImport("hdcd.dll")] 
		private static extern IntPtr hdcd_decoder_new ();
		[DllImport("hdcd.dll")] 
		private static extern void hdcd_decoder_delete(IntPtr decoder);
		[DllImport("hdcd.dll")]
		private static extern hdcd_decoder_state hdcd_decoder_get_state(IntPtr decoder);
		[DllImport("hdcd.dll")] 
		private static extern bool hdcd_decoder_set_num_channels (IntPtr decoder, Int16 num_channels);
		//HDCD_API uint16_t hdcd_decoder_get_num_channels(const hdcd_decoder *const _decoder);
		[DllImport("hdcd.dll")] 
		private static extern bool hdcd_decoder_set_sample_rate(IntPtr decoder, Int32 sample_rate);
		//HDCD_API uint32_t hdcd_decoder_get_sample_rate(const hdcd_decoder *const _decoder);
		[DllImport("hdcd.dll")] 
		private static extern bool hdcd_decoder_set_input_bps(IntPtr decoder, Int16 input_bps);
		//HDCD_API uint16_t hdcd_decoder_get_input_bps(const hdcd_decoder *const _decoder);
		[DllImport("hdcd.dll")] 
		private static extern bool hdcd_decoder_set_output_bps(IntPtr decoder, Int16 output_bps);
		//HDCD_API uint16_t hdcd_decoder_get_output_bps(const hdcd_decoder *const _decoder);
		[DllImport("hdcd.dll")] 
		private static extern hdcd_decoder_init_status hdcd_decoder_init (IntPtr decoder, IntPtr unused, hdcd_decoder_write_callback write_callback, IntPtr client_data);
		[DllImport("hdcd.dll")] 
		private static extern bool hdcd_decoder_finish(IntPtr decoder);
		[DllImport("hdcd.dll")] 
		private static extern bool hdcd_decoder_process_buffer_interleaved(IntPtr decoder, [In, Out] int [,] input_buffer, Int32 samples);
		[DllImport("hdcd.dll")]
		private static extern bool hdcd_decoder_flush_buffer(IntPtr decoder);
		[DllImport("hdcd.dll")]
		private static extern bool hdcd_decoder_reset(IntPtr decoder);
		[DllImport("hdcd.dll")] 
		private static extern bool hdcd_decoder_detected_hdcd(IntPtr decoder);
		[DllImport("hdcd.dll")] 
		private static extern IntPtr hdcd_decoder_get_statistics(IntPtr decoder);
#endif
	}
}
