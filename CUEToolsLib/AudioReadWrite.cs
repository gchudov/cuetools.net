using System;
using System.IO;
using FLACDotNet;
using WavPackDotNet;
using APEDotNet;
using ALACDotNet;
using AudioCodecsDotNet;
using System.Collections.Generic;
using System.Collections.Specialized;

namespace CUEToolsLib {
	public static class AudioReadWrite {
		public static IAudioSource GetAudioSource(string path, Stream IO)
		{
			switch (Path.GetExtension(path).ToLower())
			{
				case ".wav":
					return new WAVReader(path, IO);
				case ".m4a":
					return new ALACReader(path, IO);
#if !MONO
				case ".flac":
					return new FLACReader(path, IO);
				case ".wv":
					return new WavPackReader(path, IO, null);
				case ".ape":
					return new APEReader(path, IO);
#endif
				default:
					throw new Exception("Unsupported audio type.");
			}
		}

		public static IAudioDest GetAudioDest(string path, int bitsPerSample, int channelCount, int sampleRate, long finalSampleCount) {
			IAudioDest dest;
			switch (Path.GetExtension(path).ToLower()) {
				case ".wav":
					dest = new WAVWriter(path, bitsPerSample, channelCount, sampleRate); break;
#if !MONO
				case ".flac":
					dest = new FLACWriter(path, bitsPerSample, channelCount, sampleRate); break;
				case ".wv":
					dest = new WavPackWriter(path, bitsPerSample, channelCount, sampleRate); break;
				case ".ape":
					dest = new APEWriter(path, bitsPerSample, channelCount, sampleRate); break;
				case ".dummy":
					dest = new DummyWriter(path, bitsPerSample, channelCount, sampleRate); break;
#endif
				default:
					throw new Exception("Unsupported audio type.");
			}
			dest.FinalSampleCount = finalSampleCount;
			return dest;
		}
	}
}