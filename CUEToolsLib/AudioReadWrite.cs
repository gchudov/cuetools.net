using System;
using System.IO;
using FLACDotNet;
using WavPackDotNet;
using APEDotNet;
using ALACDotNet;
using LossyWAVDotNet;
using AudioCodecsDotNet;
using System.Collections.Generic;
using System.Collections.Specialized;

namespace CUEToolsLib {
	public static class AudioReadWrite {
		public static IAudioSource GetAudioSource(string path, Stream IO, string extension)
		{
			switch (extension)
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

		public static IAudioSource GetAudioSource(string path, Stream IO)
		{
			string extension = Path.GetExtension(path).ToLower();
			string filename = Path.GetFileNameWithoutExtension(path);
			string secondExtension = Path.GetExtension(filename).ToLower();
			if (secondExtension != ".lossy" && secondExtension != ".lwcdf")
				return GetAudioSource(path, IO, extension);

			string lossyPath = Path.Combine(Path.GetDirectoryName(path), Path.GetFileNameWithoutExtension(filename) + ".lossy" + extension);
			string lwcdfPath = Path.Combine(Path.GetDirectoryName(path), Path.GetFileNameWithoutExtension(filename) + ".lwcdf" + extension);
			IAudioSource lossySource = GetAudioSource(lossyPath, null, extension);
			IAudioSource lwcdfSource = GetAudioSource(lwcdfPath, null, extension);
			return new LossyWAVReader(lossySource, lwcdfSource);
		}

		public static IAudioDest GetAudioDest(string path, int bitsPerSample, int channelCount, int sampleRate, long finalSampleCount, string extension, CUEConfig config) {
			IAudioDest dest;
			switch (extension) {
				case ".wav":
					dest = new WAVWriter(path, bitsPerSample, channelCount, sampleRate); 
					break;
#if !MONO
				case ".flac":
					dest = new FLACWriter(path, bitsPerSample, channelCount, sampleRate);
					((FLACWriter)dest).CompressionLevel = (int)config.flacCompressionLevel;
					((FLACWriter)dest).Verify = config.flacVerify;
					break;
				case ".wv":
					dest = new WavPackWriter(path, bitsPerSample, channelCount, sampleRate);
					((WavPackWriter)dest).CompressionMode = config.wvCompressionMode;
					((WavPackWriter)dest).ExtraMode = config.wvExtraMode;
					((WavPackWriter)dest).MD5Sum = config.wvStoreMD5;
					break;
				case ".ape":
					dest = new APEWriter(path, bitsPerSample, channelCount, sampleRate);
					((APEWriter)dest).CompressionLevel = (int)config.apeCompressionLevel;
					break;
				case ".dummy":
					dest = new DummyWriter(path, bitsPerSample, channelCount, sampleRate); 
					break;
#endif
				default:
					throw new Exception("Unsupported audio type.");
			}
			dest.FinalSampleCount = finalSampleCount;
			return dest;
		}

		public static IAudioDest GetAudioDest(string path, long finalSampleCount, CUEConfig config)
		{
			string extension = Path.GetExtension(path).ToLower();
			string filename = Path.GetFileNameWithoutExtension(path);
			if (Path.GetExtension(filename).ToLower() != ".lossy")
			{
				int bitsPerSample = (config.detectHDCD && config.decodeHDCD) ? (config.decodeHDCDto24bit ? 24 : 20) : 16;
				return GetAudioDest(path, bitsPerSample, 2, 44100, finalSampleCount, extension, config);
			}

			string lwcdfPath = Path.Combine(Path.GetDirectoryName(path), Path.GetFileNameWithoutExtension(filename) + ".lwcdf" + extension);
			int destBitsPerSample = (config.detectHDCD && config.decodeHDCD) ? ((!config.decodeHDCDtoLW16 && config.decodeHDCDto24bit) ? 24 : 20) : 16;
			int lossyBitsPerSample = (config.detectHDCD && config.decodeHDCD && !config.decodeHDCDtoLW16) ? 24 : 16;
			IAudioDest lossyDest = GetAudioDest(path, lossyBitsPerSample, 2, 44100, finalSampleCount, extension, config);
			IAudioDest lwcdfDest = GetAudioDest(lwcdfPath, destBitsPerSample, 2, 44100, finalSampleCount, extension, config);
			return new LossyWAVWriter(lossyDest, lwcdfDest, destBitsPerSample, 2, 44100, config.lossyWAVQuality);
		}
	}
}