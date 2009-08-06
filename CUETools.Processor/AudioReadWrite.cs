using System;
using System.IO;
using CUETools.Codecs;
using CUETools.Codecs.ALAC;
#if !MONO
using CUETools.Codecs.FLAC;
using CUETools.Codecs.WavPack;
using CUETools.Codecs.APE;
using CUETools.Codecs.TTA;
#endif
using CUETools.Codecs.LossyWAV;
using System.Collections.Generic;
using System.Collections.Specialized;

namespace CUETools.Processor
{
	public static class AudioReadWrite {
		public static IAudioSource GetAudioSource(string path, Stream IO, string extension, CUEConfig config)
		{
			CUEToolsFormat fmt;
			if (!extension.StartsWith(".") || !config.formats.TryGetValue(extension.Substring(1), out fmt))
				throw new Exception("Unsupported audio type: " + path);
			CUEToolsUDC decoder;
			if (fmt.decoder == null || !config.decoders.TryGetValue(fmt.decoder, out decoder))
				throw new Exception("Unsupported audio type: " + path);
			switch (decoder.className)
			{
				case "WAVReader":
					return new WAVReader(path, IO);
				case "ALACReader":
					return new ALACReader(path, IO);
#if !MONO
				case "FLACReader":
					return new FLACReader(path, IO, config.disableAsm);
				case "WavPackReader":
					return new WavPackReader(path, IO, null);
				case "APEReader":
					return new APEReader(path, IO);
				case "TTAReader":
					return new TTAReader(path, IO);
#endif
				default:
					if (decoder.path == null)
						throw new Exception("Unsupported audio type: " + path);
					return new UserDefinedReader(path, IO, decoder.path, decoder.parameters);
			}
		}

		public static IAudioSource GetAudioSource(string path, Stream IO, CUEConfig config)
		{
			string extension = Path.GetExtension(path).ToLower();
			string filename = Path.GetFileNameWithoutExtension(path);
			string secondExtension = Path.GetExtension(filename).ToLower();
			if (secondExtension != ".lossy" && secondExtension != ".lwcdf")
				return GetAudioSource(path, IO, extension, config);

			string lossyPath = Path.Combine(Path.GetDirectoryName(path), Path.GetFileNameWithoutExtension(filename) + ".lossy" + extension);
			string lwcdfPath = Path.Combine(Path.GetDirectoryName(path), Path.GetFileNameWithoutExtension(filename) + ".lwcdf" + extension);
			IAudioSource lossySource = GetAudioSource(lossyPath, null, extension, config);
			IAudioSource lwcdfSource = null;
			try
			{
				lwcdfSource = GetAudioSource(lwcdfPath, null, extension, config);
			}
			catch
			{
				return lossySource;
			}
			return new LossyWAVReader(lossySource, lwcdfSource);
		}

		public static IAudioDest GetAudioDest(AudioEncoderType audioEncoderType, string path, int bitsPerSample, int channelCount, int sampleRate, long finalSampleCount, string extension, CUEConfig config) 
		{
			IAudioDest dest;
			if (audioEncoderType == AudioEncoderType.NoAudio || extension == ".dummy")
			{
				dest = new DummyWriter(path, bitsPerSample, channelCount, sampleRate);
				dest.FinalSampleCount = finalSampleCount;
				return dest;
			}
			CUEToolsFormat fmt;
			if (!extension.StartsWith(".") || !config.formats.TryGetValue(extension.Substring(1), out fmt))
				throw new Exception("Unsupported audio type: " + path);
			CUEToolsUDC encoder = audioEncoderType == AudioEncoderType.Lossless ? fmt.encoderLossless : 
				audioEncoderType == AudioEncoderType.Lossy ? fmt.encoderLossy :
				null;
			if (encoder == null)
				throw new Exception("Unsupported audio type: " + path);
			switch (encoder.className)
			{
				case "WAVWriter":
					dest = new WAVWriter(path, bitsPerSample, channelCount, sampleRate, null);
					break;
#if !MONO
				case "FLACWriter":
					dest = new FLACWriter(path, bitsPerSample, channelCount, sampleRate);
					((FLACWriter)dest).CompressionLevel = encoder.DefaultModeIndex;
					((FLACWriter)dest).Verify = config.flacVerify;
					((FLACWriter)dest).DisableAsm = config.disableAsm;
					break;
				case "WavPackWriter":
					dest = new WavPackWriter(path, bitsPerSample, channelCount, sampleRate);
					((WavPackWriter)dest).CompressionMode = encoder.DefaultModeIndex;
					((WavPackWriter)dest).ExtraMode = config.wvExtraMode;
					((WavPackWriter)dest).MD5Sum = config.wvStoreMD5;
					break;
				case "APEWriter":
					dest = new APEWriter(path, bitsPerSample, channelCount, sampleRate);
					((APEWriter)dest).CompressionLevel = encoder.DefaultModeIndex;
					break;
				case "TTAWriter":
					dest = new TTAWriter(path, bitsPerSample, channelCount, sampleRate);
					break;
#endif
				default:
					if (encoder.path == null)
						throw new Exception("Unsupported audio type: " + path);
					dest = new UserDefinedWriter(path, bitsPerSample, channelCount, sampleRate, null, encoder.path, encoder.parameters, encoder.default_mode);
					break;
			}
			dest.FinalSampleCount = finalSampleCount;
			return dest;
		}

		public static IAudioDest GetAudioDest(AudioEncoderType audioEncoderType, string path, long finalSampleCount, int bitsPerSample, int sampleRate, CUEConfig config)
		{
			string extension = Path.GetExtension(path).ToLower();
			string filename = Path.GetFileNameWithoutExtension(path);
			if (audioEncoderType == AudioEncoderType.NoAudio || audioEncoderType == AudioEncoderType.Lossless || Path.GetExtension(filename).ToLower() != ".lossy")
				return GetAudioDest(audioEncoderType, path, bitsPerSample, 2, sampleRate, finalSampleCount, extension, config);

			string lwcdfPath = Path.Combine(Path.GetDirectoryName(path), Path.GetFileNameWithoutExtension(filename) + ".lwcdf" + extension);
			int lossyBitsPerSample = (config.detectHDCD && config.decodeHDCD && !config.decodeHDCDtoLW16) ? 24 : 16;
			IAudioDest lossyDest = GetAudioDest(AudioEncoderType.Lossless, path, lossyBitsPerSample, 2, sampleRate, finalSampleCount, extension, config);
			IAudioDest lwcdfDest = audioEncoderType == AudioEncoderType.Hybrid ? GetAudioDest(AudioEncoderType.Lossless, lwcdfPath, bitsPerSample, 2, sampleRate, finalSampleCount, extension, config) : null;
			return new LossyWAVWriter(lossyDest, lwcdfDest, bitsPerSample, 2, sampleRate, config.lossyWAVQuality);
		}
	}
}