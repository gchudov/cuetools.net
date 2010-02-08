using System;
using System.IO;
using CUETools.CDImage;
using CUETools.Codecs;
using CUETools.Codecs.LossyWAV;

namespace CUETools.Processor
{
	public static class AudioReadWrite {
		public static IAudioSource GetAudioSource(string path, Stream IO, string extension, CUEConfig config)
		{
			if (extension == ".dummy")
			{
				using (StreamReader sr = (IO == null ? new StreamReader(path) : new StreamReader(IO))) {
					string slen = sr.ReadLine();
					long len;
					if (!long.TryParse(slen, out len))
						len = CDImageLayout.TimeFromString(slen) * 588;
					return new SilenceGenerator(len);
				}
			}
			if (extension == ".bin")
				return new WAVReader(path, IO, AudioPCMConfig.RedBook);
			CUEToolsFormat fmt;
			if (!extension.StartsWith(".") || !config.formats.TryGetValue(extension.Substring(1), out fmt))
				throw new Exception("Unsupported audio type: " + path);
			CUEToolsUDC decoder;
			if (fmt.decoder == null || !config.decoders.TryGetValue(fmt.decoder, out decoder))
				throw new Exception("Unsupported audio type: " + path);
			if (decoder.path != null)
				return new UserDefinedReader(path, IO, decoder.path, decoder.parameters);
			if (decoder.type == null)
				throw new Exception("Unsupported audio type: " + path);
			
			object src = Activator.CreateInstance(decoder.type, path, IO);
			if (src == null || !(src is IAudioSource))
				throw new Exception("Unsupported audio type: " + path + ": " + decoder.type.FullName);
			return src as IAudioSource;
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

		public static IAudioDest GetAudioDest(AudioEncoderType audioEncoderType, string path, AudioPCMConfig pcm, long finalSampleCount, int padding, string extension, CUEConfig config) 
		{
			IAudioDest dest;
			if (audioEncoderType == AudioEncoderType.NoAudio || extension == ".dummy")
			{
				dest = new DummyWriter(path, pcm);
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
			if (encoder.path != null)
				dest = new UserDefinedWriter(path, null, pcm, encoder.path, encoder.parameters, encoder.default_mode, padding);
			else if (encoder.type != null)
			{
				object o = Activator.CreateInstance(encoder.type, path, pcm);
				if (o == null || !(o is IAudioDest))
					throw new Exception("Unsupported audio type: " + path + ": " + encoder.type.FullName);
				dest = o as IAudioDest;
			} else
				throw new Exception("Unsupported audio type: " + path);
			dest.CompressionLevel = encoder.DefaultModeIndex;
			dest.FinalSampleCount = finalSampleCount;
			switch (encoder.type.FullName)
			{
				case "CUETools.Codecs.ALAC.ALACWriter":
					dest.Options = string.Format("--padding-length {0}", padding);
					break;
				case "CUETools.Codecs.FLAKE.FlakeWriter":
					dest.Options = string.Format("--padding-length {0}", padding);
					break;
				case "CUETools.Codecs.FlaCuda.FlaCudaWriter":
					dest.Options = string.Format("{0}{1}--padding-length {2} --cpu-threads {3}",
						config.flaCudaVerify ? "--verify " : "",
						config.flaCudaGPUOnly ? "--gpu-only " : "",
						padding,
						config.FlaCudaThreads ? 1 : 0);
					break;					
				case "CUETools.Codecs.FLAC.FLACWriter":
					dest.Options = string.Format("{0}{1}--padding-length {2}",
						config.disableAsm ? "--disable-asm " : "",
						config.flacVerify ? "--verify " : "",
						padding);
					break;
				case "CUETools.Codecs.WavPack.WavPackWriter":
					dest.Options = string.Format("{0}--extra-mode {1}",
						config.wvStoreMD5 ? "--md5 " : "",
						config.wvExtraMode);
					break;
			}
			return dest;
		}

		public static IAudioDest GetAudioDest(AudioEncoderType audioEncoderType, string path, long finalSampleCount, int bitsPerSample, int sampleRate, int padding, CUEConfig config)
		{
			string extension = Path.GetExtension(path).ToLower();
			string filename = Path.GetFileNameWithoutExtension(path);
			AudioPCMConfig pcm = new AudioPCMConfig(bitsPerSample, 2, sampleRate);
			if (audioEncoderType == AudioEncoderType.NoAudio || audioEncoderType == AudioEncoderType.Lossless || Path.GetExtension(filename).ToLower() != ".lossy")
				return GetAudioDest(audioEncoderType, path, pcm, finalSampleCount, padding, extension, config);

			string lwcdfPath = Path.Combine(Path.GetDirectoryName(path), Path.GetFileNameWithoutExtension(filename) + ".lwcdf" + extension);
			AudioPCMConfig lossypcm = new AudioPCMConfig((config.detectHDCD && config.decodeHDCD && !config.decodeHDCDtoLW16) ? 24 : 16, 2, sampleRate);
			IAudioDest lossyDest = GetAudioDest(AudioEncoderType.Lossless, path, lossypcm, finalSampleCount, padding, extension, config);
			IAudioDest lwcdfDest = audioEncoderType == AudioEncoderType.Hybrid ? GetAudioDest(AudioEncoderType.Lossless, lwcdfPath, lossypcm, finalSampleCount, padding, extension, config) : null;
			return new LossyWAVWriter(lossyDest, lwcdfDest, config.lossyWAVQuality, pcm);
		}
	}
}