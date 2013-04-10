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
			if (fmt.decoder == null)
				throw new Exception("Unsupported audio type: " + path);
			if (fmt.decoder.path != null)
				return new UserDefinedReader(path, IO, fmt.decoder.path, fmt.decoder.parameters);
			if (fmt.decoder.type == null)
				throw new Exception("Unsupported audio type: " + path);

			try
			{
				object src = Activator.CreateInstance(fmt.decoder.type, path, IO);
				if (src == null || !(src is IAudioSource))
					throw new Exception("Unsupported audio type: " + path + ": " + fmt.decoder.type.FullName);
				return src as IAudioSource;
			}
			catch (System.Reflection.TargetInvocationException ex)
			{
				if (ex.InnerException == null)
					throw ex;
				throw ex.InnerException;
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

		public static IAudioDest GetAudioDest(AudioEncoderType audioEncoderType, string path, AudioPCMConfig pcm, long finalSampleCount, int padding, string extension, CUEConfig config)
		{
			IAudioDest dest;
			if (audioEncoderType == AudioEncoderType.NoAudio || extension == ".dummy")
			{
				dest = new DummyWriter(path, new AudioEncoderSettings(pcm));
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
            var settings = encoder.settings.Clone();
            settings.PCM = pcm;
            settings.Padding = padding;
            settings.Validate();
			if (encoder.type != null)
			{
                object o = Activator.CreateInstance(encoder.type, path, settings);
				if (o == null || !(o is IAudioDest))
					throw new Exception("Unsupported audio type: " + path + ": " + encoder.type.FullName);
				dest = o as IAudioDest;
			}
			else
				throw new Exception("Unsupported audio type: " + path);
            dest.FinalSampleCount = finalSampleCount;
			return dest;
		}

        public static IAudioDest GetAudioDest(AudioEncoderType audioEncoderType, string path, long finalSampleCount, int padding, AudioPCMConfig pcm, CUEConfig config)
		{
			string extension = Path.GetExtension(path).ToLower();
			string filename = Path.GetFileNameWithoutExtension(path);
			if (audioEncoderType == AudioEncoderType.NoAudio || audioEncoderType == AudioEncoderType.Lossless || Path.GetExtension(filename).ToLower() != ".lossy")
				return GetAudioDest(audioEncoderType, path, pcm, finalSampleCount, padding, extension, config);

			string lwcdfPath = Path.Combine(Path.GetDirectoryName(path), Path.GetFileNameWithoutExtension(filename) + ".lwcdf" + extension);
			AudioPCMConfig lossypcm = new AudioPCMConfig((config.detectHDCD && config.decodeHDCD && !config.decodeHDCDtoLW16) ? 24 : 16, pcm.ChannelCount, pcm.SampleRate);
			IAudioDest lossyDest = GetAudioDest(AudioEncoderType.Lossless, path, lossypcm, finalSampleCount, padding, extension, config);
			IAudioDest lwcdfDest = audioEncoderType == AudioEncoderType.Hybrid ? GetAudioDest(AudioEncoderType.Lossless, lwcdfPath, lossypcm, finalSampleCount, padding, extension, config) : null;
			return new LossyWAVWriter(lossyDest, lwcdfDest, config.lossyWAVQuality, new AudioEncoderSettings(pcm));
		}
	}
}