using System;
using System.IO;
using CUETools.CDImage;
using CUETools.Codecs;

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
					return new Codecs.NULL.AudioDecoder(len);
				}
			}
			if (extension == ".bin")
				return new Codecs.WAV.AudioDecoder(new Codecs.WAV.DecoderSettings(), path, IO, AudioPCMConfig.RedBook);
			CUEToolsFormat fmt;
			if (!extension.StartsWith(".") || !config.formats.TryGetValue(extension.Substring(1), out fmt))
				throw new Exception("Unsupported audio type: " + path);
			if (fmt.decoder == null)
				throw new Exception("Unsupported audio type: " + path);
            var settings = fmt.decoder.Settings.Clone();
            try
			{
				object src = Activator.CreateInstance(fmt.decoder.Settings.DecoderType, settings, path, IO);
				if (src == null || !(src is IAudioSource))
					throw new Exception("Unsupported audio type: " + path + ": " + fmt.decoder.Settings.DecoderType.FullName);
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
			return GetAudioSource(path, IO, extension, config);
		}

		public static IAudioDest GetAudioDest(AudioEncoderType audioEncoderType, string path, AudioPCMConfig pcm, long finalSampleCount, int padding, string extension, CUEConfig config)
		{
			IAudioDest dest;
            if (audioEncoderType == AudioEncoderType.NoAudio || extension == ".dummy")
                return new Codecs.NULL.AudioEncoder(path, new Codecs.WAV.EncoderSettings(pcm)) { FinalSampleCount = finalSampleCount };
			CUEToolsFormat fmt;
			if (!extension.StartsWith(".") || !config.formats.TryGetValue(extension.Substring(1), out fmt))
				throw new Exception("Unsupported audio type: " + path);
			AudioEncoderSettingsViewModel encoder = audioEncoderType == AudioEncoderType.Lossless ? fmt.encoderLossless :
				audioEncoderType == AudioEncoderType.Lossy ? fmt.encoderLossy :
				null;
			if (encoder == null)
				throw new Exception("Unsupported audio type: " + path);
            var settings = encoder.Settings.Clone();
            settings.PCM = pcm;
            settings.Padding = padding;
            object o;
            try
            {
                o = Activator.CreateInstance(settings.EncoderType, settings, path, null);
            }
            catch (System.Reflection.TargetInvocationException ex)
            {
                throw ex.InnerException;
            }
			if (o == null || !(o is IAudioDest))
				throw new Exception("Unsupported audio type: " + path + ": " + settings.EncoderType.FullName);
			dest = o as IAudioDest;
            dest.FinalSampleCount = finalSampleCount;
			return dest;
		}
	}
}