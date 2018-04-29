using System;
using System.Collections.Generic;

namespace CUETools.Codecs
{
	public interface IAudioSource
	{
        IAudioDecoderSettings Settings { get; }

		AudioPCMConfig PCM { get; }
		string Path { get; }

        TimeSpan Duration { get; }
        long Length { get; }
		long Position { get; set; }
		long Remaining { get; }

		int Read(AudioBuffer buffer, int maxLength);
		void Close();
	}

    public interface IAudioTitle
    {
        List<TimeSpan> Chapters { get; }
        //IAudioSource Open { get; }
    }

    public interface IAudioContainer
    {
        List<IAudioTitle> AudioTitles { get; }
    }

    public class NoContainerAudioTitle : IAudioTitle
    {
        public NoContainerAudioTitle(IAudioSource source) { this.source = source; }
        public List<TimeSpan> Chapters => new List<TimeSpan> { TimeSpan.Zero, source.Duration };
        IAudioSource source;
    }

    public class NoContainer : IAudioContainer
    {
        public NoContainer(IAudioSource source) { this.source = source; }
        public List<IAudioTitle> AudioTitles => new List<IAudioTitle> { new NoContainerAudioTitle(source) };
        IAudioSource source;
    }
}
