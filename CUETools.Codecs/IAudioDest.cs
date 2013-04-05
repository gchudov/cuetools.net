namespace CUETools.Codecs
{
	public interface IAudioDest
	{
		AudioPCMConfig PCM { get; }
		string Path { get; }

        AudioEncoderSettings Settings { get; set; }
		long FinalSampleCount { set; }

		void Write(AudioBuffer buffer);
		void Close();
		void Delete();
	}
}
