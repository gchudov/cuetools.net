namespace CUETools.Codecs
{
	public interface IAudioDest
	{
		AudioPCMConfig PCM { get; }
		string Path { get; }

		int CompressionLevel { get; set; }
		object Settings { get; set; }
		long FinalSampleCount { set; }
		long BlockSize { set; }
		long Padding { set; }

		void Write(AudioBuffer buffer);
		void Close();
		void Delete();
	}
}
