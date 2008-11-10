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
		public static IAudioSource GetAudioSource(string path) {
			switch (Path.GetExtension(path).ToLower()) {
				case ".wav":
					return new WAVReader(path, null);
#if !MONO
				case ".flac":
					return new FLACReader(path, null);
				case ".wv":
					return new WavPackReader(path);
				case ".ape":
					return new APEReader(path);
				case ".m4a":
					return new ALACReader(path, null);
#endif
				default:
					throw new Exception("Unsupported audio type.");
			}
		}
		public static IAudioSource GetAudioSource(string path, Stream IO)
		{
			switch (Path.GetExtension(path).ToLower())
			{
				case ".wav":
					return new WAVReader(path, IO);
#if !MONO
				case ".flac":
					return new FLACReader(path, IO);
				case ".m4a":
					return new ALACReader(path, IO);
#endif
				default:
					throw new Exception("Unsupported audio type in archive.");
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

#if !MONO
	class FLACReader : IAudioSource {
		FLACDotNet.FLACReader _flacReader;
		int[,] _sampleBuffer;
		uint _bufferOffset, _bufferLength;

		public FLACReader(string path, Stream IO)
		{
			_flacReader = new FLACDotNet.FLACReader(path, IO);
			_bufferOffset = 0;
			_bufferLength = 0;
		}

		public void Close() {
			_flacReader.Close();
		}

		public NameValueCollection Tags
		{
			get { return _flacReader.Tags; }
			set { _flacReader.Tags = value; }
		}

		public bool UpdateTags(bool preserveTime)
		{
			return _flacReader.UpdateTags(preserveTime);
		}

		public ulong Length {
			get {
				return (ulong) _flacReader.Length;
			}
		}

		public ulong Remaining {
			get {
				return (ulong) _flacReader.Remaining + SamplesInBuffer;
			}
		}

		public ulong Position {
			get {
				return (ulong) _flacReader.Position - SamplesInBuffer;
			}
			set {
				_flacReader.Position = (long) value;
				_bufferOffset = 0;
				_bufferLength = 0;
			}
		}

		private uint SamplesInBuffer {
			get {
				return (uint) (_bufferLength - _bufferOffset);
			}
		}

		public int BitsPerSample {
			get {
				return _flacReader.BitsPerSample;
			}
		}

		public int ChannelCount {
			get {
				return _flacReader.ChannelCount;
			}
		}

		public int SampleRate {
			get {
				return _flacReader.SampleRate;
			}
		}

		public uint Read(int [,] buff, uint sampleCount) {
			if (_flacReader.BitsPerSample != 16) {
				throw new Exception("Reading is only supported for 16 bit sample depth.");
			}
			int chanCount = _flacReader.ChannelCount;
			uint copyCount;
			uint buffOffset = 0;
			uint samplesNeeded = sampleCount;

			while (samplesNeeded != 0) {
				if (SamplesInBuffer == 0) {
					_bufferOffset = 0;
					_bufferLength = (uint) _flacReader.Read(out _sampleBuffer);
				}

				copyCount = Math.Min(samplesNeeded, SamplesInBuffer);
				Array.Copy(_sampleBuffer, _bufferOffset * chanCount, buff, buffOffset * chanCount, copyCount * chanCount);

				samplesNeeded -= copyCount;
				buffOffset += copyCount;
				_bufferOffset += copyCount;
			}

			return sampleCount;
		}

		public string Path { get { return _flacReader.Path; } }
	}

	class FLACWriter : IAudioDest {
		FLACDotNet.FLACWriter _flacWriter;
		int _bitsPerSample;
		int _channelCount;
		int _sampleRate;

		public FLACWriter(string path, int bitsPerSample, int channelCount, int sampleRate) {
			if (bitsPerSample != 16 && bitsPerSample != 24) {
				throw new Exception("Bits per sample must be 16 or 24.");
			}
			_bitsPerSample = bitsPerSample;
			_channelCount = channelCount;
			_sampleRate = sampleRate;
			_flacWriter = new FLACDotNet.FLACWriter(path, bitsPerSample, channelCount, sampleRate);
		}

		public long FinalSampleCount {
			get {
				return _flacWriter.FinalSampleCount;
			}
			set {
				_flacWriter.FinalSampleCount = value;
			}
		}

		public int CompressionLevel {
			get {
				return _flacWriter.CompressionLevel;
			}
			set {
				_flacWriter.CompressionLevel = value;
			}
		}

		public bool Verify {
			get {
				return _flacWriter.Verify;
			}
			set {
				_flacWriter.Verify = value;
			}
		}

		public bool SetTags(NameValueCollection tags)
		{
			_flacWriter.SetTags (tags);
			return true;
		}

		public void Close() {
			_flacWriter.Close();
		}

		public void Write(int[,] buff, uint sampleCount)
		{
			_flacWriter.Write(buff, (int) sampleCount);
		}

		public string Path { get { return _flacWriter.Path; } }
	}
#endif

#if !MONO
	class APEReader : IAudioSource {
		APEDotNet.APEReader _apeReader;
		int[,] _sampleBuffer;
		uint _bufferOffset, _bufferLength;

		public APEReader(string path) {
			_apeReader = new APEDotNet.APEReader(path);
			_bufferOffset = 0;
			_bufferLength = 0;
		}

		public void Close() {
			_apeReader.Close();
		}

		public ulong Length {
			get {
				return (ulong) _apeReader.Length;
			}
		}

		public ulong Remaining {
			get {
				return (ulong) _apeReader.Remaining + SamplesInBuffer;
			}
		}

		public ulong Position {
			get {
				return (ulong) _apeReader.Position - SamplesInBuffer;
			}
			set {
				_apeReader.Position = (long) value;
				_bufferOffset = 0;
				_bufferLength = 0;
			}
		}

		private uint SamplesInBuffer {
			get {
				return (uint) (_bufferLength - _bufferOffset);
			}
		}

		public int BitsPerSample {
			get {
				return _apeReader.BitsPerSample;
			}
		}

		public int ChannelCount {
			get {
				return _apeReader.ChannelCount;
			}
		}

		public int SampleRate {
			get {
				return _apeReader.SampleRate;
			}
		}

		public NameValueCollection Tags
		{
			get { return _apeReader.Tags; }
			set { _apeReader.Tags = value; }
		}

		private unsafe void APESamplesToFLACSamples(int[,] inSamples, uint inSampleOffset,
			int[,] outSamples, uint outSampleOffset, uint sampleCount, int channelCount)
		{
			uint loopCount = sampleCount * (uint) channelCount;

			if ((inSamples.GetLength(0) - inSampleOffset < sampleCount) ||
				(outSamples.GetLength(0) - outSampleOffset < sampleCount))
			{
				throw new IndexOutOfRangeException();
			}

			fixed (int* pInSamplesFixed = &inSamples[inSampleOffset, 0]) {
				fixed (int * pOutSamplesFixed = &outSamples[outSampleOffset, 0]) {
					int* pInSamples = pInSamplesFixed;
					int* pOutSamples = pOutSamplesFixed;

					for (int i = 0; i < loopCount; i++) {
						*(pOutSamples++) = *(pInSamples++);
					}
				}
			}
		}

		public uint Read(int[,] buff, uint sampleCount) {
			if (_apeReader.BitsPerSample != 16) {
				throw new Exception("Reading is only supported for 16 bit sample depth.");
			}
			int chanCount = _apeReader.ChannelCount;
			uint samplesNeeded, copyCount, buffOffset;

			buffOffset = 0;
			samplesNeeded = sampleCount;

			while (samplesNeeded != 0) {
				if (SamplesInBuffer == 0) {
					_bufferOffset = 0;
					_bufferLength = (uint) _apeReader.Read(out _sampleBuffer);
				}

				copyCount = Math.Min(samplesNeeded, SamplesInBuffer);


				Array.Copy(_sampleBuffer, _bufferOffset * chanCount, buff, buffOffset * chanCount, copyCount * chanCount);
				//APESamplesToFLACSamples(_sampleBuffer, _bufferOffset, buff, buffOffset,
				//    copyCount, chanCount);

				samplesNeeded -= copyCount;
				buffOffset += copyCount;
				_bufferOffset += copyCount;
			}

			return sampleCount;
		}

		public string Path { get { return _apeReader.Path; } }
	}

	class APEWriter : IAudioDest
	{
		APEDotNet.APEWriter _apeWriter;
		byte[] _sampleBuffer;
		int _bitsPerSample;
		int _channelCount;
		int _sampleRate;
		int _blockAlign;

		public APEWriter(string path, int bitsPerSample, int channelCount, int sampleRate)
		{
			if (bitsPerSample != 16 && bitsPerSample != 24)
			{
				throw new Exception("Bits per sample must be 16 or 24.");
			}
			_bitsPerSample = bitsPerSample;
			_channelCount = channelCount;
			_sampleRate = sampleRate;
			_blockAlign = _channelCount * ((_bitsPerSample + 7) / 8);
			_apeWriter = new APEDotNet.APEWriter(path, bitsPerSample, channelCount, sampleRate);
		}

		public long FinalSampleCount
		{
			get { return _apeWriter.FinalSampleCount; }
			set { _apeWriter.FinalSampleCount = (int) value; }
		}
		public int CompressionLevel
		{
			get { return _apeWriter.CompressionLevel; }
			set { _apeWriter.CompressionLevel = value; }
		}
		public bool SetTags(NameValueCollection tags)
		{
			_apeWriter.SetTags(tags);
			return true;
		}
		public void Close()
		{
			_apeWriter.Close();
		}
		public void Write(int [,] buff, uint sampleCount)
		{
			if (_sampleBuffer == null || _sampleBuffer.Length < sampleCount * _blockAlign)
				_sampleBuffer = new byte[sampleCount * _blockAlign];
			AudioCodecsDotNet.AudioCodecsDotNet.FLACSamplesToBytes (buff, 0, _sampleBuffer, 0, sampleCount, _channelCount, _bitsPerSample);
			_apeWriter.Write(_sampleBuffer, sampleCount);
		}
		public string Path { get { return _apeWriter.Path; } }
	}
#endif

#if !MONO
	class WavPackReader : IAudioSource {
		WavPackDotNet.WavPackReader _wavPackReader;

		public WavPackReader(string path) {
			_wavPackReader = new WavPackDotNet.WavPackReader(path);
		}

		public void Close() {
			_wavPackReader.Close();
		}

		public ulong Length {
			get {
				return (ulong) _wavPackReader.Length;
			}
		}

		public ulong Remaining {
			get {
				return (ulong) _wavPackReader.Remaining;
			}
		}

		public ulong Position {
			get {
				return (ulong) _wavPackReader.Position;
			}
			set {
				_wavPackReader.Position = (int) value;
			}
		}

		public int BitsPerSample {
			get {
				return _wavPackReader.BitsPerSample;
			}
		}

		public int ChannelCount {
			get {
				return _wavPackReader.ChannelCount;
			}
		}

		public int SampleRate {
			get {
				return _wavPackReader.SampleRate;
			}
		}

		public NameValueCollection Tags
		{
			get { return _wavPackReader.Tags; }
			set { _wavPackReader.Tags = value; }
		}

		public uint Read(int[,] buff, uint sampleCount) {
			if (_wavPackReader.BitsPerSample != 16) {
				throw new Exception("Reading is only supported for 16 bit sample depth.");
			}
			_wavPackReader.Read(buff, (int) sampleCount);
			return sampleCount;
		}

		public string Path { get { return _wavPackReader.Path; } }
	}

	class WavPackWriter : IAudioDest {
		WavPackDotNet.WavPackWriter _wavPackWriter;
		int _bitsPerSample;
		int _channelCount;
		int _sampleRate;
		int _blockAlign;
		byte[] _sampleBuffer;

		public WavPackWriter(string path, int bitsPerSample, int channelCount, int sampleRate) {
			if (bitsPerSample != 16 && bitsPerSample != 24)
			{
				throw new Exception("Bits per sample must be 16 or 24.");
			}
			_bitsPerSample = bitsPerSample;
			_channelCount = channelCount;
			_sampleRate = sampleRate;
			_blockAlign = _channelCount * ((_bitsPerSample + 7) / 8);
			_wavPackWriter = new WavPackDotNet.WavPackWriter(path, bitsPerSample, channelCount, sampleRate);
		}

		public bool SetTags(NameValueCollection tags)
		{
			_wavPackWriter.SetTags(tags);
			return true;
		}

		public long FinalSampleCount {
			get {
				return _wavPackWriter.FinalSampleCount;
			}
			set {
				_wavPackWriter.FinalSampleCount = (int)value;
			}
		}

		public int CompressionMode {
			get {
				return _wavPackWriter.CompressionMode;
			}
			set {
				_wavPackWriter.CompressionMode = value;
			}
		}

		public int ExtraMode {
			get {
				return _wavPackWriter.ExtraMode;
			}
			set {
				_wavPackWriter.ExtraMode = value;
			}
		}

		public bool MD5Sum
		{
			get
			{
				return _wavPackWriter.MD5Sum;
			}
			set
			{
				_wavPackWriter.MD5Sum = value;
			}
		}

		public void Close() {
			_wavPackWriter.Close();
		}

		public void Write(int[,] sampleBuffer, uint sampleCount) {
			if (MD5Sum)
			{
				if (_sampleBuffer == null || _sampleBuffer.Length < sampleCount * _blockAlign)
					_sampleBuffer = new byte[sampleCount * _blockAlign];
				AudioCodecsDotNet.AudioCodecsDotNet.FLACSamplesToBytes(sampleBuffer, 0, _sampleBuffer, 0, sampleCount, _channelCount, _bitsPerSample);
				_wavPackWriter.UpdateHash(_sampleBuffer, (int) sampleCount * _blockAlign);
			}
			_wavPackWriter.Write(sampleBuffer, (int) sampleCount);
		}

		public string Path { get { return _wavPackWriter.Path; } }
	}
#endif
}