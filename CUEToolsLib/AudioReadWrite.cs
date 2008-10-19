using System;
using System.IO;
using FLACDotNet;
using WavPackDotNet;
using APEDotNet;
using System.Collections.Generic;
using System.Collections.Specialized;

namespace CUEToolsLib {
	public interface IAudioSource {
		uint Read(byte[] buff, uint sampleCount);
		ulong Length { get; }
		ulong Position { get; set; }
		NameValueCollection Tags { get; set; }
		ulong Remaining { get; }
		void Close();
		int BitsPerSample { get; }
		int ChannelCount { get; }
		int SampleRate { get; }
	}

	public interface IAudioDest {
		void Write(byte[] buff, uint sampleCount);
		bool SetTags(NameValueCollection tags);
		void Close();
		long FinalSampleCount { set; }
	}

	public static class AudioReadWrite {
		public static IAudioSource GetAudioSource(string path) {
			switch (Path.GetExtension(path).ToLower()) {
				case ".wav":
					return new WAVReader(path);
#if !MONO
				case ".flac":
					return new FLACReader(path);
				case ".wv":
					return new WavPackReader(path);
				case ".ape":
					return new APEReader(path);
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

	public class DummyWriter : IAudioDest {

		public DummyWriter (string path, int bitsPerSample, int channelCount, int sampleRate) {
		}

		public bool SetTags(NameValueCollection tags)
		{
			return false;
		}

		public void Close() {
		}

		public long FinalSampleCount {
			set {
			}
		}

		public void Write(byte[] buff, uint sampleCount) {
		}
	}

	public class WAVReader : IAudioSource {
		FileStream _fs;
		BinaryReader _br;
		ulong _dataOffset, _dataLen;
		ulong _samplePos, _sampleLen;
		int _bitsPerSample, _channelCount, _sampleRate, _blockAlign;
		bool _largeFile;

		public WAVReader(string path) {
			_fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
			_br = new BinaryReader(_fs);

			ParseHeaders();

			_sampleLen = _dataLen / (uint)_blockAlign;
			Position = 0;
		}

		public void Close() {
			_br.Close();

			_br = null;
			_fs = null;
		}

		private void ParseHeaders() {
			const long maxFileSize = 0x7FFFFFFEL;
			const uint fccRIFF = 0x46464952;
			const uint fccWAVE = 0x45564157;
			const uint fccFormat = 0x20746D66;
			const uint fccData = 0x61746164;

			uint lenRIFF;
			long fileEnd;
			bool foundFormat, foundData;

			if (_br.ReadUInt32() != fccRIFF) {
				throw new Exception("Not a valid RIFF file.");
			}

			lenRIFF = _br.ReadUInt32();
			fileEnd = (long)lenRIFF + 8;

			if (_br.ReadUInt32() != fccWAVE) {
				throw new Exception("Not a valid WAVE file.");
			}

			_largeFile = false;
			foundFormat = false;
			foundData = false;

			while (_fs.Position < fileEnd) {
				uint ckID, ckSize, ckSizePadded;
				long ckEnd;

				ckID = _br.ReadUInt32();
				ckSize = _br.ReadUInt32();
				ckSizePadded = (ckSize + 1U) & ~1U;
				ckEnd = _fs.Position + (long)ckSizePadded;

				if (ckID == fccFormat) {
					foundFormat = true;

					if (_br.ReadUInt16() != 1) {
						throw new Exception("WAVE must be PCM format.");
					}
					_channelCount = _br.ReadInt16();
					_sampleRate = _br.ReadInt32();
					_br.ReadInt32();
					_blockAlign = _br.ReadInt16();
					_bitsPerSample = _br.ReadInt16();
				}
				else if (ckID == fccData) {
					foundData = true;

					_dataOffset = (ulong) _fs.Position;
					if (_fs.Length <= maxFileSize) {
						_dataLen = ckSize;
					}
					else {
						_largeFile = true;
						_dataLen = ((ulong)_fs.Length) - _dataOffset;
					}
				}

				if ((foundFormat & foundData) || _largeFile) {
					break;
				}

				_fs.Seek(ckEnd, SeekOrigin.Begin);
			}

			if ((foundFormat & foundData) == false) {
				throw new Exception("Format or data chunk not found.");
			}

			if (_channelCount <= 0) {
				throw new Exception("Channel count is invalid.");
			}
			if (_sampleRate <= 0) {
				throw new Exception("Sample rate is invalid.");
			}
			if (_blockAlign != (_channelCount * ((_bitsPerSample + 7) / 8))) {
				throw new Exception("Block align is invalid.");
			}
			if ((_bitsPerSample <= 0) || (_bitsPerSample > 32)) {
				throw new Exception("Bits per sample is invalid.");
			}
		}

		public ulong Position {
			get {
				return _samplePos;
			}
			set {
				ulong seekPos;

				if (value > _sampleLen) {
					_samplePos = _sampleLen;
				}
				else {
					_samplePos = value;
				}

				seekPos = _dataOffset + (_samplePos * (uint)_blockAlign);
				_fs.Seek((long) seekPos, SeekOrigin.Begin);
			}
		}

		public ulong Length {
			get {
				return _sampleLen;
			}
		}

		public ulong Remaining {
			get {
				return _sampleLen - _samplePos;
			}
		}

		public int ChannelCount {
			get {
				return _channelCount;
			}
		}

		public int SampleRate {
			get {
				return _sampleRate;
			}
		}

		public int BitsPerSample {
			get {
				return _bitsPerSample;
			}
		}

		public int BlockAlign {
			get {
				return _blockAlign;
			}
		}

		public NameValueCollection Tags {
			get {
				return new NameValueCollection();
			}
			set {
			}
		}

		public void GetTags(out List<string> names, out List<string> values)
		{
			names = new List<string>();
			values = new List<string>();
		}

		public uint Read(byte[] buff, uint sampleCount) {
			if (sampleCount > Remaining)
				sampleCount = (uint) Remaining;

			uint byteCount = sampleCount * (uint) _blockAlign;

			if (sampleCount != 0) {
				if (_fs.Read(buff, 0, (int) byteCount) != byteCount) {
					throw new Exception("Incomplete file read.");
				}
				_samplePos += sampleCount;
			}

			return sampleCount;
		}
	}

	public class WAVWriter : IAudioDest {
		FileStream _fs;
		BinaryWriter _bw;
		int _bitsPerSample, _channelCount, _sampleRate, _blockAlign;
		long _sampleLen;

		public WAVWriter(string path, int bitsPerSample, int channelCount, int sampleRate) {
			_bitsPerSample = bitsPerSample;
			_channelCount = channelCount;
			_sampleRate = sampleRate;
			_blockAlign = _channelCount * ((_bitsPerSample + 7) / 8);

			_fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
			_bw = new BinaryWriter(_fs);

			WriteHeaders();
		}

		public bool SetTags(NameValueCollection tags)
		{
			return false;
		}

		private void WriteHeaders() {
			const uint fccRIFF = 0x46464952;
			const uint fccWAVE = 0x45564157;
			const uint fccFormat = 0x20746D66;
			const uint fccData = 0x61746164;

			_bw.Write(fccRIFF);
			_bw.Write((uint)0);
			_bw.Write(fccWAVE);

			_bw.Write(fccFormat);
			_bw.Write((uint)16);
			_bw.Write((ushort)1);
			_bw.Write((ushort)_channelCount);
			_bw.Write((uint)_sampleRate);
			_bw.Write((uint)(_sampleRate * _blockAlign));
			_bw.Write((ushort)_blockAlign);
			_bw.Write((ushort)_bitsPerSample);

			_bw.Write(fccData);
			_bw.Write((uint)0);
		}

		public void Close() {
			const long maxFileSize = 0x7FFFFFFEL;
			long dataLen, dataLenPadded;

			dataLen = _sampleLen * _blockAlign;

			if ((dataLen & 1) == 1) {
				_bw.Write((byte)0);
			}

			if ((dataLen + 44) > maxFileSize) {
				dataLen = ((maxFileSize - 44) / _blockAlign) * _blockAlign;
			}

			dataLenPadded = ((dataLen & 1) == 1) ? (dataLen + 1) : dataLen;

			_bw.Seek(4, SeekOrigin.Begin);
			_bw.Write((uint)(dataLenPadded + 36));

			_bw.Seek(40, SeekOrigin.Begin);
			_bw.Write((uint)dataLen);

			_bw.Close();

			_bw = null;
			_fs = null;
		}

		public long Position {
			get {
				return _sampleLen;
			}
		}

		public long FinalSampleCount {
			set {
			}
		}

		public void Write(byte[] buff, uint sampleCount) {
			if (sampleCount < 0) {
				sampleCount = 0;
			}

			if (sampleCount != 0) {
				_fs.Write(buff, 0, (int) sampleCount * _blockAlign);
				_sampleLen += sampleCount;
			}
		}
	}

#if !MONO
	class FLACReader : IAudioSource {
		FLACDotNet.FLACReader _flacReader;
		int[,] _sampleBuffer;
		uint _bufferOffset, _bufferLength;

		public FLACReader(string path) {
			_flacReader = new FLACDotNet.FLACReader(path);
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

		public bool UpdateTags()
		{
			return _flacReader.UpdateTags();
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

		private unsafe void FLACSamplesToBytes_16(int[,] inSamples, uint inSampleOffset,
			byte[] outSamples, uint outByteOffset, uint sampleCount, int channelCount)
		{
			uint loopCount = sampleCount * (uint) channelCount;

			if ((inSamples.GetLength(0) - inSampleOffset < sampleCount) ||
				(outSamples.Length - outByteOffset < loopCount * 2))
			{
				throw new IndexOutOfRangeException();
			}

			fixed (int* pInSamplesFixed = &inSamples[inSampleOffset, 0]) {
				fixed (byte* pOutSamplesFixed = &outSamples[outByteOffset]) {
					int* pInSamples = pInSamplesFixed;
					short* pOutSamples = (short*)pOutSamplesFixed;

					for (int i = 0; i < loopCount; i++) {
						*(pOutSamples++) = (short)*(pInSamples++);
					}
				}
			}
		}

		public uint Read(byte[] buff, uint sampleCount) {
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

				FLACSamplesToBytes_16(_sampleBuffer, _bufferOffset, buff, buffOffset,
					copyCount, chanCount);

				samplesNeeded -= copyCount;
				buffOffset += copyCount * (uint) chanCount * 2;
				_bufferOffset += copyCount;
			}

			return sampleCount;
		}
	}

	class FLACWriter : IAudioDest {
		FLACDotNet.FLACWriter _flacWriter;
		int[,] _sampleBuffer;
		int _bitsPerSample;
		int _channelCount;
		int _sampleRate;

		public FLACWriter(string path, int bitsPerSample, int channelCount, int sampleRate) {
			if (bitsPerSample != 16) {
				throw new Exception("Bits per sample must be 16.");
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

		private unsafe void BytesToFLACSamples_16(byte[] inSamples, int inByteOffset,
			int[,] outSamples, int outSampleOffset, uint sampleCount, int channelCount)
		{
			uint loopCount = sampleCount * (uint) channelCount;

			if ((inSamples.Length - inByteOffset < loopCount * 2) ||
				(outSamples.GetLength(0) - outSampleOffset < sampleCount))
			{
				throw new IndexOutOfRangeException();
			}

			fixed (byte* pInSamplesFixed = &inSamples[inByteOffset]) {
				fixed (int* pOutSamplesFixed = &outSamples[outSampleOffset, 0]) {
					short* pInSamples = (short*)pInSamplesFixed;
					int* pOutSamples = pOutSamplesFixed;

					for (int i = 0; i < loopCount; i++) {
						*(pOutSamples++) = (int)*(pInSamples++);
					}
				}
			}
		}

		public void Write(byte[] buff, uint sampleCount) {
			if ((_sampleBuffer == null) || (_sampleBuffer.GetLength(0) < sampleCount)) {
				_sampleBuffer = new int[sampleCount, _channelCount];
			}
			BytesToFLACSamples_16(buff, 0, _sampleBuffer, 0, sampleCount, _channelCount);
			_flacWriter.Write(_sampleBuffer, (int) sampleCount);
		}
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

		private unsafe void APESamplesToBytes_16(int[,] inSamples, uint inSampleOffset,
			byte[] outSamples, uint outByteOffset, uint sampleCount, int channelCount)
		{
			uint loopCount = sampleCount * (uint) channelCount;

			if ((inSamples.GetLength(0) - inSampleOffset < sampleCount) ||
				(outSamples.Length - outByteOffset < loopCount * 2))
			{
				throw new IndexOutOfRangeException();
			}

			fixed (int* pInSamplesFixed = &inSamples[inSampleOffset, 0]) {
				fixed (byte* pOutSamplesFixed = &outSamples[outByteOffset]) {
					int* pInSamples = pInSamplesFixed;
					short* pOutSamples = (short*)pOutSamplesFixed;

					for (int i = 0; i < loopCount; i++) {
						*(pOutSamples++) = (short)*(pInSamples++);
					}
				}
			}
		}

		public uint Read(byte[] buff, uint sampleCount) {
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

				APESamplesToBytes_16(_sampleBuffer, _bufferOffset, buff, buffOffset,
					copyCount, chanCount);

				samplesNeeded -= copyCount;
				buffOffset += copyCount * (uint) chanCount * 2;
				_bufferOffset += copyCount;
			}

			return sampleCount;
		}
	}

	class APEWriter : IAudioDest
	{
		APEDotNet.APEWriter _apeWriter;
		//int[,] _sampleBuffer;
		int _bitsPerSample;
		int _channelCount;
		int _sampleRate;

		public APEWriter(string path, int bitsPerSample, int channelCount, int sampleRate)
		{
			if (bitsPerSample != 16)
			{
				throw new Exception("Bits per sample must be 16.");
			}
			_bitsPerSample = bitsPerSample;
			_channelCount = channelCount;
			_sampleRate = sampleRate;
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
		private unsafe void BytesToAPESamples_16(byte[] inSamples, int inByteOffset,
			int[,] outSamples, int outSampleOffset, uint sampleCount, int channelCount)
		{
			uint loopCount = sampleCount * (uint)channelCount;

			if ((inSamples.Length - inByteOffset < loopCount * 2) ||
				(outSamples.GetLength(0) - outSampleOffset < sampleCount))
			{
				throw new IndexOutOfRangeException();
			}

			fixed (byte* pInSamplesFixed = &inSamples[inByteOffset])
			{
				fixed (int* pOutSamplesFixed = &outSamples[outSampleOffset, 0])
				{
					short* pInSamples = (short*)pInSamplesFixed;
					int* pOutSamples = pOutSamplesFixed;

					for (int i = 0; i < loopCount; i++)
					{
						*(pOutSamples++) = (int)*(pInSamples++);
					}
				}
			}
		}
		public void Write(byte[] buff, uint sampleCount)
		{
			//if ((_sampleBuffer == null) || (_sampleBuffer.GetLength(0) < sampleCount))
			//{
			//    _sampleBuffer = new int[sampleCount, _channelCount];
			//}
			//BytesToAPESamples_16(buff, 0, _sampleBuffer, 0, sampleCount, _channelCount);
			//_apeWriter.Write(_sampleBuffer, (int)sampleCount);
			_apeWriter.Write (buff, sampleCount);
		}
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

		private unsafe void WavPackSamplesToBytes_16(int[,] inSamples, uint inSampleOffset,
			byte[] outSamples, uint outByteOffset, uint sampleCount, int channelCount)
		{
			uint loopCount = sampleCount * (uint) channelCount;

			if ((inSamples.GetLength(0) - inSampleOffset < sampleCount) ||
				(outSamples.Length - outByteOffset < loopCount * 2))
			{
				throw new IndexOutOfRangeException();
			}

			fixed (int* pInSamplesFixed = &inSamples[inSampleOffset, 0]) {
				fixed (byte* pOutSamplesFixed = &outSamples[outByteOffset]) {
					int* pInSamples = pInSamplesFixed;
					short* pOutSamples = (short*)pOutSamplesFixed;

					for (int i = 0; i < loopCount; i++) {
						*(pOutSamples++) = (short)*(pInSamples++);
					}
				}
			}
		}

		public uint Read(byte[] buff, uint sampleCount) {
			if (_wavPackReader.BitsPerSample != 16) {
				throw new Exception("Reading is only supported for 16 bit sample depth.");
			}
			int chanCount = _wavPackReader.ChannelCount;
			int[,] sampleBuffer;

			sampleBuffer = new int[sampleCount * 2, chanCount];
			_wavPackReader.Read(sampleBuffer, (int) sampleCount);
			WavPackSamplesToBytes_16(sampleBuffer, 0, buff, 0, sampleCount, chanCount);

			return sampleCount;
		}
	}

	class WavPackWriter : IAudioDest {
		WavPackDotNet.WavPackWriter _wavPackWriter;
		int[,] _sampleBuffer;
		int _bitsPerSample;
		int _channelCount;
		int _sampleRate;

		public WavPackWriter(string path, int bitsPerSample, int channelCount, int sampleRate) {
			if (bitsPerSample != 16) {
				throw new Exception("Bits per sample must be 16.");
			}
			_bitsPerSample = bitsPerSample;
			_channelCount = channelCount;
			_sampleRate = sampleRate;
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

		public void Close() {
			_wavPackWriter.Close();
		}

		private unsafe void BytesToWavPackSamples_16(byte[] inSamples, int inByteOffset,
			int[,] outSamples, int outSampleOffset, uint sampleCount, int channelCount)
		{
			uint loopCount = sampleCount * (uint) channelCount;

			if ((inSamples.Length - inByteOffset < loopCount * 2) ||
				(outSamples.GetLength(0) - outSampleOffset < sampleCount))
			{
				throw new IndexOutOfRangeException();
			}

			fixed (byte* pInSamplesFixed = &inSamples[inByteOffset]) {
				fixed (int* pOutSamplesFixed = &outSamples[outSampleOffset, 0]) {
					short* pInSamples = (short*)pInSamplesFixed;
					int* pOutSamples = pOutSamplesFixed;

					for (int i = 0; i < loopCount; i++) {
						*(pOutSamples++) = (int)*(pInSamples++);
					}
				}
			}
		}

		public void Write(byte[] buff, uint sampleCount) {
			if ((_sampleBuffer == null) || (_sampleBuffer.GetLength(0) < sampleCount)) {
				_sampleBuffer = new int[sampleCount, _channelCount];
			}
			BytesToWavPackSamples_16(buff, 0, _sampleBuffer, 0, sampleCount, _channelCount);
			_wavPackWriter.Write(_sampleBuffer, (int) sampleCount);
		}
	}
#endif
}