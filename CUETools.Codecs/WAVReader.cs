using System;
using System.IO;

namespace CUETools.Codecs
{
    [AudioDecoderClass("cuetools", "wav", 2)]
    public class WAVReader : IAudioSource
    {
        Stream _IO;
        BinaryReader _br;
        long _dataOffset, _samplePos, _sampleLen;
        private AudioPCMConfig pcm;
        long _dataLen;
        bool _largeFile;
        string _path;

        public long Position
        {
            get
            {
                return _samplePos;
            }
            set
            {
                long seekPos;

                if (_samplePos == value)
                {
                    return;
                }

                var oldSamplePos = _samplePos;
                if (_sampleLen >= 0 && value > _sampleLen)
                    _samplePos = _sampleLen;
                else
                    _samplePos = value;

                if (_IO.CanSeek || _samplePos < oldSamplePos)
                {
                    seekPos = _dataOffset + _samplePos * PCM.BlockAlign;
                    _IO.Seek(seekPos, SeekOrigin.Begin);
                }
                else
                {
                    int offs = (int)(_samplePos - oldSamplePos) * PCM.BlockAlign;
                    while (offs > 0)
                    {
                        int chunk = Math.Min(offs, 16536);
                        _br.ReadBytes(chunk);
                        offs -= chunk;
                    }
                }
            }
        }

        public long Length
        {
            get
            {
                return _sampleLen;
            }
        }

        public long Remaining
        {
            get
            {
                return _sampleLen - _samplePos;
            }
        }

        public AudioPCMConfig PCM { get { return pcm; } }

        public string Path { get { return _path; } }

        public WAVReader(string path, Stream IO)
        {
            _path = path;
            _IO = IO != null ? IO : new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 0x10000, FileOptions.SequentialScan);
            _br = new BinaryReader(_IO);

            ParseHeaders();

            if (_dataLen < 0)
                _sampleLen = -1;
            else
                _sampleLen = _dataLen / pcm.BlockAlign;
        }

        public WAVReader(string path, Stream IO, AudioPCMConfig _pcm)
        {
            _path = path;
            _IO = IO != null ? IO : new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 0x10000, FileOptions.SequentialScan);
            _br = new BinaryReader(_IO);

            _largeFile = false;
            _dataOffset = 0;
            _samplePos = 0;
            pcm = _pcm;
            _dataLen = _IO.CanSeek ? _IO.Length : -1;
            if (_dataLen < 0)
                _sampleLen = -1;
            else
            {
                _sampleLen = _dataLen / pcm.BlockAlign;
                if ((_dataLen % pcm.BlockAlign) != 0)
                    throw new Exception("odd file size");
            }
        }

        public static AudioBuffer ReadAllSamples(string path, Stream IO)
        {
            WAVReader reader = new WAVReader(path, IO);
            AudioBuffer buff = new AudioBuffer(reader, (int)reader.Length);
            reader.Read(buff, -1);
            if (reader.Remaining != 0)
                throw new Exception("couldn't read the whole file");
            reader.Close();
            return buff;
        }

        public void Close()
        {
            if (_br != null)
            {
                _br.Close();
                _br = null;
            }
            _IO = null;
        }

        private void ParseHeaders()
        {
            const long maxFileSize = 0x7FFFFFFEL;
            const uint fccRIFF = 0x46464952;
            const uint fccWAVE = 0x45564157;
            const uint fccFormat = 0x20746D66;
            const uint fccData = 0x61746164;

            uint lenRIFF;
            bool foundFormat, foundData;

            if (_br.ReadUInt32() != fccRIFF)
            {
                throw new Exception("Not a valid RIFF file.");
            }

            lenRIFF = _br.ReadUInt32();

            if (_br.ReadUInt32() != fccWAVE)
            {
                throw new Exception("Not a valid WAVE file.");
            }

            _largeFile = false;
            foundFormat = false;
            foundData = false;
            long pos = 12;
            do
            {
                uint ckID, ckSize, ckSizePadded;
                long ckEnd;

                ckID = _br.ReadUInt32();
                ckSize = _br.ReadUInt32();
                ckSizePadded = (ckSize + 1U) & ~1U;
                pos += 8;
                ckEnd = pos + (long)ckSizePadded;

                if (ckID == fccFormat)
                {
                    foundFormat = true;

                    uint fmtTag = _br.ReadUInt16();
                    int _channelCount = _br.ReadInt16();
                    int _sampleRate = _br.ReadInt32();
                    _br.ReadInt32(); // bytes per second
                    int _blockAlign = _br.ReadInt16();
                    int _bitsPerSample = _br.ReadInt16();
                    int _channelMask = 0;
                    pos += 16;

                    if (fmtTag == 0xFFFEU && ckSize >= 34) // WAVE_FORMAT_EXTENSIBLE 
                    {
                        _br.ReadInt16(); // CbSize
                        _br.ReadInt16(); // ValidBitsPerSample
                        _channelMask = _br.ReadInt32();
                        fmtTag = _br.ReadUInt16();
                        pos += 10;
                    }

                    if (fmtTag != 1) // WAVE_FORMAT_PCM
                        throw new Exception("WAVE format tag not WAVE_FORMAT_PCM.");

                    pcm = new AudioPCMConfig(_bitsPerSample, _channelCount, _sampleRate, (AudioPCMConfig.SpeakerConfig)_channelMask);
                    if (pcm.BlockAlign != _blockAlign)
                        throw new Exception("WAVE has strange BlockAlign");
                }
                else if (ckID == fccData)
                {
                    foundData = true;

                    _dataOffset = pos;
                    if (!_IO.CanSeek || _IO.Length <= maxFileSize)
                    {
                        if (ckSize == 0 || ckSize >= 0x7fffffff)
                            _dataLen = -1;
                        else
                            _dataLen = (long)ckSize;
                    }
                    else
                    {
                        _largeFile = true;
                        _dataLen = _IO.Length - pos;
                    }
                }

                if ((foundFormat & foundData) || _largeFile)
                    break;
                if (_IO.CanSeek)
                    _IO.Seek(ckEnd, SeekOrigin.Begin);
                else
                    _br.ReadBytes((int)(ckEnd - pos));
                pos = ckEnd;
            } while (true);

            if ((foundFormat & foundData) == false || pcm == null)
                throw new Exception("Format or data chunk not found.");
            if (pcm.ChannelCount <= 0)
                throw new Exception("Channel count is invalid.");
            if (pcm.SampleRate <= 0)
                throw new Exception("Sample rate is invalid.");
            if ((pcm.BitsPerSample <= 0) || (pcm.BitsPerSample > 32))
                throw new Exception("Bits per sample is invalid.");
            if (pos != _dataOffset)
                Position = 0;
        }

        public int Read(AudioBuffer buff, int maxLength)
        {
            buff.Prepare(this, maxLength);

            byte[] bytes = buff.Bytes;
            int byteCount = (int)buff.ByteLength;
            int pos = 0;

            while (pos < byteCount)
            {
                int len = _IO.Read(bytes, pos, byteCount - pos);
                if (len <= 0)
                {
                    if ((pos % PCM.BlockAlign) != 0 || _sampleLen >= 0)
                        throw new Exception("Incomplete file read.");
                    buff.Length = pos / PCM.BlockAlign;
                    _samplePos += buff.Length;
                    _sampleLen = _samplePos;
                    return buff.Length;
                }
                pos += len;
            }
            _samplePos += buff.Length;
            return buff.Length;
        }
    }
}
