using System;
using System.Collections.Generic;
using System.IO;

namespace CUETools.Codecs.WAV
{
    [AudioEncoderClass(typeof(EncoderSettings))]
    public class AudioEncoder : IAudioDest
    {
        private Stream _IO;
        private BinaryWriter _bw;
        private long _sampleLen;
        private string _path;
        private long hdrLen = 0;
        private bool _headersWritten = false;
        private long _finalSampleCount = -1;
        private List<byte[]> _chunks = null;
        private List<uint> _chunkFCCs = null;

        public long Position
        {
            get
            {
                return _sampleLen;
            }
        }

        public long FinalSampleCount
        {
            set { _finalSampleCount = value; }
        }

        private EncoderSettings m_settings;
        public AudioEncoderSettings Settings => m_settings;

        public string Path { get { return _path; } }

        public AudioEncoder(EncoderSettings settings, string path, Stream IO = null)
        {
            m_settings = settings;
            _path = path;
            _IO = IO ?? new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
            _bw = new BinaryWriter(_IO);
        }

        public void WriteChunk(uint fcc, byte[] data)
        {
            if (_sampleLen > 0)
                throw new Exception("data already written, no chunks allowed");
            if (_chunks == null)
            {
                _chunks = new List<byte[]>();
                _chunkFCCs = new List<uint>();
            }
            _chunkFCCs.Add(fcc);
            _chunks.Add(data);
            hdrLen += 8 + data.Length + (data.Length & 1);
        }

        private void WriteHeaders()
        {
            const uint fccRIFF = 0x46464952;
            const uint fccWAVE = 0x45564157;
            const uint fccFormat = 0x20746D66;
            const uint fccData = 0x61746164;

            bool wavex = (Settings.PCM.BitsPerSample != 16 && Settings.PCM.BitsPerSample != 24) || Settings.PCM.ChannelMask != AudioPCMConfig.GetDefaultChannelMask(Settings.PCM.ChannelCount);

            hdrLen += 36 + (wavex ? 24 : 0) + 8;

            uint dataLen = (uint)(_finalSampleCount * Settings.PCM.BlockAlign);
            uint dataLenPadded = dataLen + (dataLen & 1);

            _bw.Write(fccRIFF);
            if (_finalSampleCount <= 0)
                _bw.Write((uint)0xffffffff);
            else
                _bw.Write((uint)(dataLenPadded + hdrLen - 8));
            _bw.Write(fccWAVE);
            _bw.Write(fccFormat);
            if (wavex)
            {
                _bw.Write((uint)40);
                _bw.Write((ushort)0xfffe); // WAVEX follows
            }
            else
            {
                _bw.Write((uint)16);
                _bw.Write((ushort)1); // PCM
            }
            _bw.Write((ushort)Settings.PCM.ChannelCount);
            _bw.Write((uint)Settings.PCM.SampleRate);
            _bw.Write((uint)(Settings.PCM.SampleRate * Settings.PCM.BlockAlign));
            _bw.Write((ushort)Settings.PCM.BlockAlign);
            _bw.Write((ushort)((Settings.PCM.BitsPerSample + 7) / 8 * 8));
            if (wavex)
            {
                _bw.Write((ushort)22); // length of WAVEX structure
                _bw.Write((ushort)Settings.PCM.BitsPerSample);
                _bw.Write((uint)Settings.PCM.ChannelMask);
                _bw.Write((ushort)1); // PCM Guid
                _bw.Write((ushort)0);
                _bw.Write((ushort)0);
                _bw.Write((ushort)0x10);
                _bw.Write((byte)0x80);
                _bw.Write((byte)0x00);
                _bw.Write((byte)0x00);
                _bw.Write((byte)0xaa);
                _bw.Write((byte)0x00);
                _bw.Write((byte)0x38);
                _bw.Write((byte)0x9b);
                _bw.Write((byte)0x71);
            }
            if (_chunks != null)
                for (int i = 0; i < _chunks.Count; i++)
                {
                    _bw.Write(_chunkFCCs[i]);
                    _bw.Write((uint)_chunks[i].Length);
                    _bw.Write(_chunks[i]);
                    if ((_chunks[i].Length & 1) != 0)
                        _bw.Write((byte)0);
                }

            _bw.Write(fccData);
            if (_finalSampleCount <= 0)
                _bw.Write((uint)0xffffffff);
            else
                _bw.Write(dataLen);

            _headersWritten = true;
        }

        public void Close()
        {
            if (_finalSampleCount <= 0 && _IO.CanSeek)
            {
                long dataLen = _sampleLen * Settings.PCM.BlockAlign;
                long dataLenPadded = dataLen + (dataLen & 1);
                if (dataLenPadded + hdrLen - 8 < 0xffffffff)
                {
                    if ((dataLen & 1) == 1)
                        _bw.Write((byte)0);

                    _bw.Seek(4, SeekOrigin.Begin);
                    _bw.Write((uint)(dataLenPadded + hdrLen - 8));

                    _bw.Seek((int)hdrLen - 4, SeekOrigin.Begin);
                    _bw.Write((uint)dataLen);
                }
            }

            _bw.Close();

            _bw = null;
            _IO = null;

            if (_finalSampleCount > 0 && _sampleLen != _finalSampleCount)
                throw new Exception("Samples written differs from the expected sample count.");
        }

        public void Delete()
        {
            _bw.Close();
            _bw = null;
            _IO = null;
            if (_path != "")
                File.Delete(_path);
        }

        public void Write(AudioBuffer buff)
        {
            if (buff.Length == 0)
                return;
            buff.Prepare(this);
            if (!_headersWritten)
                WriteHeaders();
            _IO.Write(buff.Bytes, 0, buff.ByteLength);
            _sampleLen += buff.Length;
        }
    }
}
