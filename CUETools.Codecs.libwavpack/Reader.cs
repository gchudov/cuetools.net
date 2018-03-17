using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using CUETools.Codecs;

namespace CUETools.Codecs.libwavpack
{
    [AudioDecoderClass("libwavpack", "wv", 1)]
    public unsafe class Reader : IAudioSource
    {
        private readonly void* IO_ID_WV = ((IntPtr)1).ToPointer();
        private readonly void* IO_ID_WVC = ((IntPtr)2).ToPointer();
        public Reader(string path, Stream IO, Stream IO_WVC)
        {
            m_read_bytes = ReadCallback;
            m_get_pos = TellCallback;
            m_set_pos_abs = SeekCallback;
            m_set_pos_rel = SeekRelativeCallback;
            m_push_back_byte = PushBackCallback;
            m_get_length = LengthCallback;
            m_can_seek = CanSeekCallback;
            
            m_ioReader  = (WavpackStreamReader64*)Marshal.AllocHGlobal(sizeof(WavpackStreamReader64)).ToPointer();
            m_ioReader->read_bytes = Marshal.GetFunctionPointerForDelegate(m_read_bytes);
            m_ioReader->write_bytes = IntPtr.Zero;
            m_ioReader->get_pos = Marshal.GetFunctionPointerForDelegate(m_get_pos);
			m_ioReader->set_pos_abs = Marshal.GetFunctionPointerForDelegate(m_set_pos_abs);
			m_ioReader->set_pos_rel = Marshal.GetFunctionPointerForDelegate(m_set_pos_rel);
			m_ioReader->push_back_byte = Marshal.GetFunctionPointerForDelegate(m_push_back_byte);
			m_ioReader->get_length = Marshal.GetFunctionPointerForDelegate(m_get_length);
			m_ioReader->can_seek = Marshal.GetFunctionPointerForDelegate(m_can_seek);
            m_ioReader->truncate_here = IntPtr.Zero;
            m_ioReader->close = IntPtr.Zero;

            _IO_ungetc = _IO_WVC_ungetc = -1;

			_path = path;

			_IO = (IO != null) ? IO : new FileStream (path, FileMode.Open, FileAccess.Read, FileShare.Read);
			_IO_WVC = (IO != null) ? IO_WVC : File.Exists (path+"c") ? new FileStream (path+"c", FileMode.Open, FileAccess.Read, FileShare.Read) : null;

            string errorMessage;
            
            _wpc = wavpackdll.WavpackOpenFileInputEx64(m_ioReader, IO_ID_WV, IO_ID_WVC, out errorMessage, OpenFlags.OPEN_WVC, 0);
			if (_wpc == null) {
				throw new Exception("Unable to initialize the decoder: " + errorMessage);
			}

			pcm = new AudioPCMConfig(
                wavpackdll.WavpackGetBitsPerSample(_wpc),
                wavpackdll.WavpackGetNumChannels(_wpc), 
			    (int)wavpackdll.WavpackGetSampleRate(_wpc),
				(AudioPCMConfig.SpeakerConfig)wavpackdll.WavpackGetChannelMask(_wpc));
			_sampleCount = wavpackdll.WavpackGetNumSamples64(_wpc);
			_sampleOffset = 0;
        }

        public Reader(string path, Stream IO)
            : this(path, IO, null)
        {}

        public AudioDecoderSettings Settings => null;

        public AudioPCMConfig PCM => pcm;

        public string Path => _path;

        public long Length => _sampleCount;

        public long Position
        {
            get => _sampleOffset;

            set
            {
                _sampleOffset = value;
                if (0 == wavpackdll.WavpackSeekSample64(_wpc, value))
                    throw new Exception("unable to seek: " + wavpackdll.WavpackGetErrorMessage(_wpc));
            }
        }

        public long Remaining => _sampleCount - _sampleOffset;

        public void Close()
        {
			if (_wpc != null)
				_wpc = wavpackdll.WavpackCloseFile(_wpc);
			if (_IO != null) 
			{
				_IO.Close ();
				_IO = null;
			}
			if (_IO_WVC != null) 
			{
				_IO_WVC.Close ();
				_IO_WVC = null;
			}
            Marshal.FreeHGlobal((IntPtr)m_ioReader);
            m_ioReader = null;
        }

        public int Read(AudioBuffer buff, int maxLength)
        {
            buff.Prepare(this, maxLength);
            fixed (int*pSampleBuffer = &buff.Samples[0,0])
            {
    			uint samplesRead = wavpackdll.WavpackUnpackSamples(_wpc, pSampleBuffer, (uint)buff.Length);
    			_sampleOffset += samplesRead;
    			if (samplesRead != buff.Length)
    				throw new Exception("Decoder returned a different number of samples than requested.");
            }
            return buff.Length;
        }

        private int ReadCallback(void* id, void* data, int bcount)
        {
            Stream IO = (id == IO_ID_WVC) ? _IO_WVC : _IO;
            int IO_ungetc = (id == IO_ID_WVC) ? _IO_WVC_ungetc : _IO_ungetc;
            int unget_len = 0;

            if (IO_ungetc != -1)
            {
                *(byte*)data = (byte) IO_ungetc;
                if (IO == _IO)
                    _IO_ungetc = -1;
                else
                    _IO_WVC_ungetc = -1;
                bcount--;
                if (bcount <= 0)
                    return 1;
                data = 1 + (byte*)data;
                unget_len = 1;
            }

            if (_readBuffer == null || _readBuffer.Length < bcount)
                _readBuffer = new byte[Math.Max(bcount, 0x4000)];
            int len = IO.Read(_readBuffer, 0, bcount);
            if (len > 0) Marshal.Copy(_readBuffer, 0, (IntPtr)data, len);
            return len + unget_len;
        }

        long TellCallback(void* id)
        {
            Stream IO = (id == IO_ID_WVC) ? _IO_WVC : _IO;
            return IO.Position;
        }

        int SeekCallback(void* id, long pos)
        {
            Stream IO = (id == IO_ID_WVC) ? _IO_WVC : _IO;
            IO.Position = pos;
            return 0;
        }

        int SeekRelativeCallback(void* id, long delta, int mode)
        {
            Stream IO = (id == IO_ID_WVC) ? _IO_WVC : _IO;
            IO.Seek(delta, (SeekOrigin)(mode));
            return 0;
        }

        int PushBackCallback(void* id, int c)
        {
            Stream IO = (id == IO_ID_WVC) ? _IO_WVC : _IO;
            if (IO == _IO)
            {
                if (_IO_ungetc != -1)
                    throw new Exception("Double PushBackCallback unsupported.");
                _IO_ungetc = c;
            }
            else
            {
                if (_IO_WVC_ungetc != -1)
                    throw new Exception("Double PushBackCallback unsupported.");
                _IO_WVC_ungetc = c;
            }

            return 0;
        }

        long LengthCallback(void* id)
        {
            Stream IO = (id == IO_ID_WVC) ? _IO_WVC : _IO;
            return IO.Length;
        }

        int CanSeekCallback(void* id)
        {
            Stream IO = (id == IO_ID_WVC) ? _IO_WVC : _IO;
            return IO.CanSeek ? 1 : 0;
        }

        WavpackContext* _wpc;
		long _sampleCount, _sampleOffset;
		Stream _IO;
		Stream _IO_WVC;
        string _path;
		int _IO_ungetc, _IO_WVC_ungetc;
        AudioPCMConfig pcm;
		WavpackStreamReader64* m_ioReader;
        DecoderReadDelegate m_read_bytes;
        DecoderTellDelegate64 m_get_pos;
        DecoderSeekDelegate64 m_set_pos_abs;
        DecoderSeekRelativeDelegate64 m_set_pos_rel;
        DecoderPushBackDelegate m_push_back_byte;
        DecoderLengthDelegate64 m_get_length;
        DecoderCanSeekDelegate m_can_seek;
        byte[] _readBuffer;
    }
}
