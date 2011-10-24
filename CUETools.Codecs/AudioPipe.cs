using System;
using System.Threading;

namespace CUETools.Codecs
{
    public class AudioPipe : IAudioSource//, IDisposable
    {
        private AudioBuffer _readBuffer, _writeBuffer;
        private AudioPCMConfig pcm;
        private long _sampleLen, _samplePos;
        private int _maxLength;
        private Thread _workThread;
        private IAudioSource _source;
        private bool _close = false;
        private bool _haveData = false;
        private int _bufferPos = 0;
        private Exception _ex = null;
        private bool own;
        private ThreadPriority priority;

        public long Position
        {
            get
            {
                return _samplePos;
            }
            set
            {
                if (value == _samplePos)
                    return;

                if (_source == null)
                    throw new NotSupportedException();

                lock (this)
                {
                    _close = true;
                    Monitor.Pulse(this);
                }
                if (_workThread != null)
                {
                    _workThread.Join();
                    _workThread = null;
                }
                _source.Position = value;
                _samplePos = value;
                _bufferPos = 0;
                _haveData = false;
                _close = false;
                //Go();
                //throw new Exception("not supported");
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

        public AudioPCMConfig PCM
        {
            get
            {
                return pcm;
            }
        }

        public string Path
        {
            get
            {
                if (_source == null)
                    return "";
                return _source.Path;
            }
        }

        public AudioPipe(AudioPCMConfig pcm, int size)
        {
            this.pcm = pcm;
            _readBuffer = new AudioBuffer(pcm, size);
            _writeBuffer = new AudioBuffer(pcm, size);
            _maxLength = size;
            _sampleLen = -1;
            _samplePos = 0;
        }

        public AudioPipe(IAudioSource source, int size, bool own, ThreadPriority priority)
            : this(source.PCM, size)
        {
            this.own = own;
            this.priority = priority;
            _source = source;
            _sampleLen = _source.Length;
            _samplePos = _source.Position;
        }

        public AudioPipe(IAudioSource source, int size)
            : this(source, size, true, ThreadPriority.BelowNormal)
        {
        }

        private void Decompress(object o)
        {
#if !DEBUG
			try
#endif
            {
                bool done = false;
                do
                {
                    done = _source.Read(_writeBuffer, -1) == 0;
                    lock (this)
                    {
                        while (_haveData && !_close)
                            Monitor.Wait(this);
                        if (_close)
                            break;
                        AudioBuffer temp = _writeBuffer;
                        _writeBuffer = _readBuffer;
                        _readBuffer = temp;
                        _haveData = true;
                        Monitor.Pulse(this);
                    }
                } while (!done);
            }
#if !DEBUG
			catch (Exception ex)
			{
				lock (this)
				{
					_ex = ex;
					Monitor.Pulse(this);
				}
			}
#endif
        }

        private void Go()
        {
            if (_workThread != null || _ex != null || _source == null) return;
            _workThread = new Thread(Decompress);
            _workThread.Priority = priority;
            _workThread.IsBackground = true;
            _workThread.Name = "AudioPipe";
            _workThread.Start(null);
        }

        //public new void Dispose()
        //{
        //    _buffer.Clear();
        //}

        public void Close()
        {
            lock (this)
            {
                _close = true;
                Monitor.Pulse(this);
            }
            if (_workThread != null)
            {
                _workThread.Join();
                _workThread = null;
            }
            if (_source != null)
            {
                if (own) _source.Close();
                _source = null;
            }
            if (_readBuffer != null)
            {
                //_readBuffer.Clear();
                _readBuffer = null;
            }
            if (_writeBuffer != null)
            {
                //_writeBuffer.Clear();
                _writeBuffer = null;
            }
        }

        public int Write(AudioBuffer buff)
        {
            if (_writeBuffer.Size < _writeBuffer.Length + buff.Length)
            {
                AudioBuffer realloced = new AudioBuffer(pcm, _writeBuffer.Size + buff.Size);
                realloced.Prepare(_writeBuffer, 0, _writeBuffer.Length);
                _writeBuffer = realloced;
            }
            if (_writeBuffer.Length == 0)
                _writeBuffer.Prepare(buff, 0, buff.Length);
            else
            {
                _writeBuffer.Load(_writeBuffer.Length, buff, 0, buff.Length);
                _writeBuffer.Length += buff.Length;
            }
            lock (this)
            {
                if (!_haveData)
                {
                    AudioBuffer temp = _writeBuffer;
                    _writeBuffer = _readBuffer;
                    _writeBuffer.Length = 0;
                    _readBuffer = temp;
                    _haveData = true;
                    Monitor.Pulse(this);
                }
            }
            return _writeBuffer.Length;
        }

        public int Read(AudioBuffer buff, int maxLength)
        {
            Go();

            bool needToCopy = false;
            if (_bufferPos != 0)
                needToCopy = true;
            else
                lock (this)
                {
                    while (!_haveData && _ex == null)
                        Monitor.Wait(this);
                    if (_ex != null)
                        throw _ex;
                    if (_bufferPos == 0 && (maxLength < 0 || _readBuffer.Length <= maxLength))
                    {
                        buff.Swap(_readBuffer);
                        _haveData = false;
                        Monitor.Pulse(this);
                    }
                    else
                        needToCopy = true;
                }
            if (needToCopy)
            {
                buff.Prepare(_readBuffer, _bufferPos, maxLength);
                _bufferPos += buff.Length;
                if (_bufferPos == _readBuffer.Length)
                {
                    _bufferPos = 0;
                    lock (this)
                    {
                        _haveData = false;
                        Monitor.Pulse(this);
                    }
                }
            }
            _samplePos += buff.Length;
            return buff.Length;
        }
    }
}
