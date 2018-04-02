using System;
using System.IO;
using System.Runtime.InteropServices;
using FFmpeg.AutoGen;

namespace CUETools.Codecs.ffmpegdll
{
    public unsafe class AudioDecoder : IAudioSource, IDisposable
    {
        private static void RegisterLibrariesSearchPath(string path)
        {
            switch (Environment.OSVersion.Platform)
            {
                case PlatformID.Win32NT:
                case PlatformID.Win32S:
                case PlatformID.Win32Windows:
                    SetDllDirectory(path);
                    break;
                    //case PlatformID.Unix:
                    //case PlatformID.MacOSX:
                    //    string currentValue = Environment.GetEnvironmentVariable(LD_LIBRARY_PATH);
                    //    if (string.IsNullOrWhiteSpace(currentValue) == false && currentValue.Contains(path) == false)
                    //    {
                    //        string newValue = currentValue + Path.PathSeparator + path;
                    //        Environment.SetEnvironmentVariable(LD_LIBRARY_PATH, newValue);
                    //    }
                    //    break;
            }
        }

        [DllImport("kernel32", SetLastError = true)]
        private static extern bool SetDllDirectory(string lpPathName);

        public AudioDecoder(DecoderSettings settings, string path, Stream IO)
        {
            m_settings = settings;

            _path = path;

            m_stream = (IO != null) ? IO : new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);

            switch (Environment.OSVersion.Platform)
            {
                case PlatformID.Win32NT:
                case PlatformID.Win32S:
                case PlatformID.Win32Windows:
                    var myPath = new Uri(typeof(AudioDecoder).Assembly.CodeBase).LocalPath;
                    var current = System.IO.Path.GetDirectoryName(myPath);
                    var probe = Environment.Is64BitProcess ? "x64" : "win32";
                    while (current != null)
                    {
                        var ffmpegDirectory = System.IO.Path.Combine(current, probe);
                        if (Directory.Exists(ffmpegDirectory))
                        {
                            System.Diagnostics.Trace.WriteLine($"FFmpeg binaries found in: {ffmpegDirectory}");
                            RegisterLibrariesSearchPath(ffmpegDirectory);
                            break;
                        }
                        current = Directory.GetParent(current)?.FullName;
                    }
                    break;
                    //case PlatformID.Unix:
                    //case PlatformID.MacOSX:
                    //    var libraryPath = Environment.GetEnvironmentVariable(LD_LIBRARY_PATH);
                    //    RegisterLibrariesSearchPath(libraryPath);
                    //    break;
            }

            pkt = ffmpeg.av_packet_alloc();
            if (pkt == null)
                throw new Exception("Unable to initialize the decoder");

            decoded_frame = ffmpeg.av_frame_alloc();
            if (decoded_frame == null)
                throw new Exception("Could not allocate audio frame");

            //ffmpeg.avcodec_register_all();
            ffmpeg.av_register_all();

#if DEBUG
            ffmpeg.av_log_set_level(ffmpeg.AV_LOG_DEBUG);

            av_log_set_callback_callback logCallback = (p0, level, format, vl) =>
            {
                if (level > ffmpeg.av_log_get_level()) return;

                var lineSize = 1024;
                var lineBuffer = stackalloc byte[lineSize];
                var printPrefix = 1;
                ffmpeg.av_log_format_line(p0, level, format, vl, lineBuffer, lineSize, &printPrefix);
                var line = Marshal.PtrToStringAnsi((IntPtr) lineBuffer);
                System.Diagnostics.Trace.Write(line);
            };

            ffmpeg.av_log_set_callback(logCallback);
#endif

            m_read_packet_callback = readPacketCallback;
            m_seek_callback = seekCallback;

            int ret;
            AVFormatContext* new_fmt_ctx = ffmpeg.avformat_alloc_context();
            if (new_fmt_ctx == null)
                throw new Exception("ffmpeg.avformat_alloc_context() failed");

            ulong avio_ctx_buffer_size = 65536;
            void* avio_ctx_buffer = ffmpeg.av_malloc(avio_ctx_buffer_size);

            AVIOContext* avio_ctx = ffmpeg.avio_alloc_context((byte*)avio_ctx_buffer, (int)avio_ctx_buffer_size,
                0, null, m_read_packet_callback, null, m_seek_callback);
            if (avio_ctx == null)
            {
                ffmpeg.avformat_free_context(new_fmt_ctx);
                throw new Exception("Cannot find stream information");
            }

            new_fmt_ctx->pb = avio_ctx;

            AVInputFormat* fmt = ffmpeg.av_find_input_format(m_settings.Format);
            if (fmt == null)
            {
                ffmpeg.avformat_free_context(new_fmt_ctx);
                throw new Exception($"Cannot find input format ${m_settings.Format}");
            }

            if ((ret = ffmpeg.avformat_open_input(&new_fmt_ctx, null, fmt, null)) < 0)
            {
                ffmpeg.avformat_free_context(new_fmt_ctx);
                throw new Exception("Cannot open input file");
            }

            if ((ret = ffmpeg.avformat_find_stream_info(new_fmt_ctx, null)) < 0)
            {
                ffmpeg.avformat_close_input(&new_fmt_ctx);
                throw new Exception("Cannot find stream information");
            }

#if FINDBESTSTREAM
            /* select the audio stream */
            ret = ffmpeg.av_find_best_stream(new_fmt_ctx, AVMediaType.AVMEDIA_TYPE_AUDIO, -1, -1, &dec, 0);
            if (ret < 0)
            {
                ffmpeg.avformat_close_input(&new_fmt_ctx);
                throw new Exception("Cannot find an audio stream in the input file");
            }
#else
            if (new_fmt_ctx->nb_streams != 1)
            {
                ffmpeg.avformat_close_input(&new_fmt_ctx);
                throw new Exception("More than one stream");
            }
#endif

            int audio_stream_index = 0; // ret

            if (new_fmt_ctx->streams[audio_stream_index]->duration > 0)
                _sampleCount = new_fmt_ctx->streams[audio_stream_index]->duration;
            else
                _sampleCount = -1;

            int bps = new_fmt_ctx->streams[audio_stream_index]->codecpar->bits_per_raw_sample != 0 ?
                new_fmt_ctx->streams[audio_stream_index]->codecpar->bits_per_raw_sample :
                new_fmt_ctx->streams[audio_stream_index]->codecpar->bits_per_coded_sample;
            int channels = new_fmt_ctx->streams[audio_stream_index]->codecpar->channels;
            int sample_rate = new_fmt_ctx->streams[audio_stream_index]->codecpar->sample_rate;
            ulong channel_layout = new_fmt_ctx->streams[audio_stream_index]->codecpar->channel_layout;
            pcm = new AudioPCMConfig(bps, channels, sample_rate, (AudioPCMConfig.SpeakerConfig)0);

            // ret = ffmpeg.av_read_frame(new_fmt_ctx, pkt);

            fmt_ctx = new_fmt_ctx;

            //m_stream.Seek(0, SeekOrigin.Begin);

            codec = ffmpeg.avcodec_find_decoder(m_settings.Codec);
            if (codec == null)
                throw new Exception("Codec not found");

            c = ffmpeg.avcodec_alloc_context3(codec);
            if (c == null)
                throw new Exception("Could not allocate audio codec context");
            // ffmpeg.av_opt_set_int(c, "refcounted_frames", 1, 0);
            ffmpeg.avcodec_parameters_to_context(c, fmt_ctx->streams[audio_stream_index]->codecpar);

            c->request_sample_fmt = AVSampleFormat.AV_SAMPLE_FMT_S32;

            /* open it */
            if (ffmpeg.avcodec_open2(c, codec, null) < 0)
                throw new Exception("Could not open codec");

            m_decoded_frame_offset = 0;
            m_decoded_frame_size = 0;
            _sampleOffset = 0;
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (disposing)
            {
                //if (m_StreamIO != null)
                //{
                //    m_StreamIO.Dispose();
                //    m_StreamIO = null;
                //}
                if (m_stream != null)
                {
                    m_stream.Dispose();
                    m_stream = null;
                }
            }

            AVFormatContext* fmt_ctx1 = fmt_ctx;
            ffmpeg.avformat_close_input(&fmt_ctx1);
            fmt_ctx = null;

            AVCodecContext* c1 = c;
            ffmpeg.avcodec_free_context(&c1);
            c = c1;
            //c = null;

            AVFrame* decoded_frame1 = decoded_frame;
            ffmpeg.av_frame_free(&decoded_frame1);
            decoded_frame = decoded_frame1;
            //decoded_frame = null;

            AVPacket* pkt1 = pkt;
            ffmpeg.av_packet_free(&pkt1);
            pkt = pkt1;
            //pkt = null;
        }

        ~AudioDecoder()
        {
            Dispose(false);
        }

        private DecoderSettings m_settings;

        public IAudioDecoderSettings Settings => m_settings;

        public AudioPCMConfig PCM => pcm;

        public string Path => _path;

        public long Length => _sampleCount;

        public long Position
        {
            get => _sampleOffset;

            set
            {
                throw new NotSupportedException();
                //_bufferOffset = 0;
                //_bufferLength = 0;
                //_sampleOffset = value;
                //int res = MACLibDll.c_APEDecompress_Seek(pAPEDecompress, (int)value);
                //if (0 != res)
                //    throw new Exception("unable to seek:" + res.ToString());
            }
        }

        public long Remaining => _sampleCount < 0 ? -1 : _sampleCount - _sampleOffset;

        public void Close()
        {
            Dispose(true);
        }

        byte[] _readBuffer;
        int readPacketCallback(void* @opaque, byte* @buf, int @buf_size)
        {
            // TODO: if instead of calling ffmpeg.av_malloc for
            // the buffer we pass to ffmpeg.avio_alloc_context
            // we just pin _readBuffer, we wouldn't need to Copy.
            if (_readBuffer == null || _readBuffer.Length < @buf_size)
                _readBuffer = new byte[Math.Max(@buf_size, 0x4000)];
            int len = m_stream.Read(_readBuffer, 0, @buf_size);
            if (len > 0) Marshal.Copy(_readBuffer, 0, (IntPtr)buf, len);
            return len;
        }

        long seekCallback(void* @opaque, long @offset, int @whence)
        {
            if (whence == ffmpeg.AVSEEK_SIZE)
                return m_stream.Length;
            whence &= ~ffmpeg.AVSEEK_FORCE;
            return m_stream.Seek(@offset, (SeekOrigin)@whence);
        }

        private void fill()
        {
            while (true)
            {
                if (m_decoded_frame_size > 0)
                    return;
                int ret = ffmpeg.avcodec_receive_frame(c, decoded_frame);
                if (ret == ffmpeg.AVERROR_EOF)
                    return;
                if (ret != ffmpeg.AVERROR(ffmpeg.EAGAIN))
                {
                    if (ret < 0) throw new Exception("Error during decoding");
                    m_decoded_frame_offset = 0;
                    m_decoded_frame_size = decoded_frame->nb_samples;
                    return;
                }
                ret = ffmpeg.av_read_frame(fmt_ctx, pkt);
                if (ret != 0)
                {
                    if (ret == ffmpeg.AVERROR_EOF)
                        return;
                    byte* buf = stackalloc byte[256];
                    ffmpeg.av_strerror(ret, buf, 256);
                    throw new Exception("Error while parsing: " + Marshal.PtrToStringAnsi((IntPtr)buf));
                }
                if (pkt->size != 0)
                {
                    /* send the packet with the compressed data to the decoder */
                    ret = ffmpeg.avcodec_send_packet(c, pkt);
                    if (ret < 0) throw new Exception("Error submitting the packet to the decoder");
                }
            }
        }

        public int Read(AudioBuffer buff, int maxLength)
        {
            buff.Prepare(this, maxLength);

            long buffOffset = 0;
            long samplesNeeded = buff.Length;
            long _channelCount = pcm.ChannelCount;

            while (samplesNeeded != 0)
            {
                if (m_decoded_frame_size == 0)
                {
                    fill();
                    if (m_decoded_frame_size == 0)
                        break;
                }
                long copyCount = Math.Min(samplesNeeded, m_decoded_frame_size);

                // TODO: if AudioBuffer supported different sample formats,
                // this would be simpler. One complication though we would still
                // need shifts.
                switch (c->sample_fmt)
                {
                    case AVSampleFormat.AV_SAMPLE_FMT_S32:
                        {
                            byte* ptr = decoded_frame->data[0u] + c->channels * 4 * m_decoded_frame_offset;
                            int rshift = 32 - pcm.BitsPerSample;
                            int* smp = (int*)ptr;
                            fixed (int* dst_start = &buff.Samples[buffOffset, 0])
                            {
                                int* dst = dst_start;
                                int* dst_end = dst_start + copyCount * c->channels;
                                while (dst < dst_end)
                                    *(dst++) = *(smp++) >> rshift;
                            }
                        }
                        break;
                    case AVSampleFormat.AV_SAMPLE_FMT_S16P:
                        for (Int32 iChan = 0; iChan < _channelCount; iChan++)
                        {
                            fixed (int* pMyBuffer = &buff.Samples[buffOffset, iChan])
                            {
                                int* pMyBufferPtr = pMyBuffer;
                                short* pFLACBuffer = (short*)(decoded_frame->data[(uint)iChan]) + m_decoded_frame_offset;
                                short* pFLACBufferEnd = pFLACBuffer + copyCount;
                                while (pFLACBuffer < pFLACBufferEnd)
                                {
                                    *pMyBufferPtr = *pFLACBuffer;
                                    pMyBufferPtr += _channelCount;
                                    pFLACBuffer++;
                                }
                            }
                        }
                        break;
                    default:
                        throw new NotSupportedException();
                }

                samplesNeeded -= copyCount;
                buffOffset += copyCount;
                m_decoded_frame_offset += copyCount;
                m_decoded_frame_size -= copyCount;
                _sampleOffset += copyCount;
            }

            buff.Length = (int)buffOffset;
            return buff.Length;
        }

        AVPacket* pkt;
        AVFrame* decoded_frame;
        AVCodec* codec;
        AVCodecContext* c;
        AVFormatContext* fmt_ctx;

        avio_alloc_context_read_packet m_read_packet_callback;
        avio_alloc_context_seek m_seek_callback;

        long _sampleCount, _sampleOffset;
        AudioPCMConfig pcm;
        string _path;
        Stream m_stream;
        long m_decoded_frame_offset;
        long m_decoded_frame_size;
    }
}
