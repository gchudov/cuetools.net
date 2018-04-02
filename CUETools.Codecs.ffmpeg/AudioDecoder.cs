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
                            Console.WriteLine($"FFmpeg binaries found in: {ffmpegDirectory}");
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

            ffmpeg.avcodec_register_all();
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

            if (m_stream.CanSeek)
            {
                m_read_packet_callback = readPacketCallback;
                m_seek_callback = seekCallback;

                int ret;
                AVFormatContext* fmt_ctx = ffmpeg.avformat_alloc_context();
                if (fmt_ctx == null)
                    throw new Exception("ffmpeg.avformat_alloc_context() failed");

                ulong avio_ctx_buffer_size = 4096;
                void* avio_ctx_buffer = ffmpeg.av_malloc(avio_ctx_buffer_size);

                AVIOContext* avio_ctx = ffmpeg.avio_alloc_context((byte*)avio_ctx_buffer, (int)avio_ctx_buffer_size,
                    0, null, m_read_packet_callback, null, m_seek_callback);
                if (avio_ctx == null)
                {
                    ffmpeg.avformat_free_context(fmt_ctx);
                    throw new Exception("Cannot find stream information");
                }

                fmt_ctx->pb = avio_ctx;

                AVInputFormat* fmt = ffmpeg.av_find_input_format(m_settings.Format);
                if (fmt==null)
                {
                    ffmpeg.avformat_free_context(fmt_ctx);
                    throw new Exception($"Cannot find input format ${m_settings.Format}");
                }

                if ((ret = ffmpeg.avformat_open_input(&fmt_ctx, null, fmt, null)) < 0)
                {
                    ffmpeg.avformat_close_input(&fmt_ctx);
                    throw new Exception("Cannot open input file");
                }

                if ((ret = ffmpeg.avformat_find_stream_info(fmt_ctx, null)) < 0)
                {
                    ffmpeg.avformat_close_input(&fmt_ctx);
                    throw new Exception("Cannot find stream information");
                }

#if FINDBESTSTREAM
                /* select the audio stream */
                ret = ffmpeg.av_find_best_stream(fmt_ctx, AVMediaType.AVMEDIA_TYPE_AUDIO, -1, -1, &dec, 0);
                if (ret < 0)
                {
                    ffmpeg.avformat_close_input(&fmt_ctx);
                    throw new Exception("Cannot find an audio stream in the input file");
                }
#else
                if (fmt_ctx->nb_streams != 1)
                {
                    ffmpeg.avformat_close_input(&fmt_ctx);
                    throw new Exception("More than one stream");
                }
#endif

                int audio_stream_index = 0; // ret

                if (fmt_ctx->streams[audio_stream_index]->duration > 0)
                    _sampleCount = fmt_ctx->streams[audio_stream_index]->duration;
                else
                    _sampleCount = -1;
                pcm = new AudioPCMConfig(
                    fmt_ctx->streams[audio_stream_index]->codecpar->bits_per_raw_sample,
                    fmt_ctx->streams[audio_stream_index]->codecpar->channels,
                    fmt_ctx->streams[audio_stream_index]->codecpar->sample_rate,
                    (AudioPCMConfig.SpeakerConfig)0);// fmt_ctx->streams[audio_stream_index]->codecpar->channel_layout);

                // ret = ffmpeg.av_read_frame(fmt_ctx, pkt);

                ffmpeg.avformat_close_input(&fmt_ctx);
                fmt_ctx = null;

                m_stream.Seek(0, SeekOrigin.Begin);
            }

            codec = ffmpeg.avcodec_find_decoder(m_settings.Codec);
            if (codec == null)
                throw new Exception("Codec not found");

            parser = ffmpeg.av_parser_init((int)codec->id);
            if (parser == null)
                throw new Exception("Parser not found\n");

            c = ffmpeg.avcodec_alloc_context3(codec);
            if (c == null)
                throw new Exception("Could not allocate audio codec context");
            // ffmpeg.av_opt_set_int(c, "refcounted_frames", 1, 0);
            // ffmpeg.avcodec_parameters_to_context(c, fmt_ctx->streams[audio_stream_index]->codecpar);

            /* open it */
            if (ffmpeg.avcodec_open2(c, codec, null) < 0)
                throw new Exception("Could not open codec");

            data_buf = new byte[AUDIO_INBUF_SIZE];
            data_size = 0;
            data_offs = 0;
            m_decoded_frame_offset = 0;
            m_decoded_frame_size = 0;

            if (pcm == null)
            {
                fill();
                _sampleCount = -1;
                pcm = new AudioPCMConfig(
                    c->bits_per_raw_sample, c->channels, c->sample_rate,
                    (AudioPCMConfig.SpeakerConfig)0); // c->channel_layout;
            }

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

            AVCodecContext* c1 = c;
            ffmpeg.avcodec_free_context(&c1);
            c = c1;
            //c = null;

            ffmpeg.av_parser_close(parser);
            parser = null;

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

        const int AUDIO_INBUF_SIZE = 65536;
        const int AUDIO_REFILL_THRESH = 4096;

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
                if (pkt->size != 0)
                {
                    /* send the packet with the compressed data to the decoder */
                    ret = ffmpeg.avcodec_send_packet(c, pkt);
                    if (ret < 0) throw new Exception("Error submitting the packet to the decoder");
                    pkt->size = 0;
                    continue;
                }
                if (data_size < AUDIO_REFILL_THRESH)
                {
                    Array.Copy(data_buf, data_offs, data_buf, 0, data_size);
                    data_offs = 0;
                    int len = m_stream.Read(data_buf, data_size, data_buf.Length - data_size);
                    data_size += len;
                }
                // int ret = ffmpeg.av_read_frame(fmt_ctx, pkt);
                fixed (byte* data = &data_buf[data_offs])
                    ret = ffmpeg.av_parser_parse2(parser, c, &pkt->data, &pkt->size,
                        data, data_size, ffmpeg.AV_NOPTS_VALUE, ffmpeg.AV_NOPTS_VALUE, 0);
                if (ret < 0 && ret != ffmpeg.AVERROR(ffmpeg.EAGAIN))
                    throw new Exception("Error while parsing");
                if (pkt->size == 0 && data_size == 0 && ret == 0)
                    return;
                data_offs += ret;
                data_size -= ret;
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
        AVCodecParserContext* parser;
        AVCodecContext* c;

        avio_alloc_context_read_packet m_read_packet_callback;
        avio_alloc_context_seek m_seek_callback;

        long _sampleCount, _sampleOffset;
        AudioPCMConfig pcm;
        string _path;
        Stream m_stream;
        long m_decoded_frame_offset;
        long m_decoded_frame_size;

        byte[] data_buf;
        int data_size;
        int data_offs;
    }
}
