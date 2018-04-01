using FFmpeg.AutoGen;
using Newtonsoft.Json;
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.ffmpegdll
{

    [JsonObject(MemberSerialization.OptIn)]
    public abstract class DecoderSettings: IAudioDecoderSettings
    {
        #region IAudioDecoderSettings implementation

        [Browsable(false)]
        public string Name => "ffmpeg";

        [Browsable(false)]
        public Type DecoderType => typeof(AudioDecoder);

        [Browsable(false)]
        public int Priority => 1;

        [Browsable(false)]
        public abstract string Extension { get; }

        public IAudioDecoderSettings Clone()
        {
            return MemberwiseClone() as IAudioDecoderSettings;
        }
        #endregion

        public abstract AVCodecID Codec { get; }

        //        [DisplayName("Version")]
        //        [Description("Library version")]
        //        public string Version => Marshal.PtrToStringAnsi(MACLibDll.GetVersionString());
    }

    public class MLPDecoderSettings : DecoderSettings, IAudioDecoderSettings
    {
        public override string Extension => "mlp";
        public override AVCodecID Codec => AVCodecID.AV_CODEC_ID_MLP;
        public MLPDecoderSettings()
        {
            this.Init();
        }
    }
    public class FLACDecoderSettings : DecoderSettings, IAudioDecoderSettings
    {
        public override string Extension => "flac";
        public override AVCodecID Codec => AVCodecID.AV_CODEC_ID_FLAC;
        public FLACDecoderSettings()
        {
            this.Init();
        }
    }

    public class WavPackDecoderSettings : DecoderSettings, IAudioDecoderSettings
    {
        public override string Extension => "wv";
        public override AVCodecID Codec => AVCodecID.AV_CODEC_ID_WAVPACK;
        public WavPackDecoderSettings()
        {
            this.Init();
        }
    }
}