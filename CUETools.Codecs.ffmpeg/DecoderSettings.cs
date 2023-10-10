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
        public string Name => "ffmpeg.dll";

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

        public abstract string Format { get; }

        public int StreamId { get; set; }

        //        [DisplayName("Version")]
        //        [Description("Library version")]
        //        public string Version => Marshal.PtrToStringAnsi(MACLibDll.GetVersionString());
    }

    public class MLPDecoderSettings : DecoderSettings, IAudioDecoderSettings
    {
        public override string Extension => "mlp";
        public override string Format => "mlp";
        public MLPDecoderSettings()
        {
            this.Init();
        }
    }
    public class FLACDecoderSettings : DecoderSettings, IAudioDecoderSettings
    {
        public override string Extension => "flac";
        public override string Format => "flac";
        public FLACDecoderSettings()
        {
            this.Init();
        }
    }
    public class WavPackDecoderSettings : DecoderSettings, IAudioDecoderSettings
    {
        public override string Extension => "wv";
        public override string Format => "wv";
        public WavPackDecoderSettings()
        {
            this.Init();
        }
    }
    public class TtaDecoderSettings : DecoderSettings, IAudioDecoderSettings
    {
        public override string Extension => "tta";
        public override string Format => "tta";
        public TtaDecoderSettings()
        {
            this.Init();
        }
    }
    public class ShnDecoderSettings : DecoderSettings, IAudioDecoderSettings
    {
        public override string Extension => "shn";
        public override string Format => "shn";
        public ShnDecoderSettings()
        {
            this.Init();
        }
    }
    public class AlacDecoderSettings : DecoderSettings, IAudioDecoderSettings
    {
        public override string Extension => "m4a";
        public override string Format => "m4a";
        public AlacDecoderSettings()
        {
            this.Init();
        }
    }
    public class APEDecoderSettings : DecoderSettings, IAudioDecoderSettings
    {
        public override string Extension => "ape";
        public override string Format => "ape";
        public APEDecoderSettings()
        {
            this.Init();
        }
    }
    public class MpegTSDecoderSettings : DecoderSettings, IAudioDecoderSettings
    {
        public override string Extension => "m2ts";
        public override string Format => "mpegts";
        public MpegTSDecoderSettings()
        {
            this.Init();
        }
    }
    public class MpegDecoderSettings : DecoderSettings, IAudioDecoderSettings
    {
        public override string Extension => "aob";
        public override string Format => "mpeg";
        public MpegDecoderSettings()
        {
            this.Init();
        }
    }
    public class AiffDecoderSettings : DecoderSettings, IAudioDecoderSettings
    {
        public override string Extension => "aiff";
        public override string Format => "aiff";
        public AiffDecoderSettings()
        {
            this.Init();
        }
    }
}