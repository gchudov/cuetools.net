using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;
using WindowsMediaLib;
using WindowsMediaLib.Defs;

namespace CUETools.Codecs.WMA
{
    [JsonObject(MemberSerialization.OptIn)]
    public abstract class EncoderSettings : IAudioEncoderSettings
    {
        #region IAudioEncoderSettings implementation
        [Browsable(false)]
        public string Extension => "wma";

        [Browsable(false)]
        public abstract string Name { get; }

        [Browsable(false)]
        public Type EncoderType => typeof(AudioEncoder);

        [Browsable(false)]
        public abstract bool Lossless { get; }

        [Browsable(false)]
        public int Priority => 1;

        [Browsable(false)]
        public string SupportedModes =>
            string.Join(" ", GetFormats(null).ConvertAll(s => s.modeName).ToArray());

        [Browsable(false)]
        public string DefaultMode =>
            GetFormats(null).ConvertAll(s => s.modeName).FindLast(x => true) ?? "";

        [Browsable(false)]
        [DefaultValue("")]
        [JsonProperty]
        public string EncoderMode { get; set; }

        [Browsable(false)]
        public AudioPCMConfig PCM { get; set; }

        [Browsable(false)]
        public int BlockSize { get; set; }

        [Browsable(false)]
        [DefaultValue(4096)]
        public int Padding { get; set; }

        public IAudioEncoderSettings Clone()
        {
            return MemberwiseClone() as IAudioEncoderSettings;
        }
        #endregion

        public EncoderSettings()
        {
            this.Init();
        }

        protected Guid m_subType;
        protected bool m_vbr = true;

        public IWMWriter GetWriter()
        {
            IWMProfileManager pProfileManager = null;
            try
            {
                WMUtils.WMCreateProfileManager(out pProfileManager);
                var pCodecInfo3 = pProfileManager as IWMCodecInfo3;
                // We have to use the same pProfileManager for enumeration,
                // because it calls SetCodecEnumerationSetting, so chosenFormat.format
                // would not point to the same format for a pProfileManager
                // with different (default) settings, and GetCodecFormat
                // would return the wrong stream config.
                var formats = GetFormats(pProfileManager);
                if (this.EncoderMode != "")
                    formats.RemoveAll(fmt => fmt.modeName != this.EncoderMode);
                if (formats.Count < 1)
                    throw new NotSupportedException("codec/format not found");
                if (formats.Count > 1)
                    throw new NotSupportedException("codec/format ambiguous");
                var chosenFormat = formats[0];
                IWMStreamConfig pStreamConfig1;
                pCodecInfo3.GetCodecFormat(MediaType.Audio, chosenFormat.codec, chosenFormat.format, out pStreamConfig1);
                try
                {
                    pStreamConfig1.SetStreamNumber(1);
                    IWMProfile pProfile;
                    pProfileManager.CreateEmptyProfile(WMVersion.V9_0, out pProfile);
                    try
                    {
                        pProfile.AddStream(pStreamConfig1);
                        IWMWriter pWriter;
                        WMUtils.WMCreateWriter(IntPtr.Zero, out pWriter);
                        try
                        {
                            pWriter.SetProfile(pProfile);
                        }
                        catch (Exception ex)
                        {
                            Marshal.ReleaseComObject(pWriter);
                            throw ex;
                        }
                        return pWriter;
                    }
                    finally
                    {
                        Marshal.ReleaseComObject(pProfile);
                    }
                }
                finally
                {
                    Marshal.ReleaseComObject(pStreamConfig1);
                }
            }
            finally
            {
                Marshal.ReleaseComObject(pProfileManager);
            }
        }

        internal IEnumerable<WMAFormatInfo> EnumerateFormatInfo(IWMProfileManager pProfileManager2)
        {
            IWMProfileManager pProfileManager = null;
            try
            {
                if (pProfileManager2 == null)
                    WMUtils.WMCreateProfileManager(out pProfileManager);
                var pCodecInfo3 = (pProfileManager2 ?? pProfileManager) as IWMCodecInfo3;
                int cCodecs;
                pCodecInfo3.GetCodecInfoCount(MediaType.Audio, out cCodecs);
                for (int iCodec = 0; iCodec < cCodecs; iCodec++)
                {
                    int szCodecName = 0;
                    pCodecInfo3.GetCodecName(MediaType.Audio, iCodec, null, ref szCodecName);
                    var codecName = new StringBuilder(szCodecName);
                    pCodecInfo3.GetCodecName(MediaType.Audio, iCodec, codecName, ref szCodecName);
                    var attrDataType = new AttrDataType();
                    int dwAttrSize = 0;
                    byte[] pAttrValue = new byte[4];
                    pCodecInfo3.GetCodecProp(MediaType.Audio, iCodec, Constants.g_wszIsVBRSupported, out attrDataType, pAttrValue, ref dwAttrSize);
                    if (pAttrValue[0] != 1)
                        continue;
                    if (m_vbr)
                    {
                        pCodecInfo3.SetCodecEnumerationSetting(MediaType.Audio, iCodec, Constants.g_wszVBREnabled, AttrDataType.BOOL, new byte[] { 1, 0, 0, 0 }, 4);
                        pCodecInfo3.SetCodecEnumerationSetting(MediaType.Audio, iCodec, Constants.g_wszNumPasses, AttrDataType.DWORD, new byte[] { 1, 0, 0, 0 }, 4);
                    }
                    else
                    {
                        pCodecInfo3.SetCodecEnumerationSetting(MediaType.Audio, iCodec, Constants.g_wszVBREnabled, AttrDataType.BOOL, new byte[] { 0, 0, 0, 0 }, 4);
                    }

                    int cFormat;
                    pCodecInfo3.GetCodecFormatCount(MediaType.Audio, iCodec, out cFormat);
                    for (int iFormat = 0; iFormat < cFormat; iFormat++)
                    {
                        IWMStreamConfig pStreamConfig;
                        int cchDesc = 1024;
                        StringBuilder szDesc = new StringBuilder(cchDesc);
                        pCodecInfo3.GetCodecFormatDesc(MediaType.Audio, iCodec, iFormat, out pStreamConfig, szDesc, ref cchDesc);
                        if (szDesc.ToString().Contains("(A/V)"))
                            continue;
                        try
                        {
                            var pProps = pStreamConfig as IWMMediaProps;
                            int cbType = 0;
                            AMMediaType pMediaType = null;
                            pProps.GetMediaType(pMediaType, ref cbType);
                            pMediaType = new AMMediaType();
                            pMediaType.formatSize = cbType - Marshal.SizeOf(typeof(AMMediaType));
                            pProps.GetMediaType(pMediaType, ref cbType);
                            try
                            {
                                if (pMediaType.majorType == MediaType.Audio && pMediaType.formatType == FormatType.WaveEx && pMediaType.subType == m_subType)
                                {
                                    var pcm = WaveFormatExtensible.FromMediaType(pMediaType).GetConfig();
                                    if (PCM == null || (pcm.ChannelCount == PCM.ChannelCount && pcm.SampleRate == PCM.SampleRate && pcm.BitsPerSample >= PCM.BitsPerSample))
                                        yield return new WMAFormatInfo()
                                        {
                                            codec = iCodec,
                                            codecName = codecName.ToString(),
                                            format = iFormat,
                                            formatName = szDesc.ToString(),
                                            subType = pMediaType.subType,
                                            pcm = pcm
                                        };
                                }
                            }
                            finally
                            {
                                WMUtils.FreeWMMediaType(pMediaType);
                            }
                        }
                        finally
                        {
                            Marshal.ReleaseComObject(pStreamConfig);
                        }
                    }
                }
            }
            finally
            {
                if (pProfileManager != null)
                    Marshal.ReleaseComObject(pProfileManager);
            }
        }

        internal List<WMAFormatInfo> GetFormats(IWMProfileManager pProfileManager)
        {
            var formats = new List<WMAFormatInfo>(this.EnumerateFormatInfo(pProfileManager));
            formats.RemoveAll(fmt => formats.Exists(fmt2 => fmt2.pcm.BitsPerSample < fmt.pcm.BitsPerSample && fmt2.pcm.ChannelCount == fmt.pcm.ChannelCount && fmt2.pcm.SampleRate == fmt.pcm.SampleRate));
            if (formats.Count < 2) return formats;
            int prefixLen = 0, suffixLen = 0;
            while (formats.TrueForAll(s => s.formatName.Length > prefixLen &&
                s.formatName.Substring(0, prefixLen + 1) ==
                formats[0].formatName.Substring(0, prefixLen + 1)))
                prefixLen++;
            while (formats.TrueForAll(s => s.formatName.Length > suffixLen &&
                s.formatName.Substring(s.formatName.Length - suffixLen - 1) ==
                formats[0].formatName.Substring(formats[0].formatName.Length - suffixLen - 1)))
                suffixLen++;
            formats.ForEach(s => s.modeName = s.formatName.Substring(prefixLen, s.formatName.Length - suffixLen - prefixLen).Trim().Replace(' ', '_'));
            int ix, iy;
            formats.Sort((Comparison<WMAFormatInfo>)((x, y) => int.TryParse(x.modeName, out ix) && int.TryParse(y.modeName, out iy) ? ix - iy : x.modeName.CompareTo(y.modeName)));
            return formats;
        }
    }

    internal class WMAFormatInfo
    {
        public int codec;
        public string codecName;
        public int format;
        public string formatName;
        public string modeName;
        public Guid subType;
        public AudioPCMConfig pcm;
    }

    public class LosslessEncoderSettings : EncoderSettings
    {
        public override string Name => "wma lossless";

        public override bool Lossless => true;

        public LosslessEncoderSettings()
            : base()
        {
            this.m_subType = MediaSubType.WMAudio_Lossless;
        }
    }

    public class LossyEncoderSettings : EncoderSettings
    {
        public override string Name => "wma lossy";

        public override bool Lossless => false;

        public LossyEncoderSettings()
            : base()
        {
            this.m_subType = MediaSubType.WMAudioV9;
        }

        public enum Codec
        {
            WMA9,
            WMA10Pro
        }

        [DefaultValue(Codec.WMA10Pro)]
        [JsonProperty]
        public Codec Version
        {
            get
            {
                return this.m_subType == MediaSubType.WMAudioV9 ? Codec.WMA10Pro : Codec.WMA9;
            }
            set
            {
                this.m_subType = value == Codec.WMA10Pro ? MediaSubType.WMAudioV9 : MediaSubType.WMAudioV8;
            }
        }

        [DefaultValue(true)]
        [JsonProperty]
        public bool VBR
        {
            get
            {
                return this.m_vbr;
            }
            set
            {
                this.m_vbr = value;
            }
        }
    }
}
