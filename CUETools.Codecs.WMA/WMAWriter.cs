using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using CUETools.Codecs;
using System.IO;
using System.Text;
using WindowsMediaLib;
using WindowsMediaLib.Defs;
using System.Collections.Generic;

namespace CUETools.Codecs.WMA
{
    public abstract class WMAWriterSettings : AudioEncoderSettings
    {
        public WMAWriterSettings()
            : base()
        {
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
                                    WaveFormatEx pWfx = new WaveFormatEx();
                                    Marshal.PtrToStructure(pMediaType.formatPtr, pWfx);
                                    var info = new WMAFormatInfo()
                                    {
                                        codec = iCodec,
                                        codecName = codecName.ToString(),
                                        format = iFormat,
                                        formatName = szDesc.ToString(),
                                        subType = pMediaType.subType,
                                        pcm = new AudioPCMConfig(pWfx.wBitsPerSample, pWfx.nChannels, pWfx.nSamplesPerSec)
                                    };
                                    if (PCM == null || (pWfx.nChannels == PCM.ChannelCount && pWfx.wBitsPerSample >= PCM.BitsPerSample && pWfx.nSamplesPerSec == PCM.SampleRate))
                                        yield return info;
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

        public override string GetSupportedModes(out string defaultMode)
        {
            var fmts = GetFormats(null);
            defaultMode = fmts.Count > 0 ? fmts[fmts.Count - 1].modeName : "";
            return string.Join(" ", fmts.ConvertAll(s => s.modeName).ToArray());
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

    public class WMALWriterSettings : WMAWriterSettings
    {
        public WMALWriterSettings()
            : base()
        {
            this.m_subType = MediaSubType.WMAudio_Lossless;
        }
    }

    public class WMALossyWriterSettings : WMAWriterSettings
    {
        public WMALossyWriterSettings()
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

    [AudioEncoderClass("wma lossless", "wma", true, 1, typeof(WMALWriterSettings))]
    [AudioEncoderClass("wma lossy", "wma", false, 1, typeof(WMALossyWriterSettings))]
    public class WMAWriter : IAudioDest
    {
        IWMWriter m_pWriter;
        private string outputPath;
        private bool closed = false;
        private bool fileCreated = false;
        private bool writingBegan = false;
        private long sampleCount, finalSampleCount;

        const ushort WAVE_FORMAT_EXTENSIBLE = 0xFFFE;
        const ushort WAVE_FORMAT_PCM = 1;

        /// <summary>
        /// From WAVEFORMATEXTENSIBLE
        /// </summary>
        [StructLayout(LayoutKind.Sequential, Pack = 2)]
        public struct WaveFormatExtensible
        {
            public short wFormatTag;        /* format type */
            public short nChannels;         /* number of channels (i.e. mono, stereo, etc.) */
            public int nSamplesPerSec;    /* sample rate */
            public int nAvgBytesPerSec;   /* for buffer estimation */
            public short nBlockAlign;       /* block size of data */
            public short wBitsPerSample;
            public short cbSize;
            public short wValidBitsPerSample;
            public int dwChannelMask;
            public Guid SubFormat;
        }

        public long FinalSampleCount
        {
            set
            {
                this.finalSampleCount = value;
            }
        }

        public string Path
        {
            get { return this.outputPath; }
        }

        WMAWriterSettings m_settings;

        public virtual AudioEncoderSettings Settings
        {
            get
            {
                return m_settings;
            }
        }

        public WMAWriter(string path, WMAWriterSettings settings)
        {
            this.m_settings = settings;
            this.outputPath = path;

            try
            {
                m_pWriter = settings.GetWriter();
                int cInputs;
                m_pWriter.GetInputCount(out cInputs);
                if (cInputs < 1) throw new InvalidOperationException();
                IWMInputMediaProps pInput;
                m_pWriter.GetInputProps(0, out pInput);
                try
                {
                    int cbType = 0;
                    AMMediaType pMediaType = null;
                    pInput.GetMediaType(pMediaType, ref cbType);
                    pMediaType = new AMMediaType();
                    pMediaType.formatSize = cbType - Marshal.SizeOf(typeof(AMMediaType));
                    pInput.GetMediaType(pMediaType, ref cbType);
                    try
                    {
                        var wfe = new WaveFormatExtensible();
                        wfe.nChannels = (short)m_settings.PCM.ChannelCount;
                        wfe.nSamplesPerSec = m_settings.PCM.SampleRate;
                        wfe.nBlockAlign = (short)m_settings.PCM.BlockAlign;
                        wfe.wBitsPerSample = (short)m_settings.PCM.BitsPerSample;
                        wfe.nAvgBytesPerSec = wfe.nSamplesPerSec * wfe.nBlockAlign;
                        if ((m_settings.PCM.BitsPerSample == 8 || m_settings.PCM.BitsPerSample == 16 || m_settings.PCM.BitsPerSample == 24) &&
                            (m_settings.PCM.ChannelCount == 1 || m_settings.PCM.ChannelCount == 2))
                        {
                            wfe.wFormatTag = unchecked((short)WAVE_FORMAT_PCM);
                            wfe.cbSize = 0;
                        }
                        else
                        {
                            wfe.wFormatTag = unchecked((short)WAVE_FORMAT_EXTENSIBLE);
                            wfe.cbSize = 22;
                            wfe.wValidBitsPerSample = wfe.wBitsPerSample;
                            wfe.nBlockAlign = (short)((wfe.wBitsPerSample / 8) * wfe.nChannels);
                            wfe.dwChannelMask = (int)m_settings.PCM.ChannelMask;
                            wfe.SubFormat = MediaSubType.PCM;
                        }
                        Marshal.FreeCoTaskMem(pMediaType.formatPtr);
                        pMediaType.formatPtr = IntPtr.Zero;
                        pMediaType.formatSize = 0;
                        pMediaType.formatPtr = Marshal.AllocCoTaskMem(Marshal.SizeOf(wfe));
                        pMediaType.formatSize = Marshal.SizeOf(wfe);
                        Marshal.StructureToPtr(wfe, pMediaType.formatPtr, false);
                        pInput.SetMediaType(pMediaType);
                        m_pWriter.SetInputProps(0, pInput);
                    }
                    finally
                    {
                        WMUtils.FreeWMMediaType(pMediaType);
                    }
                }
                finally
                {
                    Marshal.ReleaseComObject(pInput);
                }
            }
            catch (Exception ex)
            {
                if (m_pWriter != null)
                {
                    Marshal.ReleaseComObject(m_pWriter);
                    m_pWriter = null;
                }
                throw ex;
            }
        }

        public void Close()
        {
            if (!this.closed)
            {
                try
                {
                    if (this.writingBegan)
                    {
                        m_pWriter.EndWriting();
                        this.writingBegan = false;
                    }
                }
                finally
                {
                    if (m_pWriter != null)
                    {
                        Marshal.ReleaseComObject(m_pWriter);
                        m_pWriter = null;
                    }
                }

                this.closed = true;
            }
        }

        public void Delete()
        {
            if (this.outputPath == null)
                throw new InvalidOperationException("This writer was not created from file.");

            if (!this.closed)
            {
                this.Close();

                if (this.fileCreated)
                {
                    File.Delete(this.outputPath);
                    this.fileCreated = false;
                }
            }
        }

        public void Write(AudioBuffer buffer)
        {
            if (this.closed)
                throw new InvalidOperationException("Writer already closed.");

            if (!this.fileCreated)
            {
                this.m_pWriter.SetOutputFilename(outputPath);
                this.fileCreated = true;
            }
            if (!this.writingBegan)
            {
                this.m_pWriter.BeginWriting();
                this.writingBegan = true;
            }

            buffer.Prepare(this);
            INSSBuffer pSample;
            m_pWriter.AllocateSample(buffer.ByteLength, out pSample);
            IntPtr pdwBuffer;
            pSample.GetBuffer(out pdwBuffer);
            pSample.SetLength(buffer.ByteLength);
            Marshal.Copy(buffer.Bytes, 0, pdwBuffer, buffer.ByteLength);
            long cnsSampleTime = sampleCount * 10000000L / Settings.PCM.SampleRate;
            m_pWriter.WriteSample(0, cnsSampleTime, SampleFlag.CleanPoint, pSample);
            Marshal.ReleaseComObject(pSample);
            sampleCount += buffer.Length;
        }
    }
}
