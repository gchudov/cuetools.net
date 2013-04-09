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
        public WMAWriterSettings(Guid subType)
            : base()
        {
            this.m_subType = subType;
        }

        private readonly Guid m_subType;
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
                                    if (PCM == null || (pWfx.nChannels == PCM.ChannelCount && pWfx.wBitsPerSample == PCM.BitsPerSample && pWfx.nSamplesPerSec == PCM.SampleRate))
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
            : base(MediaSubType.WMAudio_Lossless)
        {
        }
    }

    public class WMAV8VBRWriterSettings : WMAWriterSettings
    {
        public WMAV8VBRWriterSettings()
            : base(MediaSubType.WMAudioV8)
        {
            this.m_vbr = true;
        }
    }

    public class WMAV8CBRWriterSettings : WMAWriterSettings
    {
        public WMAV8CBRWriterSettings()
            : base(MediaSubType.WMAudioV8)
        {
            this.m_vbr = false;
        }
    }

    public class WMAV9CBRWriterSettings : WMAWriterSettings
    {
        public WMAV9CBRWriterSettings()
            : base(MediaSubType.WMAudioV9)
        {
            this.m_vbr = false;
        }
    }

    [AudioEncoderClass("wma lossless", "wma", true, 1, typeof(WMALWriterSettings))]
    [AudioEncoderClass("wma v8 vbr", "wma", false, 3, typeof(WMAV8VBRWriterSettings))]
    [AudioEncoderClass("wma v9 cbr", "wma", false, 2, typeof(WMAV9CBRWriterSettings))]
    [AudioEncoderClass("wma v8 cbr", "wma", false, 1, typeof(WMAV8CBRWriterSettings))]
    public class WMAWriter : IAudioDest
    {
        IWMWriter m_pWriter;
        private string outputPath;
        private bool closed = false;
        private long sampleCount, finalSampleCount;

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
                //for (int iInput = 0; iInput < cInputs; iInput++)
                //{
                //}
                //IWMInputMediaProps pInput;
                //pWriter.GetInputProps(0, out pInput);
                //pInput.GetMediaType(pType, ref cbType);
                // fill (WAVEFORMATEX*)pType->pbFormat
                // WAVEFORMATEXTENSIBLE if needed (dwChannelMask, wValidBitsPerSample)
                // if (chg)
                //pInput.SetMediaType(pType);
                //pWriter.SetInputProps(0, pInput);

                //{ DWORD dwFormatCount = 0; hr = pWriter->GetInputFormatCount(0, &dwFormatCount); TEST(hr); TESTB(dwFormatCount > 0); }
                //// GetInputFormatCount failed previously for multichannel formats, before ...mask = guessChannelMask() added. Leave this check j.i.c.
                m_pWriter.SetOutputFilename(outputPath);
                m_pWriter.BeginWriting();
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
                    m_pWriter.EndWriting();
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
            {
                throw new InvalidOperationException("This writer was not created from file.");
            }

            if (!closed)
            {
                this.Close();
                File.Delete(this.outputPath);
            }
        }

        public void Write(AudioBuffer buffer)
        {
            if (this.closed)
            {
                throw new InvalidOperationException("Writer already closed.");
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
