using System;
using System.Runtime.InteropServices;
using CUETools.Codecs;
using System.IO;
using WindowsMediaLib;
using WindowsMediaLib.Defs;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.WMA
{
    public class WMAWriterSettings
    {
        public WMAWriterSettings() {  }
    }


    [AudioEncoderClass("windows", "wma", true, "", "", 1, typeof(WMAWriterSettings))]
    public class WMAWriter : IAudioDest
    {
        IWMProfileManager m_pProfileManager;
        IWMWriter m_pWriter;
        int m_iCodec, m_iFormat;

        private string outputPath;

        private bool closed = false;
        private AudioPCMConfig pcm;
        private long sampleCount, finalSampleCount;

        public long BlockSize
        {
            set { }
        }

        public virtual int CompressionLevel
        {
            get { return 0; }
            set { }
        }

        public long FinalSampleCount
        {
            set
            {
                this.finalSampleCount = value;
            }
        }

        public AudioPCMConfig PCM
        {
            get { return this.pcm; }
        }

        public long Padding
        {
            set { }
        }

        public string Path
        {
            get { return this.outputPath; }
        }

        public virtual object Settings
        {
            get; set;
        }

        public WMAWriter(string path, AudioPCMConfig pcm)
        {
            this.CheckPCMConfig(pcm);
            this.pcm = pcm;
            this.outputPath = path;

            try
            {
                WMUtils.WMCreateProfileManager(out m_pProfileManager);
                var pCodecInfo3 = m_pProfileManager as IWMCodecInfo3;
                int cCodecs;
                pCodecInfo3.GetCodecInfoCount(MediaType.Audio, out cCodecs);
                bool codecFound = false;
                for (int iCodec = 0; iCodec < cCodecs; iCodec++)
                {
        		    //if (codec != WMAvoice)
                    try
                    {
                        pCodecInfo3.SetCodecEnumerationSetting(MediaType.Audio, iCodec, Constants.g_wszVBREnabled, AttrDataType.BOOL, new byte[] {1, 0, 0, 0}, 4);
                        //pCodecInfo3.SetCodecEnumerationSetting(MediaType.Audio, iCodec, Constants.g_wszNumPasses, AttrDataType.DWORD, new byte[] {1, 0, 0, 0}, 4);
		            }
                    catch (COMException)
                    {
                    }

                    int cFormat;
                    pCodecInfo3.GetCodecFormatCount(MediaType.Audio, iCodec, out cFormat);
                    for (int iFormat = 0; iFormat < cFormat; iFormat++)
                    {
                        IWMStreamConfig pStreamConfig;
                        pCodecInfo3.GetCodecFormat(MediaType.Audio, iCodec, iFormat, out pStreamConfig);
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
                                if (pMediaType.majorType == MediaType.Audio && pMediaType.formatType == FormatType.WaveEx && pMediaType.subType == MediaSubType.WMAudio_Lossless)
                                {
                                    WaveFormatEx pWfx = new WaveFormatEx();
                                    Marshal.PtrToStructure(pMediaType.formatPtr, pWfx);
                                    if (pWfx.nChannels == pcm.ChannelCount && pWfx.wBitsPerSample == pcm.BitsPerSample && pWfx.nSamplesPerSec == pcm.SampleRate)
                                    {
                                        m_iCodec = iCodec;
                                        m_iFormat = iFormat;
                                        codecFound = true;
                                    }
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
                if (!codecFound)
                    throw new NotSupportedException("codec not found");
                IWMStreamConfig pStreamConfig1;
                pCodecInfo3.GetCodecFormat(MediaType.Audio, m_iCodec, m_iFormat, out pStreamConfig1);
                try
                {
                    pStreamConfig1.SetStreamNumber(1);
                    IWMProfile pProfile;
                    m_pProfileManager.CreateEmptyProfile(WMVersion.V9_0, out pProfile);
                    try
                    {
                        pProfile.AddStream(pStreamConfig1);
                        WMUtils.WMCreateWriter(IntPtr.Zero, out m_pWriter);
                        m_pWriter.SetProfile(pProfile);
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
            catch (Exception ex)
            {
                if (m_pWriter != null)
                {
                    Marshal.ReleaseComObject(m_pWriter);
                    m_pWriter = null;
                }
                if (m_pProfileManager != null)
                {
                    Marshal.ReleaseComObject(m_pProfileManager);
                    m_pProfileManager = null;
                }
                throw ex;
            }
        }

        private void CheckPCMConfig(AudioPCMConfig pcm)
        {
            if (pcm.BitsPerSample != 16)
            {
                throw new ArgumentException("LAME only supports 16 bits/sample.");
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
                    if (m_pProfileManager != null)
                    {
                        Marshal.ReleaseComObject(m_pProfileManager);
                        m_pProfileManager = null;
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
            long cnsSampleTime = sampleCount * 10000000L / pcm.SampleRate;
            m_pWriter.WriteSample(0, cnsSampleTime, SampleFlag.CleanPoint, pSample);
            Marshal.ReleaseComObject(pSample);
            sampleCount += buffer.Length;
        }
    }
}
