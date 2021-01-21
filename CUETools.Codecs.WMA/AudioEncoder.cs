using System;
using System.IO;
using System.Runtime.InteropServices;
using WindowsMediaLib;
using WindowsMediaLib.Defs;

namespace CUETools.Codecs.WMA
{
    public class AudioEncoder : IAudioDest
    {
        IWMWriter m_pEncoder;
        private string outputPath;
        private bool closed = false;
        private bool fileCreated = false;
        private bool writingBegan = false;
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

        EncoderSettings m_settings;

        public IAudioEncoderSettings Settings => m_settings;

        public AudioEncoder(EncoderSettings settings, string path, Stream IO = null)
        {
            this.m_settings = settings;
            this.outputPath = path;

            try
            {
                m_pEncoder = settings.GetWriter();
                int cInputs;
                m_pEncoder.GetInputCount(out cInputs);
                if (cInputs < 1) throw new InvalidOperationException();
                IWMInputMediaProps pInput;
                m_pEncoder.GetInputProps(0, out pInput);
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
                        var wfe = new WaveFormatExtensible(m_settings.PCM);
                        Marshal.FreeCoTaskMem(pMediaType.formatPtr);
                        pMediaType.formatPtr = IntPtr.Zero;
                        pMediaType.formatSize = 0;
                        pMediaType.formatPtr = Marshal.AllocCoTaskMem(Marshal.SizeOf(wfe));
                        pMediaType.formatSize = Marshal.SizeOf(wfe);
                        Marshal.StructureToPtr(wfe, pMediaType.formatPtr, false);
                        pInput.SetMediaType(pMediaType);
                        m_pEncoder.SetInputProps(0, pInput);
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
                if (m_pEncoder != null)
                {
                    Marshal.ReleaseComObject(m_pEncoder);
                    m_pEncoder = null;
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
                        m_pEncoder.EndWriting();
                        this.writingBegan = false;
                    }
                }
                finally
                {
                    if (m_pEncoder != null)
                    {
                        Marshal.ReleaseComObject(m_pEncoder);
                        m_pEncoder = null;
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
                this.m_pEncoder.SetOutputFilename(outputPath);
                this.fileCreated = true;
            }
            if (!this.writingBegan)
            {
                this.m_pEncoder.BeginWriting();
                this.writingBegan = true;
            }

            buffer.Prepare(this);
            INSSBuffer pSample;
            m_pEncoder.AllocateSample(buffer.ByteLength, out pSample);
            IntPtr pdwBuffer;
            pSample.GetBuffer(out pdwBuffer);
            pSample.SetLength(buffer.ByteLength);
            Marshal.Copy(buffer.Bytes, 0, pdwBuffer, buffer.ByteLength);
            long cnsSampleTime = sampleCount * 10000000L / Settings.PCM.SampleRate;
            m_pEncoder.WriteSample(0, cnsSampleTime, SampleFlag.CleanPoint, pSample);
            Marshal.ReleaseComObject(pSample);
            sampleCount += buffer.Length;
        }
    }
}
