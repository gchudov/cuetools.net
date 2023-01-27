/**
 * CUETools.WMA: WMA audio decoder
 * Copyright (c) 2013-2023 Grigory Chudov
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using WindowsMediaLib;
using WindowsMediaLib.Defs;

namespace CUETools.Codecs.WMA
{
    public class AudioDecoder : IAudioSource
    {
        IWMSyncReader m_syncReader;
        INSSBuffer m_pSample;
        int m_pSampleOffset = 0, m_pSampleSize = 0;
        short m_wStreamNum = -1;
        int m_dwAudioOutputNum = -1;

        AudioPCMConfig pcm;

        long m_sampleCount = -1, m_sampleOffset = 0;

        string m_path;
        Stream m_IO;
        StreamWrapper m_streamWrapper;

        public AudioDecoder(DecoderSettings settings, string path, Stream IO)
        {
            m_settings = settings;
            m_path = path;
            isValid(path);
            bool pfIsProtected;
            WMUtils.WMIsContentProtected(path, out pfIsProtected);
            if (pfIsProtected)
                throw new Exception("DRM present");
            WMUtils.WMCreateSyncReader(IntPtr.Zero, Rights.None, out m_syncReader);

            if (path == null)
            {
                m_IO = IO != null ? IO : new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 0x10000);
                m_streamWrapper = new StreamWrapper(m_IO);
                m_syncReader.OpenStream(m_streamWrapper);
            }
            else
            {
                m_syncReader.Open(path);
            }
            var pProfile = (m_syncReader as IWMProfile);
            int dwStreamCount;
            pProfile.GetStreamCount(out dwStreamCount);
            for (int dwIndex = 0; dwIndex < dwStreamCount; dwIndex++)
            {
                IWMStreamConfig pConfig = null;
                pProfile.GetStream(dwIndex, out pConfig);
                try
                {
                    Guid guid;
                    pConfig.GetStreamType(out guid);
                    if (MediaType.Audio != guid)
                        continue;
                    short wStreamNum;
                    pConfig.GetStreamNumber(out wStreamNum);
                    int dwBitrate = -1;
                    pConfig.GetBitrate(out dwBitrate);
                    var pIWMMediaProps = pConfig as IWMMediaProps;
                    int cbType = 0;
                    pIWMMediaProps.GetMediaType(null, ref cbType);
                    var pMediaType = new AMMediaType();
                    pMediaType.formatSize = cbType;
                    pIWMMediaProps.GetMediaType(pMediaType, ref cbType);
                    if (pMediaType.formatType != FormatType.WaveEx)
                        continue;
                    if (pMediaType.subType != MediaSubType.WMAudio_Lossless)
                        continue;
                    m_wStreamNum = wStreamNum;
                    pcm = WaveFormatExtensible.FromMediaType(pMediaType).GetConfig();
                    break;
                }
                finally
                {
                    Marshal.ReleaseComObject(pConfig);
                }
            }
            if (m_wStreamNum == -1)
                throw new Exception("No WMA lossless streams found");

            m_syncReader.SetReadStreamSamples(m_wStreamNum, false);
            bool pfCompressed;
            m_syncReader.GetReadStreamSamples(m_wStreamNum, out pfCompressed);
            if (pfCompressed)
                throw new Exception("doesn't decompress");
            m_syncReader.GetOutputNumberForStream(m_wStreamNum, out m_dwAudioOutputNum);
            IWMOutputMediaProps pProps;
            m_syncReader.GetOutputProps(m_dwAudioOutputNum, out pProps);

            try
            {
                StringBuilder sName = null;
                AMMediaType pMediaType = null;
                int cbType = 0;

                cbType = 0;
                pMediaType = null;
                pProps.GetMediaType(pMediaType, ref cbType);

                // Get the name of the output we'll be using
                sName = null;
                short iName = 0;
                pProps.GetConnectionName(sName, ref iName);

                sName = new StringBuilder(iName);
                pProps.GetConnectionName(sName, ref iName);

                if (pcm.ChannelCount > 2)
                {
                    m_syncReader.SetOutputSetting(m_dwAudioOutputNum, Constants.g_wszEnableDiscreteOutput, AttrDataType.BOOL, new byte[] { 1, 0, 0, 0 }, 4);
                    m_syncReader.SetOutputSetting(m_dwAudioOutputNum, Constants.g_wszSpeakerConfig, AttrDataType.DWORD, new byte[] { 0, 0, 0, 0 }, 4);
                }

                pMediaType = new AMMediaType();
                pMediaType.formatSize = cbType - Marshal.SizeOf(typeof(AMMediaType));

                //
                // Get the value for MediaType
                //
                pProps.GetMediaType(pMediaType, ref cbType);

                try
                {
                    if (MediaType.Audio != pMediaType.majorType)
                        throw new Exception("not Audio");
                    if (FormatType.WaveEx != pMediaType.formatType)
                        throw new Exception("not WaveEx");
                    var wfe = new WaveFormatExtensible(pcm);
                    Marshal.FreeCoTaskMem(pMediaType.formatPtr);
                    pMediaType.formatPtr = IntPtr.Zero;
                    pMediaType.formatSize = 0;
                    pMediaType.formatPtr = Marshal.AllocCoTaskMem(Marshal.SizeOf(wfe));
                    pMediaType.formatSize = Marshal.SizeOf(wfe);
                    Marshal.StructureToPtr(wfe, pMediaType.formatPtr, false);
                    pProps.SetMediaType(pMediaType);
                    m_syncReader.SetOutputProps(m_dwAudioOutputNum, pProps);
                }
                finally
                {
                    WMUtils.FreeWMMediaType(pMediaType);
                }
            }
            finally
            {
                Marshal.ReleaseComObject(pProps);
            }

            //try
            //{
            //    AttrDataType wmtType;
            //    short cbLength = 0;
            //    short wAnyStream = 0;
            //    var pHeaderInfo = m_syncReader as IWMHeaderInfo;
            //    pHeaderInfo.GetAttributeByName(ref wAnyStream, Constants.g_wszWMDuration, out wmtType, null, ref cbLength);
            //    var pbValue = new byte[cbLength];
            //    pHeaderInfo.GetAttributeByName(ref wAnyStream, Constants.g_wszWMDuration, out wmtType, pbValue, ref cbLength);
            //    var m_cnsFileDuration = BitConverter.ToInt64(pbValue, 0);
            //    _sampleCount = m_cnsFileDuration * m_pWfx.nSamplesPerSec / 10000000;
            //    // NOT ACCURATE ENOUGH (~1ms precision observed)
            //}
            //catch (COMException)
            //{
            //}

            //try
            //{
            //    var pHeaderInfo = m_syncReader as IWMHeaderInfo2;
            //    int nCodec;
            //    pHeaderInfo.GetCodecInfoCount(out nCodec);
            //    for (int wIndex = 0; wIndex < nCodec; wIndex++)
            //    {
            //        CodecInfoType enumCodecType;
            //        short cchName = 0;
            //        short cchDescription = 0;
            //        short cbCodecInfo = 0;
            //        pHeaderInfo.GetCodecInfo(wIndex, ref cchName, null,
            //            ref cchDescription, null, out enumCodecType,
            //            ref cbCodecInfo, null);
            //        var pwszName = new StringBuilder(cchName);
            //        var pwszDescription = new StringBuilder(cchDescription);
            //        var pbCodecInfo = new byte[cbCodecInfo];
            //        pHeaderInfo.GetCodecInfo(wIndex, ref cchName, pwszName,
            //            ref cchDescription, pwszDescription, out enumCodecType,
            //            ref cbCodecInfo, pbCodecInfo);
            //        if (enumCodecType == CodecInfoType.Audio)
            //        {
            //            // pbCodecInfo = {99,1} ??/
            //        }
            //    }
            //}
            //catch (COMException)
            //{
            //}

            //int cbMax;
            //m_syncReader.GetMaxOutputSampleSize(m_dwAudioOutputNum, out cbMax);
        }

        private DecoderSettings m_settings;
        public IAudioDecoderSettings Settings => null;

        public void isValid(string filename)
        {
            int pdwDataSize = 0;
            WMUtils.WMValidateData(null, ref pdwDataSize);
            byte[] data = new byte[pdwDataSize];
            using (FileStream s = new FileStream(filename, FileMode.Open, FileAccess.Read))
            {
                if (s.Read(data, 0, pdwDataSize) < pdwDataSize)
                    throw new Exception("partial read"); // TODO
            }
            WMUtils.WMValidateData(data, ref pdwDataSize);
        }

        public void Close()
        {
            //if (m_streamWrapper != null)
            //    m_streamWrapper.Close();
            if (m_IO != null)
                m_IO.Close();
            if (m_pSample != null)
                Marshal.ReleaseComObject(m_pSample);
            if (m_syncReader != null)
            {
                m_syncReader.Close();
                Marshal.ReleaseComObject(m_syncReader);
            }
            m_IO = null;
            m_pSample = null;
            m_syncReader = null;
        }

        public TimeSpan Duration => Length < 0 ? TimeSpan.Zero : TimeSpan.FromSeconds((double)Length / PCM.SampleRate);

        public long Length
        {
            get
            {
                return m_sampleCount;
            }
        }

        public long Remaining
        {
            get
            {
                return Length - Position;
            }
        }

        public long Position
        {
            get
            {
                return m_sampleOffset / PCM.BlockAlign;
            }
            set
            {
                if (m_sampleCount < 0 || value > m_sampleCount)
                    throw new Exception("seeking past end of stream");
                if (value < Position)
                    throw new NotSupportedException();
                if (value < Position)
                    throw new Exception("cannot seek backwards");
                var buff = new AudioBuffer(this, 0x10000);
                while (value > Position && Read(buff, (int)Math.Min(Int32.MaxValue, value - Position)) != 0)
                    ;
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
                return m_path;
            }
        }

        public int Read(AudioBuffer buff, int maxLength)
        {
            buff.Prepare(this, maxLength);

            int buff_offset = 0;
            int buff_size = buff.ByteLength;

            while (m_pSampleSize < buff_size)
            {
                if (m_pSampleSize > 0)
                {
                    IntPtr pdwBuffer;
                    m_pSample.GetBuffer(out pdwBuffer);
                    Marshal.Copy((IntPtr)(pdwBuffer.ToInt64() + m_pSampleOffset), buff.Bytes, buff_offset, m_pSampleSize);
                    buff_size -= m_pSampleSize;
                    buff_offset += m_pSampleSize;
                    m_sampleOffset += m_pSampleSize;
                    m_pSampleSize = 0;
                    Marshal.ReleaseComObject(m_pSample);
                    m_pSample = null;
                }

                long cnsSampleTime;
                long cnsDuration;
                SampleFlag flags;
                int dwOutputNum;
                short wStreamNum;
                try
                {
                    m_syncReader.GetNextSample(m_wStreamNum, out m_pSample, out cnsSampleTime, out cnsDuration, out flags, out dwOutputNum, out wStreamNum);
                }
                catch (COMException ex)
                {
                    // EOF
                    if (ex.ErrorCode == NSResults.E_NO_MORE_SAMPLES)
                    {
                        if ((m_sampleOffset % PCM.BlockAlign) != 0)
                            throw new Exception("(m_sampleOffset % PCM.BlockAlign) != 0");
                        m_sampleCount = m_sampleOffset / PCM.BlockAlign;
                        if ((buff_offset % PCM.BlockAlign) != 0)
                            throw new Exception("(buff_offset % PCM.BlockAlign) != 0");
                        return buff.Length = buff_offset / PCM.BlockAlign;
                    }
                    throw ex;
                }
                //if (dwOutputNum != m_dwAudioOutputNum || wStreamNum != m_wStreamNum)
                //{
                //}
                m_pSampleOffset = 0;
                m_pSample.GetLength(out m_pSampleSize);
            }

            if (buff_size > 0)
            {
                IntPtr pdwBuffer;
                m_pSample.GetBuffer(out pdwBuffer);
                Marshal.Copy((IntPtr)(pdwBuffer.ToInt64() + m_pSampleOffset), buff.Bytes, buff_offset, buff_size);
                m_pSampleOffset += buff_size;
                m_pSampleSize -= buff_size;
                m_sampleOffset += buff_size;
                buff_offset += buff_size;
                buff_size = 0;
            }
            if ((buff_offset % PCM.BlockAlign) != 0)
                throw new Exception("(buff_offset % PCM.BlockAlign) != 0");
            return buff.Length = buff_offset / PCM.BlockAlign;
        }
    }
}
