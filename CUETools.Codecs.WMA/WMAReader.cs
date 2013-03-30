/**
 * CUETools.WMA: WMA audio decoder
 * Copyright (c) 20139 Grigory Chudov
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
using System.Collections.Generic;
using System.Text;
using System.IO;
using WindowsMediaLib;
using WindowsMediaLib.Defs;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.WMA
{
    [AudioDecoderClass("builtin wma", "wma")]
    public class WMAReader : IAudioSource
    {
        IWMSyncReader m_syncReader;
        short m_wStreamNum = -1;
        int m_dwAudioOutputNum = -1;

        AudioPCMConfig pcm;

        byte[] samplesBuffer;
        int _samplesInBuffer, _samplesBufferOffset;
        long _sampleCount = -1, _sampleOffset = 0;

        string _path;
        //Stream _IO;

        public WMAReader(string path, Stream IO)
        {
            isValid(path);
            bool pfIsProtected;
            WMUtils.WMIsContentProtected(path, out pfIsProtected);
            if (pfIsProtected)
                throw new Exception("DRM present");
            WMUtils.WMCreateSyncReader(IntPtr.Zero, Rights.None, out m_syncReader);

            //m_syncReader.OpenStream()
            m_syncReader.Open(path);
            var pProfile = (m_syncReader as IWMProfile);
            int dwStreamCount;
            pProfile.GetStreamCount(out dwStreamCount);
            for (int dwIndex = 0; dwIndex < dwStreamCount; dwIndex++)
            {
                IWMStreamConfig pConfig = null;
                pProfile.GetStream(0, out pConfig);
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
                    var mt = new AMMediaType();
                    mt.formatSize = cbType;
                    pIWMMediaProps.GetMediaType(mt, ref cbType);
                    if (mt.formatType != FormatType.WaveEx)
                        continue;
                    if (mt.subType != MediaSubType.WMAudio_Lossless)
                        continue;
                    m_wStreamNum = wStreamNum;
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
            var m_pWfx = new WaveFormatEx();
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
                    Marshal.PtrToStructure(pMediaType.formatPtr, m_pWfx);
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

            pcm = new AudioPCMConfig(m_pWfx.wBitsPerSample, m_pWfx.nChannels, m_pWfx.nSamplesPerSec);
            _path = path;

            int cbMax;
            m_syncReader.GetMaxOutputSampleSize(m_dwAudioOutputNum, out cbMax);
            //m_syncReader.GetMaxStreamSampleSize(m_wStreamNum, out cbMax);
            //var ra = m_syncReader as IWMReaderAdvanced;
            //ra.SetAllocateForOutput(m_dwAudioOutputNum, true);

            samplesBuffer = new byte[cbMax];
        }

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
            //_IO.Close();
            m_syncReader.Close();
            Marshal.ReleaseComObject(m_syncReader);
        }

        public long Length
        {
            get
            {
                return _sampleCount;
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
                return _sampleOffset - _samplesInBuffer;
            }
            set
            {
                if (_sampleCount < 0 || value > _sampleCount)
                    throw new Exception("seeking past end of stream");
                throw new NotSupportedException();
                //if (value < Position || value > _sampleOffset)
                //{
                //    if (seek_table != null && _IO.CanSeek)
                //    {
                //        int best_st = -1;
                //        for (int st = 0; st < seek_table.Length; st++)
                //        {
                //            if (seek_table[st].number <= value &&
                //                (best_st == -1 || seek_table[st].number > seek_table[best_st].number))
                //                best_st = st;
                //        }
                //        if (best_st != -1)
                //        {
                //            _framesBufferLength = 0;
                //            _samplesInBuffer = 0;
                //            _samplesBufferOffset = 0;
                //            _IO.Position = (long)seek_table[best_st].offset + first_frame_offset;
                //            _sampleOffset = seek_table[best_st].number;
                //        }
                //    }
                //    if (value < Position)
                //        throw new Exception("cannot seek backwards without seek table");
                //}
                //while (value > _sampleOffset)
                //{
                //    _samplesInBuffer = 0;
                //    _samplesBufferOffset = 0;

                //    fill_frames_buffer();
                //    if (_framesBufferLength == 0)
                //        throw new Exception("seek failed");

                //    int bytesDecoded = DecodeFrame(_framesBuffer, _framesBufferOffset, _framesBufferLength);
                //    _framesBufferLength -= bytesDecoded;
                //    _framesBufferOffset += bytesDecoded;

                //    _sampleOffset += _samplesInBuffer;
                //};
                //int diff = _samplesInBuffer - (int)(_sampleOffset - value);
                //_samplesInBuffer -= diff;
                //_samplesBufferOffset += diff;
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
                return _path;
            }
        }

        const int BS = 4;

        public int Read(AudioBuffer buff, int maxLength)
        {
            buff.Prepare(this, maxLength);

            int offset = 0;
            int sampleCount = buff.Length;

            while (_samplesInBuffer < sampleCount)
            {
                if (_samplesInBuffer > 0)
                {
                    Array.Copy(samplesBuffer, _samplesBufferOffset * BS, buff.Bytes, offset * BS, _samplesInBuffer * BS);
                    sampleCount -= _samplesInBuffer;
                    offset += _samplesInBuffer;
                    _samplesInBuffer = 0;
                    _samplesBufferOffset = 0;
                }

                INSSBuffer pSample;
                long cnsSampleTime;
                long cnsDuration;
                SampleFlag flags;
                int dwOutputNum;
                short wStreamNum;
                int dwLength;
                IntPtr pdwBuffer;
                try
                {
                    m_syncReader.GetNextSample(m_wStreamNum, out pSample, out cnsSampleTime, out cnsDuration, out flags, out dwOutputNum, out wStreamNum);
                }
                catch (COMException ex)
                {
                    // EOF
                    if (ex.ErrorCode == NSResults.E_NO_MORE_SAMPLES)
                    {
                        _sampleCount = _sampleOffset;
                        return offset;
                    }
                    throw ex;
                }
                //if (dwOutputNum != m_dwAudioOutputNum || wStreamNum != m_wStreamNum)
                //{
                //}
                pSample.GetBufferAndLength(out pdwBuffer, out dwLength);
                Marshal.Copy(pdwBuffer, samplesBuffer, 0, dwLength);
                Marshal.ReleaseComObject(pSample);
                _samplesInBuffer = dwLength / BS;
                _sampleOffset += _samplesInBuffer;

                //_samplesInBuffer -= _samplesBufferOffset; // can be set by Seek, otherwise zero
                //_sampleOffset += _samplesInBuffer;
            }

            Array.Copy(samplesBuffer, _samplesBufferOffset * BS, buff.Bytes, offset * BS, sampleCount * BS);
            _samplesInBuffer -= sampleCount;
            _samplesBufferOffset += sampleCount;
            if (_samplesInBuffer == 0)
                _samplesBufferOffset = 0;
            return offset + sampleCount;
        }
    }
}
