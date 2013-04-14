using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;
using WindowsMediaLib.Defs;

namespace CUETools.Codecs.WMA
{
    /// <summary>
    /// From WAVEFORMATEXTENSIBLE
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 2)]
    public struct WaveFormatExtensible
    {
        public const ushort WAVE_FORMAT_EXTENSIBLE = 0xFFFE;
        public const ushort WAVE_FORMAT_PCM = 1;

        public WaveFormatExtensible(AudioPCMConfig pcm)
        {
            this.nChannels = (short)pcm.ChannelCount;
            this.nSamplesPerSec = pcm.SampleRate;
            this.nBlockAlign = (short)pcm.BlockAlign;
            this.wBitsPerSample = (short)pcm.BitsPerSample;
            if ((pcm.BitsPerSample == 8 || pcm.BitsPerSample == 16 || pcm.BitsPerSample == 24) &&
                (pcm.ChannelCount == 1 || pcm.ChannelCount == 2))
            {
                this.wFormatTag = unchecked((short)WAVE_FORMAT_PCM);
                this.cbSize = 0;
                // just to make compiler happy
                this.wValidBitsPerSample = 0;
                this.dwChannelMask = 0;
                this.SubFormat = Guid.Empty;
            }
            else
            {
                this.wFormatTag = unchecked((short)WAVE_FORMAT_EXTENSIBLE);
                this.cbSize = 22;
                this.wValidBitsPerSample = this.wBitsPerSample;
                this.nBlockAlign = (short)((this.wBitsPerSample / 8) * this.nChannels);
                this.dwChannelMask = (int)pcm.ChannelMask;
                this.SubFormat = MediaSubType.PCM;
            }
            this.nAvgBytesPerSec = this.nSamplesPerSec * this.nBlockAlign;
        }

        public static WaveFormatExtensible FromMediaType(AMMediaType pMediaType)
        {
            if (MediaType.Audio != pMediaType.majorType)
                throw new Exception("not Audio");
            if (FormatType.WaveEx != pMediaType.formatType || pMediaType.formatSize < 18)
                throw new Exception("not WaveEx");
            WaveFormatEx pWfx = new WaveFormatEx();
            Marshal.PtrToStructure(pMediaType.formatPtr, pWfx);
            if (pWfx.wFormatTag == unchecked((short)WAVE_FORMAT_EXTENSIBLE) && pWfx.cbSize >= 22)
            {
                var pWfe = new WaveFormatExtensible();
                Marshal.PtrToStructure(pMediaType.formatPtr, pWfe);
                return pWfe;
            }
            return new WaveFormatExtensible() {
                nChannels = pWfx.nChannels, 
                nSamplesPerSec = pWfx.nSamplesPerSec, 
                nBlockAlign = pWfx.nBlockAlign, 
                wBitsPerSample = pWfx.wBitsPerSample, 
                nAvgBytesPerSec = pWfx.nAvgBytesPerSec, 
                wFormatTag = pWfx.wFormatTag, 
                cbSize = 0
            };
        }
        
        public AudioPCMConfig GetConfig()
        {
            return new AudioPCMConfig(
                wBitsPerSample,
                nChannels,
                nSamplesPerSec,
                (AudioPCMConfig.SpeakerConfig)(wFormatTag == unchecked((short)WAVE_FORMAT_EXTENSIBLE) && cbSize >= 22 ? dwChannelMask : 0));
        }

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
}
