using System;
using System.Runtime.InteropServices;

namespace CUETools.Codecs.LAME.Interop
{
    [StructLayout(LayoutKind.Sequential, Size = 327), Serializable]
    public struct LHV1 // BE_CONFIG_LAME LAME header version 1
    {
        public const uint MPEG1 = 1;
        public const uint MPEG2 = 0;

        // STRUCTURE INFORMATION
        public uint dwStructVersion;
        public uint dwStructSize;
        // BASIC ENCODER SETTINGS
        public uint dwSampleRate;		// SAMPLERATE OF INPUT FILE
        public uint dwReSampleRate;		// DOWNSAMPLERATE, 0=ENCODER DECIDES	
        public MpegMode nMode;	// STEREO, MONO
        public uint dwBitrate;			// CBR bitrate, VBR min bitrate
        public uint dwMaxBitrate;		// CBR ignored, VBR Max bitrate
        public LAME_QUALITY_PRESET nPreset;			// Quality preset
        public uint dwMpegVersion;		// MPEG-1 OR MPEG-2
        public uint dwPsyModel;			// FUTURE USE, SET TO 0
        public uint dwEmphasis;			// FUTURE USE, SET TO 0
        // BIT STREAM SETTINGS
        public int bPrivate;			// Set Private Bit (TRUE/FALSE)
        public int bCRC;	// Insert CRC (TRUE/FALSE)
        public int bCopyright;			// Set Copyright Bit (TRUE/FALSE)
        public int bOriginal;			// Set Original Bit (TRUE/FALSE)
        // VBR STUFF
        public int bWriteVBRHeader;	// WRITE XING VBR HEADER (TRUE/FALSE)
        public int bEnableVBR;			// USE VBR ENCODING (TRUE/FALSE)
        public int nVBRQuality;		// VBR QUALITY 0..9
        public uint dwVbrAbr_bps;		// Use ABR in stead of nVBRQuality
        public VBRMETHOD nVbrMethod;
        public int bNoRes;	// Disable Bit resorvoir (TRUE/FALSE)
        // MISC SETTINGS
        public int bStrictIso;			// Use strict ISO encoding rules (TRUE/FALSE)
        public ushort nQuality;			// Quality Setting, HIGH BYTE should be NOT LOW byte, otherwhise quality=5
        // FUTURE USE, SET TO 0, align strucutre to 331 bytes
        //[ MarshalAs( UnmanagedType.ByValArray, SizeConst=255-4*4-2 )]
        //public byte[]	 btReserved;//[255-4*sizeof(DWORD) - sizeof( WORD )];

        public LHV1(AudioPCMConfig format, uint MpeBitRate, uint quality)
        {
            dwStructVersion = 1;
            dwStructSize = (uint)Marshal.SizeOf(typeof(BE_CONFIG));
            switch (format.SampleRate)
            {
                case 16000:
                case 22050:
                case 24000:
                    dwMpegVersion = MPEG2;
                    break;
                case 32000:
                case 44100:
                case 48000:
                    dwMpegVersion = MPEG1;
                    break;
                default:
                    throw new ArgumentOutOfRangeException("format", "Unsupported sample rate");
            }
            dwSampleRate = (uint)format.SampleRate;	// INPUT FREQUENCY
            dwReSampleRate = 0;		// DON'T RESAMPLE
            switch (format.ChannelCount)
            {
                case 1:
                    nMode = MpegMode.MONO;
                    break;
                case 2:
                    nMode = MpegMode.STEREO;
                    break;
                default:
                    throw new ArgumentOutOfRangeException("format", "Invalid number of channels");
            }
            switch (MpeBitRate)
            {
                case 0:
                case 32:
                case 40:
                case 48:
                case 56:
                case 64:
                case 80:
                case 96:
                case 112:
                case 128:
                case 160: //Allowed bit rates in MPEG1 and MPEG2
                    break;
                case 192:
                case 224:
                case 256:
                case 320: //Allowed only in MPEG1
                    if (dwMpegVersion != MPEG1)
                    {
                        throw new ArgumentOutOfRangeException("MpsBitRate", "Bit rate not compatible with input format");
                    }
                    break;
                case 8:
                case 16:
                case 24:
                case 144: //Allowed only in MPEG2
                    if (dwMpegVersion != MPEG2)
                    {
                        throw new ArgumentOutOfRangeException("MpsBitRate", "Bit rate not compatible with input format");
                    }
                    break;
                default:
                    throw new ArgumentOutOfRangeException("MpsBitRate", "Unsupported bit rate");
            }
            dwBitrate = MpeBitRate;		// MINIMUM BIT RATE
            nPreset = LAME_QUALITY_PRESET.LQP_NORMAL_QUALITY;		// QUALITY PRESET SETTING
            dwPsyModel = 0;		// USE DEFAULT PSYCHOACOUSTIC MODEL 
            dwEmphasis = 0;		// NO EMPHASIS TURNED ON
            bOriginal = 1;		// SET ORIGINAL FLAG
            bWriteVBRHeader = 0;
            bNoRes = 0;		// No Bit resorvoir
            bCopyright = 0;
            bCRC = 0;
            bEnableVBR = 0;
            bPrivate = 0;
            bStrictIso = 0;
            dwMaxBitrate = 0;
            dwVbrAbr_bps = 0;
            nQuality = (ushort)(quality | ((~quality) << 8));
            nVbrMethod = VBRMETHOD.VBR_METHOD_NONE;
            nVBRQuality = 0;
        }
    }
}
