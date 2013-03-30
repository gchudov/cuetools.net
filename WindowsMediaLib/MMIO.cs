#region license

/*
WindowsMediaLib - Provide access to Windows Media interfaces via .NET
Copyright (C) 2008
http://sourceforge.net/projects/windowsmedianet

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
*/

#endregion

using System;
using System.Text;
using System.Runtime.InteropServices;
using System.Security;

using WindowsMediaLib;
using WindowsMediaLib.Defs;

namespace MultiMedia
{
    #region Enums

    [Flags]
    public enum WaveOpenFlags
    {
        None = 0,
        FormatQuery = 0x0001,
        AllowSync = 0x0002,
        Mapped = 0x0004,
        FormatDirect = 0x0008,
        Null = 0x00000000,      /* no callback */
        Window = 0x00010000,    /* dwCallback is a HWND */
        Thread = 0x00020000,    /* dwCallback is a THREAD */
        Function = 0x00030000,  /* dwCallback is a FARPROC */
        Event = 0x00050000      /* dwCallback is an EVENT Handle */
    }

    [Flags]
    public enum SupportedFormats
    {
        InvalidFormat = 0x00000000,       /* invalid format */
        F1M08 = 0x00000001,  /* 11.025 kHz, Mono,   8-bit  */
        F1S08 = 0x00000002,  /* 11.025 kHz, Stereo, 8-bit  */
        F1M16 = 0x00000004,  /* 11.025 kHz, Mono,   16-bit */
        F1S16 = 0x00000008,  /* 11.025 kHz, Stereo, 16-bit */
        F2M08 = 0x00000010,  /* 22.05  kHz, Mono,   8-bit  */
        F2S08 = 0x00000020,  /* 22.05  kHz, Stereo, 8-bit  */
        F2M16 = 0x00000040,  /* 22.05  kHz, Mono,   16-bit */
        F2S16 = 0x00000080,  /* 22.05  kHz, Stereo, 16-bit */

        F44M08 = 0x00000100,   /* 44.1   kHz, Mono,   8-bit  */
        F44S08 = 0x00000200,   /* 44.1   kHz, Stereo, 8-bit  */
        F44M16 = 0x00000400,   /* 44.1   kHz, Mono,   16-bit */
        F44S16 = 0x00000800,   /* 44.1   kHz, Stereo, 16-bit */
        F48M08 = 0x00001000,   /* 48     kHz, Mono,   8-bit  */
        F48S08 = 0x00002000,   /* 48     kHz, Stereo, 8-bit  */
        F48M16 = 0x00004000,   /* 48     kHz, Mono,   16-bit */
        F48S16 = 0x00008000,   /* 48     kHz, Stereo, 16-bit */
        F96M08 = 0x00010000,   /* 96     kHz, Mono,   8-bit  */
        F96S08 = 0x00020000,   /* 96     kHz, Stereo, 8-bit  */
        F96M16 = 0x00040000,   /* 96     kHz, Mono,   16-bit */
        F96S16 = 0x00080000,   /* 96     kHz, Stereo, 16-bit */
    }

    public enum MOWM
    {
        WOM_OPEN = 0x3BB,
        WOM_CLOSE = 0x3BC,
        WOM_DONE = 0x3BD,
    }

    public enum MIWM
    {
        WIM_OPEN = 0x3BE,           /* waveform input */
        WIM_CLOSE = 0x3BF,
        WIM_DATA = 0x3C0
    }

    [Flags]
    public enum WaveCapsFlags
    {
        None = 0,
        Pitch = 0x0001,         /* supports pitch control */
        PlaybackRate = 0x0002,  /* supports playback rate control */
        Volume = 0x0004,        /* supports volume control */
        LRVolume = 0x0008,      /* separate left-right volume control */
        Sync = 0x0010,
        SampleAccurate = 0x0020
    }

    [Flags]
    public enum RiffChunkFlags
    {
        None = 0,
        FindChunk = 0x0010,     /* mmioDescend: find a chunk by ID */
        FindRiff = 0x0020,      /* mmioDescend: find a LIST chunk */
        FindList = 0x0040,      /* mmioDescend: find a RIFF chunk */
        CreateRiff = 0x0020,    /* mmioCreateChunk: make a LIST chunk */
        CreateList = 0x0040     /* mmioCreateChunk: make a RIFF chunk */
    }

    [Flags]
    public enum MMIOCloseFlags
    {
        None = 0,
        FHOPEN = 0x0010  /* mmioClose: keep file handle open */
    }

    [Flags]
    public enum MMIOFlushFlags
    {
        None = 0,
        EmptyBuf = 0x0010  /* mmioFlush: empty the I/O buffer */
    }

    public enum MMIOSeekFlags
    {
        Set = 0,
        Cur = 1,
        End = 2
    }

    public enum RWMode
    {
        Read = 0x00000000,          /* open file for reading only */
        Write = 0x00000001,         /* open file for writing only */
        ReadWrite = 0x00000002      /* open file for reading and writing */
    }

    public enum MMIOError
    {
        NoError = 0,
        FileNotFound = 257,     /* file not found */
        OutOfMemory = 258,      /* out of memory */
        CannotOpen = 259,       /* cannot open */
        CannotClose = 260,      /* cannot close */
        CannotRead = 261,       /* cannot read */
        CannotWrite = 262,      /* cannot write */
        CannotSeek = 263,       /* cannot seek */
        CannotExpand = 264,     /* cannot expand file */
        ChunkNotFound = 265,    /* chunk not found */
        Unbuffered = 266,       /*  */
        PathNotFound = 267,     /* path incorrect */
        AccessDenied = 268,     /* file was protected */
        SharingViolation = 269, /* file in use */
        NetworkError = 270,     /* network not responding */
        TooManyOpenFiles = 271, /* no more file handles  */
        InvalidFile = 272       /* default error file error */
    }

    [Flags]
    public enum MMIOFlags
    {
        /* constants for dwFlags field of MMIOINFO */
        Create = 0x00001000,        /* create new file (or truncate file) */
        Parse = 0x00000100,         /* parse new file returning path */
        Delete = 0x00000200,        /* create new file (or truncate file) */
        Exist = 0x00004000,         /* checks for existence of file */
        AllocBuf = 0x00010000,      /* mmioOpen() should allocate a buffer */
        GetTemp = 0x00020000,       /* mmioOpen() should retrieve temp name */

        Dirty = 0x10000000,         /* I/O buffer is dirty */

        /* read/write mode numbers (bit field MMIO_RWMODE) */
        Read = 0x00000000,          /* open file for reading only */
        Write = 0x00000001,         /* open file for writing only */
        ReadWrite = 0x00000002,     /* open file for reading and writing */

        /* share mode numbers (bit field MMIO_SHAREMODE) */
        Compat = 0x00000000,        /* compatibility mode */
        Exclusive = 0x00000010,     /* exclusive-access mode */
        DenyWrite = 0x00000020,     /* deny writing to other processes */
        DenyRead = 0x00000030,      /* deny reading to other processes */
        DenyNone = 0x00000040,      /* deny nothing to other processes */
    }

    // From MIXER_GETLINEINFOF_* defines and MIXER_OBJECTF_* defines
    [Flags]
    public enum MIXER_GETLINEINFOF
    {
        Destination = 0x00000000,
        Source = 0x00000001,
        LineID = 0x00000002,
        ComponentType = 0x00000003,
        TargetType = 0x00000004,

        Mixer = 0x00000000,
        WaveOut = 0x10000000,
        WaveIn = 0x20000000,
        MidiOut = 0x30000000,
        MidiIn = 0x40000000,
        Aux = 0x50000000,
        Handle = unchecked((int)0x80000000),
        HMidiIn = (Handle | MidiIn),
        HMidiOut = (Handle | MidiOut),
        HMixer = (Handle | Mixer),
        HWaveIn = (Handle | WaveIn),
        HWaveOut = (Handle | WaveOut)
    }

    // From MIXER_SETCONTROLDETAILSF_* defines and MIXER_OBJECTF_* defines
    [Flags]
    public enum MIXER_SETCONTROLDETAILSF
    {
        Value = 0x00000000,
        Custom = 0x00000001,

        QueryMask = 0x0000000F,

        Mixer = 0x00000000,
        WaveOut = 0x10000000,
        WaveIn = 0x20000000,
        MidiOut = 0x30000000,
        MidiIn = 0x40000000,
        Aux = 0x50000000,
        Handle = unchecked((int)0x80000000),
        HMidiIn = (Handle | MidiIn),
        HMidiOut = (Handle | MidiOut),
        HMixer = (Handle | Mixer),
        HWaveIn = (Handle | WaveIn),
        HWaveOut = (Handle | WaveOut)
    }

    // From MIXER_GETLINECONTROLSF_* defines combined with MIXER_OBJECTF_* defines
    [Flags]
    public enum MIXER_GETLINECONTROLSF
    {
        All = 0x00000000,
        OneByID = 0x00000001,
        OneByType = 0x00000002,
        QueryMask = 0x0000000F,

        Mixer = 0x00000000,
        WaveOut = 0x10000000,
        WaveIn = 0x20000000,
        MidiOut = 0x30000000,
        MidiIn = 0x40000000,
        Aux = 0x50000000,
        Handle = unchecked((int)0x80000000),
        HMidiIn = (Handle | MidiIn),
        HMidiOut = (Handle | MidiOut),
        HMixer = (Handle | Mixer),
        HWaveIn = (Handle | WaveIn),
        HWaveOut = (Handle | WaveOut)
    }

    // From MIXER_OBJECTF_* defines
    [Flags]
    public enum MIXER_OBJECTF
    {
        GetControlDetailsF_Value = 0x0,
        GetControlDetailsF_ListText = 0x00000001,

        CallBack_Window = 0x00010000,    /* dwCallback is an HWND */

        Mixer = 0x00000000,
        WaveOut = 0x10000000,
        WaveIn = 0x20000000,
        MidiOut = 0x30000000,
        MidiIn = 0x40000000,
        Aux = 0x50000000,
        Handle = unchecked((int)0x80000000),
        HMidiIn = (Handle | MidiIn),
        HMidiOut = (Handle | MidiOut),
        HMixer = (Handle | Mixer),
        HWaveIn = (Handle | WaveIn),
        HWaveOut = (Handle | WaveOut)
    }

    // From MIXERCONTROL_CONTROLF_* defines
    [Flags]
    public enum MIXERCONTROL_CONTROLF
    {
        None = 0,
        Uniform = 0x00000001,
        Multiple = 0x00000002,
        Disabled = unchecked((int)0x80000000)
    }

    // From MIXERCONTROL_CONTROLTYPE defines
    [Flags]
    public enum MIXERCONTROL_CONTROLTYPE
    {
        ClassFader = 0x50000000,
        ClassList = 0x70000000,
        SCListSingle = 0x00000000,
        UnitsBoolean = 0x00010000,
        UnitsUnsigned = 0x30000,
        SCListMultiple = 0x01000000,

        Fader = (ClassFader | UnitsUnsigned),
        MultipleSelect = (ClassList | SCListMultiple | UnitsBoolean),
        SingleSelect = (ClassList | SCListSingle | UnitsBoolean),
        Mixer = (MultipleSelect + 1),
        Mux = (SingleSelect + 1),
        Volume = (Fader + 1)
    }

    // From MIXERCONTROL_CT_* defines combined with MIXERCONTROL_CONTROLTYPE_* defines
    [Flags]
    public enum ControlType
    {
        Class_Mask = unchecked((int)0xF0000000),
        Class_Custom = 0x00000000,
        Class_Meter = 0x10000000,
        Class_Switch = 0x20000000,
        Class_Number = 0x30000000,
        Class_Slider = 0x40000000,
        Class_Fader = 0x50000000,
        Class_Time = 0x60000000,
        Class_List = 0x70000000,

        SubClassMask = 0x0F000000,

        SC_SwitchBoolean = 0x00000000,
        SC_SwitchButton = 0x01000000,

        SC_MeterPolled = 0x00000000,

        SC_TimeMicroSecs = 0x00000000,
        SC_TimeMilliSecs = 0x01000000,

        SC_ListSingle = 0x00000000,
        SC_ListMultiple = 0x01000000,

        Units_Mask = 0x00FF0000,
        Units_Custom = 0x00000000,
        Units_Boolean = 0x00010000,
        Units_Signed = 0x00020000,
        Units_Unsigned = 0x00030000,
        Units_Decibels = 0x00040000, /* in 10ths */
        Units_Percent = 0x00050000, /* in 10ths */

        Custom = (Class_Custom | Units_Custom),
        BooleanMeter = (Class_Meter | SC_MeterPolled | Units_Boolean),
        SignedMeter = (Class_Meter | SC_MeterPolled | Units_Signed),
        PeakMeter = (SignedMeter + 1),
        UnsignedMeter = (Class_Meter | SC_MeterPolled | Units_Unsigned),
        Boolean = (Class_Switch | SC_SwitchBoolean | Units_Boolean),
        OnOff = (Boolean + 1),
        Mute = (Boolean + 2),
        Mono = (Boolean + 3),
        Loudness = (Boolean + 4),
        StereoEnh = (Boolean + 5),
        BassBoost = (Boolean + 0x00002277),
        Button = (Class_Switch | SC_SwitchButton | Units_Boolean),
        Decibels = (Class_Number | Units_Decibels),
        Signed = (Class_Number | Units_Signed),
        Unsigned = (Class_Number | Units_Unsigned),
        Percent = (Class_Number | Units_Percent),
        Slider = (Class_Slider | Units_Signed),
        Pan = (Slider + 1),
        QSoundPan = (Slider + 2),
        Fader = (Class_Fader | Units_Unsigned),
        Volume = (Fader + 1),
        Bass = (Fader + 2),
        Treble = (Fader + 3),
        Equalizer = (Fader + 4),
        SingleSelect = (Class_List | SC_ListSingle | Units_Boolean),
        Mux = (SingleSelect + 1),
        MultipleSelect = (Class_List | SC_ListMultiple | Units_Boolean),
        Mixer = (MultipleSelect + 1),
        MicroTime = (Class_Time | SC_TimeMicroSecs | Units_Unsigned),
        MilliTime = (Class_Time | SC_TimeMilliSecs | Units_Unsigned)
    }

    // From MIXERLINE_TARGETTYPE_* defines
    public enum MIXERLINE_TARGETTYPE
    {
        Undefined = 0,
        WaveOut,
        WaveIn,
        MidiOut,
        MidiIn,
        Aux
    }

    // From MIXERLINE_COMPONENTTYPE_* defines
    public enum MIXERLINE_COMPONENTTYPE
    {
        DST_First = 0x00000000,
        DST_Undefined = (DST_First + 0),
        DST_Digital = (DST_First + 1),
        DST_Line = (DST_First + 2),
        DST_Monitor = (DST_First + 3),
        DST_Speakers = (DST_First + 4),
        DST_Headphones = (DST_First + 5),
        DST_Telephone = (DST_First + 6),
        DST_WaveIn = (DST_First + 7),
        DST_VoiceIn = (DST_First + 8),
        DST_Last = (DST_First + 8),

        SRC_First = 0x00001000,
        SRC_Undefined = (SRC_First + 0),
        SRC_Digital = (SRC_First + 1),
        SRC_Line = (SRC_First + 2),
        SRC_Microphone = (SRC_First + 3),
        SRC_Synthesizer = (SRC_First + 4),
        SRC_CompactDisc = (SRC_First + 5),
        SRC_Telephone = (SRC_First + 6),
        SRC_PCSpeaker = (SRC_First + 7),
        SRC_WaveOut = (SRC_First + 8),
        SRC_Auxillary = (SRC_First + 9),
        SRC_Analog = (SRC_First + 10),
        SRC_Last = (SRC_First + 10),

        Invalid = -1
    }

    // From MIXERLINE_LINEF_* defines
    [Flags]
    public enum MIXERLINE_LINEF
    {
        None = 0,
        Active = 0x00000001,
        Disconnected = 0x00008000,
        Source = unchecked((int)0x80000000)
    }

    // From MMSYSERR_* defines
    public enum MMSYSERR
    {
        NoError = 0,    /* no error */
        Error,          /* unspecified error */
        BadDeviceID,    /* device ID out of range */
        NotEnabled,     /* driver failed enable */
        Allocated,      /* device already allocated */
        InvalHandle,    /* device handle is invalid */
        NoDriver,       /* no device driver present */
        NoMem,          /* memory allocation error */
        NotSupported,   /* function isn't supported */
        BadErrNum,      /* error value out of range */
        InvalFlag,      /* invalid flag passed */
        InvalParam,     /* invalid parameter passed */
        HandleBusy,     /* handle being used simultaneously on another thread (eg callback) */
        InvalidAlias,   /* specified alias not found */
        BadDB,          /* bad registry database */
        KeyNotFound,    /* registry key not found */
        ReadError,      /* registry read error */
        WriteError,     /* registry write error */
        DeleteError,    /* registry delete error */
        ValNotFound,    /* registry value not found */
        NoDriverCB,     /* driver does not call DriverCallback */
        MoreData,       /* more data to be returned */

        InvalLine = (1024 + 0),
        InvalControl,
        InvalValue
    }

    #endregion

    #region Structs

    [StructLayout(LayoutKind.Sequential, Pack = 4, CharSet = CharSet.Unicode),
    UnmanagedName("WAVEOUTCAPSW")]
    public class WaveOutCaps
    {
        public short wMid;                  /* manufacturer ID */
        public short wPid;                  /* product ID */
        public int vDriverVersion;          /* version of the driver */
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
        public string szPname;              /* product name (NULL terminated string) */
        public SupportedFormats dwFormats;  /* formats supported */
        public short wChannels;             /* number of sources supported */
        public short wReserved1;            /* packing */
        public WaveCapsFlags dwSupport;     /* functionality supported by driver */
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4, CharSet = CharSet.Unicode),
    UnmanagedName("WAVEINCAPSW")]
    public class WaveInCaps
    {
        public short wMid;
        public short wPid;
        public int vDriverVersion;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
        public string szPname;
        public SupportedFormats dwFormats;
        public short wChannels;
        public short wReserved1;
    }

    [StructLayout(LayoutKind.Sequential), UnmanagedName("MMTIME")]
    public class MMTIME
    {
        [Flags]
        public enum MMTimeFlags
        {
            MS = 0x0001,        /* time in milliseconds */
            Samples = 0x0002,   /* number of wave samples */
            Bytes = 0x0004,     /* current byte offset */
            SMPTE = 0x0008,     /* SMPTE time */
            Midi = 0x0010,      /* MIDI time */
            Ticks = 0x0020      /* Ticks within MIDI stream */
        }

        public MMTimeFlags wType;
        public int u;
        public int x;
    }

    [StructLayout(LayoutKind.Sequential), UnmanagedName("WAVEHDR")]
    public class WAVEHDR : IDisposable
    {
        [Flags]
        public enum WHDR
        {
            None = 0x0,
            Done = 0x00000001,      /* done bit */
            Prepared = 0x00000002,  /* set if this header has been prepared */
            BeginLoop = 0x00000004, /* loop start block */
            EndLoop = 0x00000008,   /* loop end block */
            InQueue = 0x00000010    /* reserved for driver */
        }

        public IntPtr lpData;
        public int dwBufferLength;
        public int dwBytesRecorded;
        public IntPtr dwUser;
        public WHDR dwFlags;
        public int dwLoops;
        public IntPtr lpNext;
        public IntPtr Reserved;

        public WAVEHDR()
        {
        }

        public WAVEHDR(int iMaxSize)
        {
            lpData = Marshal.AllocCoTaskMem(iMaxSize);
            dwBufferLength = iMaxSize;
            dwUser = IntPtr.Zero;
            dwFlags = WHDR.None;
            dwLoops = 0;
            lpNext = IntPtr.Zero;
            Reserved = IntPtr.Zero;
        }

        #region IDisposable Members

        public void Dispose()
        {
            if (lpData != IntPtr.Zero)
            {
                Marshal.FreeCoTaskMem(lpData);
                lpData = IntPtr.Zero;
            }
        }

        #endregion
    }

    [StructLayout(LayoutKind.Sequential), UnmanagedName("MMCKINFO")]
    public class MMCKINFO
    {
        public FourCC ckid;
        public int ckSize;
        public FourCC fccType;
        public int dwDataOffset;
        public MMIOFlags dwFlags;
    }

    [StructLayout(LayoutKind.Sequential, Pack=1), UnmanagedName("MMIOINFO")]
    public class MMIOINFO
    {
        public MMIOFlags dwFlags;
        public FourCC fccIOProc;
        public IntPtr pIOProc;
        public MMIOError wErrorRet;
        public IntPtr htask;
        public int cchBuffer;
        public IntPtr pchBuffer;
        public IntPtr pchNext;
        public IntPtr pchEndRead;
        public IntPtr pchEndWrite;
        public int lBufOffset;
        public int lDiskOffset;
        public int adwInfo1;
        public int adwInfo2;
        public int adwInfo3;
        public int dwReserved1;
        public int dwReserved2;
        public IntPtr hmmio;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Unicode), UnmanagedName("MIXERCAPS")]
    public class MixerCaps
    {
        public const int MAXPNAMELEN = 32;

        public short wMid;
        public short wPid;
        public int vDriverVersion;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = MAXPNAMELEN)]
        public string szPname;
        public int fdwSupport; // Zero
        public int cDestinations;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1), UnmanagedName("MIXERCONTROLDETAILS")]
    public class MixerControlDetails
    {
        public int cbStruct;        //  size in Bytes of MIXERCONTROLDETAILS
        public int dwControlID;     // control id to get/set details on
        public int cChannels;       // number of channels in paDetails array
        public MCDUnion item;       // hwndOwner or cMultipleItems
        public int cbDetails;       // size of _one_ details_XX struct
        public IntPtr paDetails;    // pointer to array of details_XX structs

        public MixerControlDetails()
        {
            item = new MCDUnion();
        }
    }

    [StructLayout(LayoutKind.Explicit, Pack = 1)]
    public class MCDUnion
    {
        [FieldOffset(0)]
        public IntPtr hwndOwner;
        [FieldOffset(0)]
        public int cMultipleItems;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1), UnmanagedName("MIXERLINECONTROLS")]
    public class MixerLineControls
    {
        public int cbStruct;       //  size in Byte of MIXERLINECONTROLS
        public int dwLineID;       //  line id (from MIXERLINE.dwLineID)
        public int dwControl;      //  MIXER_GETLINECONTROLSF_ONEBYID or MIXER_GETLINECONTROLSF_ONEBYTYPE
        public int cControls;      //  count of controls pmxctrl points to
        public int cbmxctrl;       //  size in Byte of _one_ MIXERCONTROL
        public MixerControl[] pamxctrl;       //  pointer to first MIXERCONTROL array
    }

    [StructLayout(LayoutKind.Sequential), UnmanagedName("MIXERCONTROLDETAILS_UNSIGNED")]
    public class MixerControlDetailsUnsigned
    {
        public int dwValue;        //  value of the control
    }

    [StructLayout(LayoutKind.Sequential), UnmanagedName("MIXERCONTROLDETAILS_SIGNED")]
    public struct MixerControlDetailsSigned
    {
        public int dwValue;        //  value of the control
    }

    [StructLayout(LayoutKind.Sequential), UnmanagedName("MIXERCONTROLDETAILS_BOOLEAN")]
    public struct MixerControlDetailsBoolean
    {
        [MarshalAs(UnmanagedType.Bool)]
        public bool fValue;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Unicode), UnmanagedName("MIXERCONTROLDETAILS_LISTTEXT")]
    public struct MixerControlDetailsListText
    {
        public const int MIXER_LONG_NAME_CHARS = 64;

        public int dwParam1;
        public int dwParam2;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = MIXER_LONG_NAME_CHARS)]
        public string szName;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Unicode), UnmanagedName("MIXERLINE")]
    public class MixerLine
    {
        public const int MIXER_SHORT_NAME_CHARS = 16;
        public const int MIXER_LONG_NAME_CHARS = 64;
        public const int MAXPNAMELEN = 32;

        public int cbStruct;               //  size of MIXERLINE structure
        public int dwDestination;          //  zero based destination index
        public int dwSource;               //  zero based source index (if source)
        public int dwLineID;               //  unique line id for mixer device
        public MIXERLINE_LINEF fdwLine;    //  state/information about line
        public IntPtr dwUser;              //  driver specific information
        public MIXERLINE_COMPONENTTYPE dwComponentType;        //  component type line connects to
        public int cChannels;              //  number of channels line supports
        public int cConnections;           //  number of connections (possible)
        public int cControls;              //  number of controls at this line
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = MIXER_SHORT_NAME_CHARS)]
        public string szShortName;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = MIXER_LONG_NAME_CHARS)]
        public string szName;
        public MIXERLINE_TARGETTYPE dwType;
        public int dwDeviceID;
        public short wMid;
        public short wPid;
        public int vDriverVersion;
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = MAXPNAMELEN)]
        public string szPname;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Unicode), UnmanagedName("MIXERCONTROL")]
    public class MixerControl
    {
        public const int MIXER_SHORT_NAME_CHARS = 16;
        public const int MIXER_LONG_NAME_CHARS = 64;
        public const int RESERVED1 = 4;
        public const int RESERVED2 = 5;

        public int cbStruct;                        //  size in Byte of MIXERCONTROL
        public int dwControlID;                     //  unique control id for mixer device
        public ControlType dwControlType;           //  MIXERCONTROL_CONTROLTYPE_xxx
        public MIXERCONTROL_CONTROLF fdwControl;    //  MIXERCONTROL_CONTROLF_xxx
        public int cMultipleItems;                  //  if MIXERCONTROL_CONTROLF_MULTIPLE set
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = MIXER_SHORT_NAME_CHARS)]
        public string szShortName;                  //  short name of control
        [MarshalAs(UnmanagedType.ByValTStr, SizeConst = MIXER_LONG_NAME_CHARS)]
        public string szName;                       //  int name of control
        public int lMinimum;                        //  Minimum value
        public int lMaximum;                        //  Maximum value
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = RESERVED1)]
        public int[] reservedBytes1;                //  reserved structure space
        public int cSteps;                          // # of steps between min & max
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = RESERVED2)]
        public int[] reservedBytes2;                //  reserved structure space
    }

    #endregion

    public static class MMIO
    {
        public delegate int MMIOProc(
          [In] string lpmmioinfo,
          int uMsg,
          IntPtr lParam1,
          IntPtr lParam2
        );

        public static string Errorstring(MMIOError i)
        {
            string sRet;

            switch (i)
            {
                case MMIOError.NoError:
                    sRet = "No error";
                    break;
                case MMIOError.FileNotFound:
                    sRet = "File not found";
                    break;
                case MMIOError.OutOfMemory:
                    sRet = "Out of memory";
                    break;
                case MMIOError.CannotOpen:
                    sRet = "Cannot open";
                    break;
                case MMIOError.CannotClose:
                    sRet = "Cannot close";
                    break;
                case MMIOError.CannotRead:
                    sRet = "Cannot read";
                    break;
                case MMIOError.CannotWrite:
                    sRet = "Cannot write";
                    break;
                case MMIOError.CannotSeek:
                    sRet = "Cannot seek";
                    break;
                case MMIOError.CannotExpand:
                    sRet = "Cannot expand file";
                    break;
                case MMIOError.ChunkNotFound:
                    sRet = "Chunk not found";
                    break;
                case MMIOError.Unbuffered:
                    sRet = "Unbuffered";
                    break;
                case MMIOError.PathNotFound:
                    sRet = "Path incorrect";
                    break;
                case MMIOError.AccessDenied:
                    sRet = "File was protected (Access denied)";
                    break;
                case MMIOError.SharingViolation:
                    sRet = "file in use (Sharing violation)";
                    break;
                case MMIOError.NetworkError:
                    sRet = "Network not responding";
                    break;
                case MMIOError.TooManyOpenFiles:
                    sRet = "No more file handles";
                    break;
                case MMIOError.InvalidFile:
                    sRet = "Invalid File";
                    break;
                default:
                    sRet = "Unknown error number: " + i.ToString();
                    break;
            }
            return sRet;
        }

        public static void ThrowExceptionForError(MMIOError i)
        {
            if (i != MMIOError.NoError)
            {
                throw new Exception(Errorstring(i));
            }
        }

        #region Externs

#if false
        // These methods are unnecessary or deprecated

        [DllImport("winmm.dll",
        CharSet = CharSet.Unicode,
        ExactSpelling=true,
        EntryPoint="mmioStringToFOURCCW"),
        SuppressUnmanagedCodeSecurity]
        public static extern FourCC mmioStringToFOURCC(
            [In] string sz,
            int uFlags);

        //LPMMIOPROC WINAPI mmioInstallIOProc( FOURCC fccIOProc, LPMMIOPROC pIOProc, DWORD dwFlags);
#endif

        [DllImport("winmm.dll", ExactSpelling = true, CharSet = CharSet.Unicode, EntryPoint = "mmioRenameW"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMIOError Rename(
            [In] string pszFileName,
            [In] string pszNewFileName,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] MMIOINFO pmmioinfo,
            int fdwRename);

        [DllImport("winmm.dll", EntryPoint = "mmioGetInfo"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMIOError GetInfo(
            IntPtr hmmio,
            [Out, MarshalAs(UnmanagedType.LPStruct)] MMIOINFO pmmioinfo,
            int fuInfo);

        [DllImport("winmm.dll", EntryPoint = "mmioSetInfo"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMIOError SetInfo(
            IntPtr hmmio,
            [In, MarshalAs(UnmanagedType.LPStruct)] MMIOINFO pmmioinfo,
            int fuInfo);

        [DllImport("winmm.dll", EntryPoint = "mmioSetBuffer"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMIOError SetBuffer(
            IntPtr hmmio,
            IntPtr pchBuffer,
            int cchBuffer,
            int fuBuffer);

        [DllImport("winmm.dll", EntryPoint = "mmioAdvance"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMIOError Advance(
            IntPtr hmmio,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] MMIOINFO pmmioinfo,
            RWMode fuAdvance);

        [DllImport("winmm.dll", EntryPoint = "mmioSendMessage"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMIOError SendMessage(
            IntPtr hmmio,
            int uMsg,
            IntPtr lParam1,
            IntPtr lParam2);

        [DllImport("winmm.dll", EntryPoint = "mmioClose"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMIOError Close(
            IntPtr hmmio,
            MMIOCloseFlags uFlags);

        [DllImport("winmm.dll", EntryPoint = "mmioFlush"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMIOError Flush(
            IntPtr hmmio,
            MMIOFlushFlags uFlags);

        [DllImport("winmm.dll", EntryPoint = "mmioDescend"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMIOError Descend(
            IntPtr hmmio,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] MMCKINFO lpck,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] MMCKINFO lpckParent,
            RiffChunkFlags uFlags);

        [DllImport("winmm.dll", EntryPoint = "mmioCreateChunk"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMIOError CreateChunk(
            IntPtr hmmio,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] MMCKINFO lpck,
            RiffChunkFlags uFlags);

        [DllImport("winmm.dll", ExactSpelling = true, CharSet = CharSet.Unicode, EntryPoint = "mmioOpenW"),
        SuppressUnmanagedCodeSecurity]
        public static extern IntPtr Open(
            [In] string szFileName,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] MMIOINFO lpmmioinfo,
            MMIOFlags dwOpenFlags);

        [DllImport("winmm.dll", EntryPoint = "mmioRead"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMIOError Read(
            IntPtr hmmio,
            IntPtr pch,
            int cch);

        [DllImport("winmm.dll", EntryPoint = "mmioWrite"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Write(
            IntPtr hmmio,
            IntPtr h,
            int cch);

        [DllImport("winmm.dll", EntryPoint = "mmioSeek"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Seek(
            IntPtr hmmio,
            int lOffset,
            MMIOSeekFlags iOrigin);

        [DllImport("winmm.dll", EntryPoint = "mmioAscend"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMIOError Ascend(
            IntPtr hmmio,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] MMCKINFO lpck,
            int uFlags); // must be zero

        #endregion

    }

    public static class waveIn
    {
        private const short MAXERRORLENGTH = 256;

        public delegate void WaveInDelegate(
            IntPtr hwo,
            MIWM uMsg,
            IntPtr dwInstance,
            IntPtr dwParam1,
            IntPtr dwParam2);

        public static void ThrowExceptionForError(int rc)
        {
            if (rc != 0)
            {
                StringBuilder foo = new StringBuilder(MAXERRORLENGTH);
                GetErrorText(rc, foo, MAXERRORLENGTH);

                throw new Exception(foo.ToString());
            }
        }

        #region Externs

        // These methods are unnecessary or deprecated

        [DllImport("winmm.dll", EntryPoint="waveInGetID"),
        SuppressUnmanagedCodeSecurity]
        public static extern int GetID(
            IntPtr hwi,
            out int puDeviceID);

        [DllImport("winmm.dll", EntryPoint = "waveInGetPosition"),
        SuppressUnmanagedCodeSecurity]
        public static extern int GetPosition(
            IntPtr hwi,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] MMTIME pmmt,
            int cbmmt);

        [DllImport("winmm.dll", EntryPoint = "waveInMessage"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Message(
            IntPtr hwi,
            int uMsg,
            IntPtr dw1,
            IntPtr dw2);

        [DllImport("winmm.dll", ExactSpelling = true, CharSet = CharSet.Unicode, EntryPoint = "waveInGetDevCapsW"),
        SuppressUnmanagedCodeSecurity]
        public static extern int GetDevCaps(
            int uDeviceID,
            [Out, MarshalAs(UnmanagedType.LPStruct)] WaveInCaps pwic,
            int cbwic);

        [DllImport("winmm.dll", EntryPoint = "waveInGetNumDevs"),
        SuppressUnmanagedCodeSecurity]
        public static extern int GetNumDevs();

        [DllImport("winmm.dll", EntryPoint = "waveInStop"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Stop(
            IntPtr hwi);

        [DllImport("winmm.dll", ExactSpelling = true, CharSet = CharSet.Unicode, EntryPoint = "waveInGetErrorTextW"),
        SuppressUnmanagedCodeSecurity]
        public static extern int GetErrorText(
            int errvalue,
            [Out] StringBuilder lpText,
            int uSize);

        [DllImport("winmm.dll", EntryPoint = "waveInOpen"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Open(
            out IntPtr hwi,
            int uDeviceID,
            [In, MarshalAs(UnmanagedType.LPStruct)] WaveFormatEx b,
            WaveInDelegate dwCallback,
            IntPtr dwCallbackInstance,
            WaveOpenFlags dwFlags);

        [DllImport("winmm.dll", EntryPoint = "waveInPrepareHeader"),
        SuppressUnmanagedCodeSecurity]
        public static extern int PrepareHeader(
            IntPtr hwi,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] WAVEHDR lpWaveInHdr,
            int uSize);

        [DllImport("winmm.dll", EntryPoint = "waveInUnprepareHeader"),
        SuppressUnmanagedCodeSecurity]
        public static extern int UnprepareHeader(
            IntPtr hwi,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] WAVEHDR lpWaveInHdr,
            int uSize);

        [DllImport("winmm.dll", EntryPoint = "waveInStart"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Start(
            IntPtr hwi);

        [DllImport("winmm.dll", EntryPoint = "waveInAddBuffer"),
        SuppressUnmanagedCodeSecurity]
        public static extern int AddBuffer(
            IntPtr hwi,
            IntPtr lpWaveInHdr,
            int uSize);

        [DllImport("winmm.dll", EntryPoint = "waveInClose"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Close(
            IntPtr hwi);

        [DllImport("winmm.dll", EntryPoint = "waveInReset"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Reset(
            IntPtr hwi);

        #endregion

    }

    public static class waveOut
    {
        private const short MAXERRORLENGTH = 256;

        public delegate void WaveOutDelegate(
            IntPtr hwo,
            MOWM uMsg,
            IntPtr dwInstance,
            IntPtr dwParam1,
            IntPtr dwParam2);

        public static void ThrowExceptionForError(int rc)
        {
            if (rc != 0)
            {
                StringBuilder foo = new StringBuilder(MAXERRORLENGTH);
                GetErrorText(rc, foo, MAXERRORLENGTH);

                throw new Exception(foo.ToString());
            }
        }

        #region Externs

        [DllImport("winmm.dll", EntryPoint = "waveOutGetID"),
        SuppressUnmanagedCodeSecurity]
        public static extern int GetID(
            IntPtr hwo,
            out int puDeviceID);

        [DllImport("winmm.dll", EntryPoint = "waveOutGetNumDevs"),
        SuppressUnmanagedCodeSecurity]
        public static extern int GetNumDevs();

        [DllImport("winmm.dll", ExactSpelling = true, CharSet = CharSet.Unicode, EntryPoint = "waveOutGetDevCapsW"),
        SuppressUnmanagedCodeSecurity]
        public static extern int GetDevCaps(
            int uDeviceID,
            [Out, MarshalAs(UnmanagedType.LPStruct)] WaveOutCaps pwoc,
            int cbwoc);

        [DllImport("winmm.dll", EntryPoint = "waveOutGetPitch"),
        SuppressUnmanagedCodeSecurity]
        public static extern int GetPitch(
            IntPtr hwo,
            out int pdwPitch);

        [DllImport("winmm.dll", EntryPoint = "waveOutSetPitch"),
        SuppressUnmanagedCodeSecurity]
        public static extern int SetPitch(
            IntPtr hwo,
            int dwPitch);

        [DllImport("winmm.dll", EntryPoint = "waveOutGetPlaybackRate"),
        SuppressUnmanagedCodeSecurity]
        public static extern int GetPlaybackRate(
            IntPtr hwo,
            out int pdwRate);

        [DllImport("winmm.dll", EntryPoint = "waveOutSetPlaybackRate"),
        SuppressUnmanagedCodeSecurity]
        public static extern int SetPlaybackRate(
            IntPtr hwo,
            int dwRate);

        [DllImport("winmm.dll", EntryPoint = "waveOutMessage"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Message(
            IntPtr hwo,
            int uMsg,
            IntPtr dw1,
            IntPtr dw2);

        [DllImport("winmm.dll", EntryPoint = "waveOutBreakLoop"),
        SuppressUnmanagedCodeSecurity]
        public static extern int BreakLoop(
            IntPtr hwo);

        [DllImport("winmm.dll", EntryPoint = "waveOutGetPosition"),
        SuppressUnmanagedCodeSecurity]
        public static extern int GetPosition(
            IntPtr hWaveOut,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] MMTIME lpInfo,
            int uSize);

        [DllImport("winmm.dll", ExactSpelling = true, CharSet = CharSet.Unicode, EntryPoint = "waveOutGetErrorTextW"),
        SuppressUnmanagedCodeSecurity]
        public static extern int GetErrorText(
            int errvalue,
            [Out] StringBuilder lpText,
            int uSize);

        [DllImport("winmm.dll", EntryPoint = "waveOutOpen"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Open(
            out IntPtr hWaveOut,
            int uDeviceID,
            [In, MarshalAs(UnmanagedType.LPStruct)] WaveFormatEx b,
            WaveOutDelegate dwCallback, // If using Function callback
            IntPtr dwCallbackInstance,
            WaveOpenFlags dwFlags);

        [DllImport("winmm.dll", EntryPoint = "waveOutOpen"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Open(
            out IntPtr hWaveOut,
            int uDeviceID,
            [In, MarshalAs(UnmanagedType.LPStruct)] WaveFormatEx b,
            IntPtr dwCallback, // If using Event
            IntPtr dwCallbackInstance,
            WaveOpenFlags dwFlags);

        [DllImport("winmm.dll", EntryPoint = "waveOutPrepareHeader"),
        SuppressUnmanagedCodeSecurity]
        public static extern int PrepareHeader(
            IntPtr hWaveOut,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] WAVEHDR lpWaveOutHdr,
            int uSize);

        [DllImport("winmm.dll", EntryPoint = "waveOutReset"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Reset(
            IntPtr hWaveOut);

        [DllImport("winmm.dll", EntryPoint = "waveOutUnprepareHeader"),
        SuppressUnmanagedCodeSecurity]
        public static extern int UnprepareHeader(
            IntPtr hWaveOut,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] WAVEHDR lpWaveOutHdr,
            int uSize);

        [DllImport("winmm.dll", EntryPoint = "waveOutClose"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Close(
            IntPtr hWaveOut);

        [DllImport("winmm.dll", EntryPoint = "waveOutWrite"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Write(
            IntPtr hWaveOut,
            IntPtr lpWaveOutHdr,
            int uSize);

        [DllImport("winmm.dll", EntryPoint = "waveOutPause"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Pause(
            IntPtr hWaveOut);

        [DllImport("winmm.dll", EntryPoint = "waveOutRestart"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Restart(
            IntPtr hWaveOut);

        [DllImport("winmm.dll", EntryPoint = "waveOutSetVolume"),
        SuppressUnmanagedCodeSecurity]
        public static extern int SetVolume(
            IntPtr uDeviceID,
            int dwVolume);

        [DllImport("winmm.dll", EntryPoint = "waveOutGetVolume"),
        SuppressUnmanagedCodeSecurity]
        public static extern int GetVolume(
            IntPtr uDeviceID,
            out int lpdwVolume);

        #endregion

    }

    public static class Mixer
    {
        public const int LINE_CHANGE = 0x3D0;           /* mixer line change notify */
        public const int CONTROL_CHANGE = 0x3D1;        /* mixer control change notify */

        #region Externals

        [DllImport("winmm.dll", EntryPoint = "mixerSetControlDetails"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMSYSERR SetControlDetails(
            IntPtr hmxobj,
            [In, MarshalAs(UnmanagedType.LPStruct)] MixerControlDetails pmxcd,
            MIXER_SETCONTROLDETAILSF fdwDetails);

        [DllImport("winmm.dll", CharSet = CharSet.Unicode, ExactSpelling = true, EntryPoint = "mixerGetControlDetailsW"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMSYSERR GetControlDetails(
            IntPtr hmxobj,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] MixerControlDetails pmxcd,
            MIXER_OBJECTF fdwDetails);

        [DllImport("winmm.dll", EntryPoint = "mixerOpen"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMSYSERR Open(
            out IntPtr phmx,
            int uMxId,
            IntPtr dwCallback,
            IntPtr dwInstance,
            MIXER_OBJECTF fdwOpen);

        [DllImport("winmm.dll", EntryPoint = "mixerClose"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMSYSERR Close(
            IntPtr phmx);

        [DllImport("winmm.dll", CharSet = CharSet.Unicode, ExactSpelling = true, EntryPoint = "mixerGetLineInfoW"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMSYSERR GetLineInfo(
            IntPtr hmxobj,
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] MixerLine pmxl,
            MIXER_GETLINEINFOF fdwInfo);

        [DllImport("winmm.dll", CharSet = CharSet.Unicode, ExactSpelling = true, EntryPoint = "mixerGetLineControlsW"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMSYSERR GetLineControls(
            IntPtr hmxobj,
            [In, Out, MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(MLCMarshaler))] MixerLineControls pmxlc,
            MIXER_GETLINECONTROLSF fdwControls);

        [DllImport("winmm.dll", CharSet = CharSet.Unicode, ExactSpelling = true, EntryPoint = "mixerGetDevCapsW"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMSYSERR GetDevCaps(
            IntPtr uMxId,
            [Out, MarshalAs(UnmanagedType.LPStruct)] MixerCaps pmxcaps,
            int cbmxcaps);

        [DllImport("winmm.dll", EntryPoint = "mixerGetNumDevs"),
        SuppressUnmanagedCodeSecurity]
        public static extern int GetNumDevs();

        [DllImport("winmm.dll", EntryPoint = "mixerGetID"),
        SuppressUnmanagedCodeSecurity]
        public static extern MMSYSERR GetID(
            IntPtr hmxobj,
            out int puMxId,
            MIXER_OBJECTF fdwId);

        [DllImport("winmm.dll", EntryPoint = "mixerMessage"),
        SuppressUnmanagedCodeSecurity]
        public static extern int Message(
            IntPtr driverID,
            int uMsg,
            IntPtr dwParam1,
            IntPtr dwParam2);

        #endregion

        #region Public functions

        public static string Errorstring(MMSYSERR iError)
        {
            string sRet;

            switch (iError)
            {
                case MMSYSERR.NoError:
                    sRet = "The specified command was carried out.";
                    break;
                case MMSYSERR.Error:
                    sRet = "Undefined external error.";
                    break;
                case MMSYSERR.BadDeviceID:
                    sRet = "A device ID has been used that is out of range for your system.";
                    break;
                case MMSYSERR.NotEnabled:
                    sRet = "The driver was not enabled.";
                    break;
                case MMSYSERR.Allocated:
                    sRet = "The specified device is already in use.  Wait until it is free, and then try again.";
                    break;
                case MMSYSERR.InvalHandle:
                    sRet = "The specified device handle is invalid.";
                    break;
                case MMSYSERR.NoDriver:
                    sRet = "There is no driver installed on your system.";
                    break;
                case MMSYSERR.NoMem:
                    sRet = "There is not enough memory available for this task.  Quit one or more applications to increase avai";
                    break;
                case MMSYSERR.NotSupported:
                    sRet = "This function is not supported.  Use the Capabilities function to determine which functions and mes";
                    break;
                case MMSYSERR.BadErrNum:
                    sRet = "An error number was specified that is not defined in the system.";
                    break;
                case MMSYSERR.InvalFlag:
                    sRet = "An invalid flag was passed to a system function.";
                    break;
                case MMSYSERR.InvalParam:
                    sRet = "An invalid parameter was passed to a system function.";
                    break;
                case MMSYSERR.HandleBusy:
                    sRet = "Handle being used simultaneously on another thread (eg callback).";
                    break;
                case MMSYSERR.InvalidAlias:
                    sRet = "Specified alias not found in WIN.INI.";
                    break;
                case MMSYSERR.BadDB:
                    sRet = "The registry database is corrupt.";
                    break;
                case MMSYSERR.KeyNotFound:
                    sRet = "The specified registry key was not found.";
                    break;
                case MMSYSERR.ReadError:
                    sRet = "The registry could not be opened or could not be read.";
                    break;
                case MMSYSERR.WriteError:
                    sRet = "The registry could not be written to.";
                    break;
                case MMSYSERR.DeleteError:
                    sRet = "The specified registry key could not be deleted.";
                    break;
                case MMSYSERR.ValNotFound:
                    sRet = "The specified registry key value could not be found.";
                    break;
                case MMSYSERR.NoDriverCB:
                    sRet = "The driver did not generate a valid OPEN callback.";
                    break;
                case MMSYSERR.MoreData:
                    sRet = "More data to be returned";
                    break;

                case MMSYSERR.InvalLine:
                    sRet = "The line reference is invalid.";
                    break;
                case MMSYSERR.InvalControl:
                    sRet = "The control reference is invalid.";
                    break;
                case MMSYSERR.InvalValue:
                    sRet = "The value is invalid.";
                    break;
                default:
                    sRet = "Unknown error code";
                    break;
            }

            return sRet;
        }

        public static void ThrowExceptionForError(MMSYSERR i)
        {
            if (i != MMSYSERR.NoError)
            {
                throw new Exception(Errorstring(i));
            }
        }

        #endregion

    }

    #region Internal code

    // Custom marshaler for Mixer.GetLineControls
    internal class MLCMarshaler : ICustomMarshaler
    {
        // The managed object passed in to MarshalManagedToNative
        protected MixerLineControls m_Control;

        protected int iMixContSize = Marshal.SizeOf(typeof(MixerControl));
        protected int iMixLineContSize = Marshal.SizeOf(typeof(MixerLineControls));

        public IntPtr MarshalManagedToNative(object managedObj)
        {
            IntPtr p;

            // Cast the object back to a PropVariant
            m_Control = managedObj as MixerLineControls;

            if (m_Control != null)
            {
                // Create an appropriately sized buffer, blank it, and send it to
                // the marshaler to make the COM call with.
                int iSize2 = m_Control.cControls * iMixContSize;

                p = Marshal.AllocCoTaskMem(iMixLineContSize + iSize2);
#if DEBUG
                for (int x = 0; x < iMixLineContSize + iSize2; x++)
                {
                    Marshal.WriteByte(p, x, 0xcc);
                }
#endif
                Marshal.StructureToPtr(m_Control, p, false);
                Marshal.WriteIntPtr(p, iMixLineContSize - IntPtr.Size, new IntPtr(p.ToInt64() + iMixLineContSize));
            }
            else
            {
                p = IntPtr.Zero;
            }

            return p;
        }

        // Called just after invoking the COM method.  The IntPtr is the same one that just got returned
        // from MarshalManagedToNative.  The return value is unused.
        public object MarshalNativeToManaged(IntPtr pNativeData)
        {
            m_Control.cbStruct = Marshal.ReadInt32(pNativeData);
            m_Control.dwLineID = Marshal.ReadInt32(pNativeData, 4);
            m_Control.dwControl = Marshal.ReadInt32(pNativeData, 8);
            m_Control.cControls = Marshal.ReadInt32(pNativeData, 12);
            m_Control.cbmxctrl = Marshal.ReadInt32(pNativeData, 16);
            m_Control.pamxctrl = new MixerControl[m_Control.cControls];

            IntPtr pData = new IntPtr(pNativeData.ToInt64() + iMixLineContSize);

            for (int x = 0; x < m_Control.cControls; x++)
            {
                m_Control.pamxctrl[x] = new MixerControl();
                IntPtr ip = new IntPtr(pData.ToInt64() + (x * iMixContSize));
                Marshal.PtrToStructure(ip, m_Control.pamxctrl[x]);
            }

            m_Control = null;

            return m_Control;
        }

        // It appears this routine is never called
        public void CleanUpManagedData(object ManagedObj)
        {
            m_Control = null;
        }

        public void CleanUpNativeData(IntPtr pNativeData)
        {
            Marshal.FreeCoTaskMem(pNativeData);
        }

        // The number of bytes to marshal out
        public int GetNativeDataSize()
        {
            return 0;
        }

        // This method is called by interop to create the custom marshaler.  The (optional)
        // cookie is the value specified in MarshalCookie="asdf", or "" is none is specified.
        public static ICustomMarshaler GetInstance(string cookie)
        {
            return new MLCMarshaler();
        }
    }

    #endregion
}
