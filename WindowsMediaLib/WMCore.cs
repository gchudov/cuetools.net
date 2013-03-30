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
using System.Runtime.InteropServices.ComTypes;
using System.Security;

using WindowsMediaLib.Defs;

namespace WindowsMediaLib
{
    /// <summary>
    /// CLSID_ClientNetManager
    /// </summary>
    [ComImport, Guid("CD12A3CE-9C42-11D2-BEED-0060082F2054")]
    public class ClientNetManager
    {
    }

    #region Declarations

#if ALLOW_UNTESTED_INTERFACES

    [UnmanagedName("WMT_WATERMARK_ENTRY_TYPE")]
    public enum WaterMarkEntryType
    {
        // Fields
        Audio = 1,
        Video = 2
    }

    [StructLayout(LayoutKind.Sequential), UnmanagedName("WMDRM_IMPORT_INIT_STRUCT")]
    public class ImportInitStruct
    {
        public int dwVersion;
        public int cbEncryptedSessionKeyMessage;
        public IntPtr pbEncryptedSessionKeyMessage;
        public int cbEncryptedKeyMessage;
        public IntPtr pbEncryptedKeyMessage;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4), UnmanagedName("DRM_MINIMUM_OUTPUT_PROTECTION_LEVELS")]
    public struct MinimumOutputProtectionLevels
    {
        short wCompressedDigitalVideo;
        short wUncompressedDigitalVideo;
        short wAnalogVideo;
        short wCompressedDigitalAudio;
        short wUncompressedDigitalAudio;
    }

    [StructLayout(LayoutKind.Sequential), UnmanagedName("DRM_OPL_OUTPUT_IDS")]
    public struct OPLOutputIds
    {
        short cIds;
        Guid [] rgIds;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4), UnmanagedName("DRM_OUTPUT_PROTECTION")]
    public struct VideoOutputProtection
    {
        Guid guidId;
        byte bConfigData;
    }

    [StructLayout(LayoutKind.Sequential), UnmanagedName("DRM_COPY_OPL")]
    public struct CopyOPL
    {
        short wMinimumCopyLevel;
        OPLOutputIds oplIdIncludes;
        OPLOutputIds oplIdExcludes;
    }

    [StructLayout(LayoutKind.Sequential), UnmanagedName("DRM_VIDEO_OUTPUT_PROTECTION_IDS")]
    public struct VideoOutputProtectionIDs
    {
        short cEntries;
        VideoOutputProtection [] rgVop;
    }

    [StructLayout(LayoutKind.Sequential), UnmanagedName("DRM_PLAY_OPL")]
    public struct PlayOpl
    {
        MinimumOutputProtectionLevels minOPL;
        OPLOutputIds oplIdReserved;
        VideoOutputProtectionIDs vopi;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 1), UnmanagedName("DRM_VAL16")]
    public struct Val16
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst=16)]  byte [] val;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4), UnmanagedName("WMT_WATERMARK_ENTRY")]
    public struct WaterMarkEntry
    {
        public WaterMarkEntryType wmetType;
        public Guid clsid;
        public int cbDisplayName;
        [MarshalAs(UnmanagedType.LPWStr)]
        public string pwszDisplayName;
    }

#endif

    [Flags]
    public enum LicenseStateDataFlags
    {
        None = 0,
        Vague = 1,
        OPLPresent = 2,
        SAPPresent = 4
    }

    [UnmanagedName("DRM_LICENSE_STATE_CATEGORY")]
    public enum LicenseStateCategory
    {
        NoRight = 0,
        UnLimited,
        Count,
        From,
        Until,
        FromUntil,
        CountFrom,
        CountUntil,
        CountFromUntil,
        ExpirationAfterFristUse
    }

    [UnmanagedName("WMT_INDEX_TYPE")]
    public enum IndexType
    {
        None = 0,
        NearestDataUnit = 1,
        NearestObject = 2,
        NearestCleanPoint = 3
    }

    [UnmanagedName("WM_AETYPE")]
    public enum AEType
    {
        Exclude = 0x65,
        Include = 0x69
    }

    [UnmanagedName("WMT_ATTR_DATATYPE")]
    public enum AttrDataType
    {
        DWORD = 0,
        STRING = 1,
        BINARY = 2,
        BOOL = 3,
        QWORD = 4,
        WORD = 5,
        GUID = 6
    }

    [Flags, UnmanagedName("From defines")]
    public enum BackupRestoreFlags
    {
        None = 0,
        OverWrite = 0x00000001,
        Individualize = 0x00000002
    }

    [UnmanagedName("WMT_CODEC_INFO_TYPE")]
    public enum CodecInfoType
    {
        Audio = 0,
        Video = 1,
        Unknown = 0xffffff
    }

    [Flags]
    public enum CredentialFlag
    {
        None = 0,
        AutoSavePW = 1
    }

    [Flags, UnmanagedName("WMT_CREDENTIAL_FLAGS")]
    public enum CredentialFlags
    {
        Save = 0x1,
        DontCache = 0x2,
        ClearText = 0x4,
        Proxy = 0x8,
        Encrypt = 0x10
    }

    [Flags, UnmanagedName("WMT_FILESINK_MODE")]
    public enum FileSinkMode
    {
        None = 0,
        SingleBuffers = 0x1,
        FileSinkDataUnits = 0x2,
        FileSinkUnbuffered = 0x4
    }

    [UnmanagedName("WMT_INDEXER_TYPE")]
    public enum IndexerType
    {
        // Fields
        FrameNumbers = 1,
        PresentationTime = 0,
        TimeCode = 2
    }

    [Flags]
    public enum MetaDataAccess
    {
        None = 0,
        Write = 0x40000000,
        Read = unchecked((int)0x80000000)
    }

    [Flags]
    public enum MetaDataShare
    {
        None = 0,
        Read = 0x00000001,
        Delete = 0x00000004
    }

    [UnmanagedName("WMT_NET_PROTOCOL")]
    public enum NetProtocol
    {
        HTTP = 0
    }

    [UnmanagedName("NETSOURCE_URLCREDPOLICY_SETTINGS")]
    public enum NetSourceURLCredPolicySettings
    {
        AnonymousOnly = 2,
        MustPromptUser = 1,
        SilentLogonOk = 0
    }

    [UnmanagedName("WMT_OFFSET_FORMAT")]
    public enum OffsetFormat
    {
        // Fields
        HundredNS = 0,
        FrameNumbers = 1,
        PlaylistOffset = 2,
        Timecode = 3
    }

    [UnmanagedName("WMT_PLAY_MODE")]
    public enum PlayMode
    {
        // Fields
        AutoSelect = 0,
        Download = 2,
        Local = 1,
        Streaming = 3
    }

    [UnmanagedName("WMT_PROXY_SETTINGS")]
    public enum ProxySettings
    {
        // Fields
        Auto = 2,
        Browser = 3,
        Manual = 1,
        Max = 4,
        None = 0
    }

    [Flags, UnmanagedName("WMT_RIGHTS")]
    public enum Rights
    {
        None = 0,
        Playback = 0x00000001,
        CopyToNonSDMIDevice = 0x00000002,
        CopyToCD = 0x00000008,
        CopyToSDMIDevice = 0x00000010,
        OneTime = 0x00000020,
        SaveStreamProtected = 0x00000040,
        Copy = 0x00000080,
        CollaborativePlay = 0x00000100,
        SDMITrigger = 0x00010000,
        SDMINoMoreCopies = 0x00020000
    }

    [Flags, UnmanagedName("From unnamed enum")]
    public enum SampleFlag
    {
        None = 0,
        CleanPoint = 0x1,
        Discontinuity = 0x2,
        Dataloss = 0x4
    }

    [Flags]
    public enum SampleFlagEx
    {
        None = 0,
        NotASyncPoint = 0x2,
        Dataloss = 0x4
    }

    [UnmanagedName("WMT_STATUS")]
    public enum Status
    {
        Error = 0,
        Opened = 1,
        BufferingStart = 2,
        BufferingStop = 3,
        EOF = 4,
        EndOfFile = 4,
        EndOfSegment = 5,
        EndOfStreaming = 6,
        Locating = 7,
        Connecting = 8,
        NoRights = 9,
        MissingCodec = 10,
        Started = 11,
        Stopped = 12,
        Closed = 13,
        Striding = 14,
        Timer = 15,
        IndexProgress = 16,
        SaveasStart = 17,
        SaveasStop = 18,
        NewSourceflags = 19,
        NewMetadata = 20,
        BackuprestoreBegin = 21,
        SourceSwitch = 22,
        AcquireLicense = 23,
        Individualize = 24,
        NeedsIndividualization = 25,
        NoRightsEx = 26,
        BackuprestoreEnd = 27,
        BackuprestoreConnecting = 28,
        BackuprestoreDisconnecting = 29,
        ErrorWithurl = 30,
        RestrictedLicense = 31,
        ClientConnect = 32,
        ClientDisconnect = 33,
        NativeOutputPropsChanged = 34,
        ReconnectStart = 35,
        ReconnectEnd = 36,
        ClientConnectEx = 37,
        ClientDisconnectEx = 38,
        SetFECSpan = 39,
        PrerollReady = 40,
        PrerollComplete = 41,
        ClientProperties = 42,
        LicenseURLSignatureState = 43,
        InitPlaylistBurn = 44,
        TranscryptorInit = 45,
        TranscryptorSeeked = 46,
        TranscryptorRead = 47,
        TranscryptorClosed = 48,
        ProximityResult = 49,
        ProximityCompleted = 50,
        ContentEnabler = 51
    }

    [UnmanagedName("WMT_STORAGE_FORMAT")]
    public enum StorageFormat
    {
        // Fields
        MP3 = 0,
        V1 = 1
    }

    [UnmanagedName("WMT_STREAM_SELECTION")]
    public enum StreamSelection
    {
        CleanPointOnly = 1,
        Off = 0,
        On = 2
    }

    [UnmanagedName("WMT_TRANSPORT_TYPE")]
    public enum TransportType
    {
        // Fields
        Reliable = 1,
        Unreliable = 0
    }

    [UnmanagedName("WMT_VERSION")]
    public enum WMVersion
    {
        V4_0 = 0x00040000,
        V7_0 = 0x00070000,
        V8_0 = 0x00080000,
        V9_0 = 0x00090000
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4), UnmanagedName("WMT_BUFFER_SEGMENT")]
    public struct BufferSegment
    {
        public INSSBuffer pBuffer;
        public int cbOffset;
        public int cbLength;
    }

    [StructLayout(LayoutKind.Sequential), UnmanagedName("WMT_FILESINK_DATA_UNIT")]
    public class FileSinkDataUnit
    {
        public BufferSegment packetHeaderBuffer;
        public int cPayloads;
        public IntPtr pPayloadHeaderBuffers;
        public int cPayloadDataFragments;
        public IntPtr pPayloadDataFragments;
    }

    [StructLayout(LayoutKind.Explicit, Pack = 1)]
    public class RA3Union
    {
        [FieldOffset(0)]
        public long lOffsetStart;
        [FieldOffset(0)]
        public int iOffsetStart;

        [FieldOffset(0)]
        public short wRange;
        [FieldOffset(2)]
        public int Timecode;
        [FieldOffset(6)]
        public int dwUserbits;
        [FieldOffset(10)]
        public int dwAmFlags;

        public RA3Union()
        {
        }

        public RA3Union(long l)
        {
            lOffsetStart = l;
        }

        public RA3Union(int i)
        {
            iOffsetStart = i;
        }

        public RA3Union(TimeCodeExtensionData w)
        {
            wRange = w.wRange;
            Timecode = w.dwTimecode;
            dwUserbits = w.dwUserbits;
            dwAmFlags = w.dwAmFlags;
        }

    }

    [StructLayout(LayoutKind.Sequential, Pack = 2), UnmanagedName("WMT_TIMECODE_EXTENSION_DATA")]
    public class TimeCodeExtensionData
    {
        public short wRange;
        public int dwTimecode;
        public int dwUserbits;
        public int dwAmFlags;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4), UnmanagedName("WM_ADDRESS_ACCESSENTRY")]
    public struct WMAddressAccessEntry
    {
        public int dwIPAddress;
        public int dwMask;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4), UnmanagedName("WM_CLIENT_PROPERTIES")]
    public struct WMClientProperties
    {
        public int dwIPAddress;
        public int dwPort;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 2), UnmanagedName("WM_PORT_NUMBER_RANGE")]
    public struct WMPortNumberRange
    {
        public short wPortBegin;
        public short wPortEnd;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 8), UnmanagedName("WM_READER_CLIENTINFO")]
    public class WMReaderClientInfo
    {
        public int cbSize;
        public string wszLang;
        public string wszBrowserUserAgent;
        public string wszBrowserWebPage;
        public long qwReserved;
        public IntPtr pReserved;
        public string wszHostExe;
        public long qwHostVersion;
        public string wszPlayerUserAgent;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4), UnmanagedName("WM_READER_STATISTICS")]
    public class WMReaderStatistics
    {
        public int cbSize;
        public int dwBandwidth;
        public int cPacketsReceived;
        public int cPacketsRecovered;
        public int cPacketsLost;
        public short wQuality;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 2), UnmanagedName("WM_STREAM_PRIORITY_RECORD")]
    public struct WMStreamPrioritizationRecord
    {
        public short wStreamNumber;
        public bool fMandatory;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4), UnmanagedName("WM_WRITER_STATISTICS_EX")]
    public struct WMWriterStatisticsEx
    {
        public int dwBitratePlusOverhead;
        public int dwCurrentSampleDropRateInQueue;
        public int dwCurrentSampleDropRateInCodec;
        public int dwCurrentSampleDropRateInMultiplexer;
        public int dwTotalSampleDropsInQueue;
        public int dwTotalSampleDropsInCodec;
        public int dwTotalSampleDropsInMultiplexer;
    }

    [StructLayout(LayoutKind.Sequential), UnmanagedName("WM_WRITER_STATISTICS")]
    public struct WriterStatistics
    {
        public long qwSampleCount;
        public long qwByteCount;
        public long qwDroppedSampleCount;
        public long qwDroppedByteCount;
        public int dwCurrentBitrate;
        public int dwAverageBitrate;
        public int dwExpectedBitrate;
        public int dwCurrentSampleRate;
        public int dwAverageSampleRate;
        public int dwExpectedSampleRate;
    }

    [StructLayout(LayoutKind.Sequential), UnmanagedName("WM_LICENSE_STATE_DATA")]
    public struct LicenseStateData
    {
        public LicenseStateData(byte[] b)
        {
            int LICENSESTATEDATASIZE = Marshal.SizeOf(typeof(DRMLicenseStateData));
            dwSize = BitConverter.ToInt32(b, 0);
            dwNumStates = BitConverter.ToInt32(b, 4);

            stateData = new DRMLicenseStateData[dwNumStates];

            for (int x = 0; x < dwNumStates; x++)
            {
                stateData[x] = new DRMLicenseStateData(b, (x * LICENSESTATEDATASIZE) + 8);
            }
        }

        public int dwSize;
        public int dwNumStates;
        public DRMLicenseStateData[] stateData;
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4), UnmanagedName("DRM_LICENSE_STATE_DATA")]
    public struct DRMLicenseStateData
    {
        public DRMLicenseStateData(byte[] b, int iOffset)
        {
            dwStreamId = BitConverter.ToInt32(b, 0 + iOffset);
            dwCategory = (LicenseStateCategory)BitConverter.ToInt32(b, 4 + iOffset);
            dwNumCounts = BitConverter.ToInt32(b, 8 + iOffset);

            dwCount = new int[4];
            dwCount[0] = BitConverter.ToInt32(b, 12 + iOffset);
            dwCount[1] = BitConverter.ToInt32(b, 16 + iOffset);
            dwCount[2] = BitConverter.ToInt32(b, 20 + iOffset);
            dwCount[3] = BitConverter.ToInt32(b, 24 + iOffset);
            dwNumDates = BitConverter.ToInt32(b, 28 + iOffset);

            datetime = new long[4];
            datetime[0] = BitConverter.ToInt64(b, 32 + iOffset);
            datetime[1] = BitConverter.ToInt64(b, 40 + iOffset);
            datetime[2] = BitConverter.ToInt64(b, 48 + iOffset);
            datetime[3] = BitConverter.ToInt64(b, 56 + iOffset);

            dwVague = (LicenseStateDataFlags)BitConverter.ToInt32(b, 64 + iOffset);
        }

        public int dwStreamId;
        public LicenseStateCategory dwCategory;
        public int dwNumCounts;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        public int[] dwCount;
        public int dwNumDates;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 4)]
        public long[] datetime;
        public LicenseStateDataFlags dwVague;
    }

    public static class MutexType
    {
        /// <summary> CLSID_WMMUTEX_Language </summary>
        public static readonly Guid Language = new Guid(0xD6E22A00, 0x35DA, 0x11D1, 0x90, 0x34, 0x00, 0xA0, 0xC9, 0x03, 0x49, 0xBE);

        /// <summary> CLSID_WMMUTEX_Bitrate </summary>
        public static readonly Guid Bitrate = new Guid(0xD6E22A01, 0x35DA, 0x11D1, 0x90, 0x34, 0x00, 0xA0, 0xC9, 0x03, 0x49, 0xBE);

        /// <summary> CLSID_WMMUTEX_Presentation </summary>
        public static readonly Guid Presentation = new Guid(0xD6E22A02, 0x35DA, 0x11D1, 0x90, 0x34, 0x00, 0xA0, 0xC9, 0x03, 0x49, 0xBE);

        /// <summary> CLSID_WMMUTEX_Unknown </summary>
        public static readonly Guid Unknown = new Guid(0xD6E22A03, 0x35DA, 0x11D1, 0x90, 0x34, 0x00, 0xA0, 0xC9, 0x03, 0x49, 0xBE);
    }

    public static class BandwidthSharingType
    {
        /// <summary> CLSID_WMBandwidthSharing_Exclusive </summary>
        public static readonly Guid Exclusive = new Guid(0xaf6060aa, 0x5197, 0x11d2, 0xb6, 0xaf, 0x00, 0xc0, 0x4f, 0xd9, 0x08, 0xe9);

        /// <summary> CLSID_WMBandwidthSharing_Partial </summary>
        public static readonly Guid Partial = new Guid(0xaf6060ab, 0x5197, 0x11d2, 0xb6, 0xaf, 0x00, 0xc0, 0x4f, 0xd9, 0x08, 0xe9);
    }

    [UnmanagedName("WM_SampleExtensionGUID_*")]
    public sealed class WM_SampleExtensionGUID
    {
        public static readonly Guid OutputCleanPoint = new Guid(0xf72a3c6f, 0x6eb4, 0x4ebc, 0xb1, 0x92, 0x9, 0xad, 0x97, 0x59, 0xe8, 0x28);
        public static readonly Guid Timecode = new Guid(0x399595ec, 0x8667, 0x4e2d, 0x8f, 0xdb, 0x98, 0x81, 0x4c, 0xe7, 0x6c, 0x1e);
        public static readonly Guid ChromaLocation = new Guid(0x4c5acca0, 0x9276, 0x4b2c, 0x9e, 0x4c, 0xa0, 0xed, 0xef, 0xdd, 0x21, 0x7e);
        public static readonly Guid ColorSpaceInfo = new Guid(0xf79ada56, 0x30eb, 0x4f2b, 0x9f, 0x7a, 0xf2, 0x4b, 0x13, 0x9a, 0x11, 0x57);
        public static readonly Guid UserDataInfo = new Guid(0x732bb4fa, 0x78be, 0x4549, 0x99, 0xbd, 0x2, 0xdb, 0x1a, 0x55, 0xb7, 0xa8);
        public static readonly Guid FileName = new Guid(0xe165ec0e, 0x19ed, 0x45d7, 0xb4, 0xa7, 0x25, 0xcb, 0xd1, 0xe2, 0x8e, 0x9b);
        public static readonly Guid ContentType = new Guid(0xd590dc20, 0x07bc, 0x436c, 0x9c, 0xf7, 0xf3, 0xbb, 0xfb, 0xf1, 0xa4, 0xdc);
        public static readonly Guid PixelAspectRatio = new Guid(0x1b1ee554, 0xf9ea, 0x4bc8, 0x82, 0x1a, 0x37, 0x6b, 0x74, 0xe4, 0xc4, 0xb8);
        public static readonly Guid SampleDuration = new Guid(0xc6bd9450, 0x867f, 0x4907, 0x83, 0xa3, 0xc7, 0x79, 0x21, 0xb7, 0x33, 0xad);
        public static readonly Guid SampleProtectionSalt = new Guid(0x5403deee, 0xb9ee, 0x438f, 0xaa, 0x83, 0x38, 0x4, 0x99, 0x7e, 0x56, 0x9d);
    }

    #endregion

    #region Interfaces

#if ALLOW_UNTESTED_INTERFACES

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("F369E2F0-E081-4FE6-8450-B810B2F410D1"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderTimecode
    {
        void GetTimecodeRangeCount(
            [In] short wStreamNum,
            out short pwRangeCount
            );

        void GetTimecodeRangeBounds(
            [In] short wStreamNum,
            [In] short wRangeNum,
            out int pStartTimecode,
            out int pEndTimecode
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("f6211f03-8d21-4e94-93e6-8510805f2d99"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMDeviceRegistration
    {
        void RegisterDevice(
            [In] int dwRegisterType,
            [In] IntPtr pbCertificate,
            int cbCertificate,
            Val16 SerialNumber,
            out IWMRegisteredDevice ppDevice);

        void UnregisterDevice(
            [In] int dwRegisterType,
            IntPtr pbCertificate,
            [In] int cbCertificate,
            Val16 SerialNumber);

        void GetRegistrationStats(
            [In] int dwRegisterType,
            out int pcRegisteredDevices);

        void GetFirstRegisteredDevice(
            [In] int dwRegisterType,
            out IWMRegisteredDevice ppDevice);

        void GetNextRegisteredDevice(
            out IWMRegisteredDevice ppDevice);

        void GetRegisteredDeviceByID(
            int dwRegisterType,
            IntPtr pbCertificate,
            int cbCertificate,
            Val16 SerialNumber,
            out IWMRegisteredDevice ppDevice);

    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("A73A0072-25A0-4c99-B4A5-EDE8101A6C39"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMDRMMessageParser
    {
        void ParseRegistrationReqMsg(
            IntPtr pbRegistrationReqMsg,
            int cbRegistrationReqMsg,
            out INSSBuffer ppDeviceCert,
            out Val16 pDeviceSerialNumber);

        void ParseLicenseRequestMsg(
            IntPtr pbLicenseRequestMsg,
            int cbLicenseRequestMsg,
            out INSSBuffer ppDeviceCert,
            out Val16 pDeviceSerialNumber,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrAction);
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("D2827540-3EE7-432C-B14C-DC17F085D3B3"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMDRMReader
    {
        void AcquireLicense(
            [In] int dwFlags
            );

        void CancelLicenseAcquisition();

        void Individualize(
            [In] int dwFlags
            );

        void CancelIndividualization();

        void MonitorLicenseAcquisition();

        void CancelMonitorLicenseAcquisition();

        void SetDRMProperty(
            [In] string pwstrName,
            [In] AttrDataType dwType,
            [In] byte[] pValue,
            [In] short cbLength
            );

        void GetDRMProperty(
            [In] string pwstrName,
            out AttrDataType pdwType,
            [Out] byte[] pValue,
            ref short pcbLength
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("befe7a75-9f1d-4075-b9d9-a3c37bda49a0"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMDRMReader2 : IWMDRMReader
    {
        #region IWMDRMReader Methods

        new void AcquireLicense(
            [In] int dwFlags
            );

        new void CancelLicenseAcquisition();

        new void Individualize(
            [In] int dwFlags
            );

        new void CancelIndividualization();

        new void MonitorLicenseAcquisition();

        new void CancelMonitorLicenseAcquisition();

        new void SetDRMProperty(
            [In] string pwstrName,
            [In] AttrDataType dwType,
            [In] byte[] pValue,
            [In] short cbLength
            );

        new void GetDRMProperty(
            [In] string pwstrName,
            out AttrDataType pdwType,
            [Out] byte[] pValue,
            ref short pcbLength
            );

        #endregion

        void SetEvaluateOutputLevelLicenses(
            [MarshalAs(UnmanagedType.Bool)] bool fEvaluate);

        void GetPlayOutputLevels(
            out PlayOpl pPlayOPL,
            out int pcbLength,
            out int pdwMinAppComplianceLevel);

        void GetCopyOutputLevels(
            out CopyOPL pCopyOPL,
            ref int pcbLength,
            out int pdwMinAppComplianceLevel);

        void TryNextLicense();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("e08672de-f1e7-4ff4-a0a3-fc4b08e4caf8"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMDRMReader3 : IWMDRMReader2
    {
        #region IWMDRMReader Methods

        new void AcquireLicense(
            [In] int dwFlags
            );

        new void CancelLicenseAcquisition();

        new void Individualize(
            [In] int dwFlags
            );

        new void CancelIndividualization();

        new void MonitorLicenseAcquisition();

        new void CancelMonitorLicenseAcquisition();

        new void SetDRMProperty(
            [In] string pwstrName,
            [In] AttrDataType dwType,
            [In] byte[] pValue,
            [In] short cbLength
            );

        new void GetDRMProperty(
            [In] string pwstrName,
            out AttrDataType pdwType,
            [Out] byte[] pValue,
            ref short pcbLength
            );

        #endregion

        #region IWMDRMReader2 Methods

        new void SetEvaluateOutputLevelLicenses(
            [MarshalAs(UnmanagedType.Bool)] bool fEvaluate);

        new void GetPlayOutputLevels(
            out PlayOpl pPlayOPL,
            out int pcbLength,
            out int pdwMinAppComplianceLevel);

        new void GetCopyOutputLevels(
            out CopyOPL pCopyOPL,
            ref int pcbLength,
            out int pdwMinAppComplianceLevel);

        new void TryNextLicense();

        #endregion

        void GetInclusionList(
            [MarshalAs(UnmanagedType.LPStruct)] out Guid ppGuids,
            out int pcGuids);
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("b1a887b2-a4f0-407a-b02e-efbd23bbecdf"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMDRMTranscryptionManager
    {
        void CreateTranscryptor(
            out IWMDRMTranscryptor ppTranscryptor);
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("69059850-6E6F-4bb2-806F-71863DDFC471"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMDRMTranscryptor
    {
        void Initialize(
            [MarshalAs(UnmanagedType.BStr)] string bstrFileName,
            [MarshalAs(UnmanagedType.BStr)] string pbLicenseRequestMsg,
            int cbLicenseRequestMsg,
            out INSSBuffer ppLicenseResponseMsg,
            IWMStatusCallback pCallback,
            IntPtr pvContext);

        void Seek(
            long hnsTime);

        void Read(
            IntPtr pbData,
            ref int pcbData);

        void Close();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("e0da439f-d331-496a-bece-18e5bac5dd23"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMDRMTranscryptor2 : IWMDRMTranscryptor
    {
        #region IWMDRMTranscryptor Methods

        new void Initialize(
            [MarshalAs(UnmanagedType.BStr)] string bstrFileName,
            [MarshalAs(UnmanagedType.BStr)] string pbLicenseRequestMsg,
            int cbLicenseRequestMsg,
            out INSSBuffer ppLicenseResponseMsg,
            IWMStatusCallback pCallback,
            IntPtr pvContext);

        new void Seek(
            long hnsTime);

        new void Read(
            IntPtr pbData,
            ref int pcbData);

        new void Close();

        #endregion

        void SeekEx(
            long cnsStartTime,
            long cnsDuration,
            float flRate,
            [MarshalAs(UnmanagedType.Bool)] bool fIncludeFileHeader);

        void ZeroAdjustTimestamps(
            [MarshalAs(UnmanagedType.Bool)] bool fEnable);

        void GetSeekStartTime(
            out long pcnsTime);

        void GetDuration(
            out long pcnsDuration);
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("D6EA5DD0-12A0-43F4-90AB-A3FD451E6A07"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMDRMWriter
    {
        void GenerateKeySeed(
            [Out] StringBuilder pwszKeySeed,
            ref int pcwchLength
            );

        void GenerateKeyID(
            [Out] StringBuilder pwszKeyID,
            ref int pcwchLength
            );

        void GenerateSigningKeyPair(
            [Out] StringBuilder pwszPrivKey,
            ref int pcwchPrivKeyLength,
            [Out] StringBuilder pwszPubKey,
            ref int pcwchPubKeyLength
            );

        void SetDRMAttribute(
            [In] short wStreamNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In] byte[] pValue,
            [In] short cbLength
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("38ee7a94-40e2-4e10-aa3f-33fd3210ed5b"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMDRMWriter2 : IWMDRMWriter
    {
        #region IWMDRMWriter Methods

        new void GenerateKeySeed(
            [Out] StringBuilder pwszKeySeed,
            ref int pcwchLength
            );

        new void GenerateKeyID(
            [Out] StringBuilder pwszKeyID,
            ref int pcwchLength
            );

        new void GenerateSigningKeyPair(
            [Out] StringBuilder pwszPrivKey,
            ref int pcwchPrivKeyLength,
            [Out] StringBuilder pwszPubKey,
            ref int pcwchPubKeyLength
            );

        new void SetDRMAttribute(
            [In] short wStreamNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In] byte[] pValue,
            [In] short cbLength
            );

        #endregion

        void SetWMDRMNetEncryption(
            [MarshalAs(UnmanagedType.Bool)] bool fSamplesEncrypted,
            IntPtr pbKeyID,
            int cbKeyID);
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("a7184082-a4aa-4dde-ac9c-e75dbd1117ce"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMDRMWriter3 : IWMDRMWriter2
    {
        #region IWMDRMWriter Methods

        new void GenerateKeySeed(
            [Out] StringBuilder pwszKeySeed,
            ref int pcwchLength
            );

        new void GenerateKeyID(
            [Out] StringBuilder pwszKeyID,
            ref int pcwchLength
            );

        new void GenerateSigningKeyPair(
            [Out] StringBuilder pwszPrivKey,
            ref int pcwchPrivKeyLength,
            [Out] StringBuilder pwszPubKey,
            ref int pcwchPubKeyLength
            );

        new void SetDRMAttribute(
            [In] short wStreamNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In] byte[] pValue,
            [In] short cbLength
            );

        #endregion

        #region IWMDRMWriter2 Methods

        new void SetWMDRMNetEncryption(
            [MarshalAs(UnmanagedType.Bool)] bool fSamplesEncrypted,
            IntPtr pbKeyID,
            int cbKeyID);

        #endregion

        void SetProtectStreamSamples(
            [In, MarshalAs(UnmanagedType.LPStruct)] ImportInitStruct pImportInitStruct);
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("6967F2C9-4E26-4b57-8894-799880F7AC7B"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMLicenseRevocationAgent
    {
        void GetLRBChallenge(
            IntPtr pMachineID,
            int dwMachineIDLength,
            IntPtr pChallenge,
            int dwChallengeLength,
            IntPtr pChallengeOutput,
            out int pdwChallengeOutputLength);

        void ProcessLRB(
            IntPtr pSignedLRB,
            int dwSignedLRBLength,
            IntPtr pSignedACK,
            out int pdwSignedACKLength);
    };

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("6A9FD8EE-B651-4bf0-B849-7D4ECE79A2B1"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMProximityDetection
    {
        void StartDetection(
            IntPtr pbRegistrationMsg,
            int cbRegistrationMsg,
            IntPtr pbLocalAddress,
            int cbLocalAddress,
            int dwExtraPortsAllowed,
            out INSSBuffer ppRegistrationResponseMsg,
            IWMStatusCallback pCallback,
            IntPtr pvContext);
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("18A2E7F8-428F-4acd-8A00-E64639BC93DE"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderAdvanced6 : IWMReaderAdvanced5
    {
        #region IWMReaderAdvanced Methods

        new void SetUserProvidedClock(
            [In, MarshalAs(UnmanagedType.Bool)] bool fUserClock
            );

        new void GetUserProvidedClock(
            [MarshalAs(UnmanagedType.Bool)] out bool pfUserClock
            );

        new void DeliverTime(
            [In] long cnsTime
            );

        new void SetManualStreamSelection(
            [In, MarshalAs(UnmanagedType.Bool)] bool fSelection
            );

        new void GetManualStreamSelection(
            [MarshalAs(UnmanagedType.Bool)] out bool pfSelection
            );

        new void SetStreamsSelected(
            [In] short cStreamCount,
            [In] short[] pwStreamNumbers,
            [In] StreamSelection[] pSelections
            );

        new void GetStreamSelected(
            [In] short wStreamNum,
            out StreamSelection pSelection
            );

        new void SetReceiveSelectionCallbacks(
            [In, MarshalAs(UnmanagedType.Bool)] bool fGetCallbacks
            );

        new void GetReceiveSelectionCallbacks(
            [MarshalAs(UnmanagedType.Bool)] out bool pfGetCallbacks
            );

        new void SetReceiveStreamSamples(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fReceiveStreamSamples
            );

        new void GetReceiveStreamSamples(
            [In] short wStreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfReceiveStreamSamples
            );

        new void SetAllocateForOutput(
            [In] int dwOutputNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fAllocate
            );

        new void GetAllocateForOutput(
            [In] int dwOutputNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfAllocate
            );

        new void SetAllocateForStream(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fAllocate
            );

        new void GetAllocateForStream(
            [In] short dwSreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfAllocate
            );

        new void GetStatistics(
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] WMReaderStatistics pStatistics
            );

        new void SetClientInfo(
            [In, MarshalAs(UnmanagedType.LPStruct)] WMReaderClientInfo pClientInfo
            );

        new void GetMaxOutputSampleSize(
            [In] int dwOutput,
            out int pcbMax
            );

        new void GetMaxStreamSampleSize(
            [In] short wStream,
            out int pcbMax
            );

        new void NotifyLateDelivery(
            long cnsLateness
            );

        #endregion

        #region IWMReaderAdvanced2 Methods

        new void SetPlayMode(
            [In] PlayMode Mode
            );

        new void GetPlayMode(
            out PlayMode pMode
            );

        new void GetBufferProgress(
            out int pdwPercent,
            out long pcnsBuffering
            );

        new void GetDownloadProgress(
            out int pdwPercent,
            out long pqwBytesDownloaded,
            out long pcnsDownload
            );

        new void GetSaveAsProgress(
            out int pdwPercent
            );

        new void SaveFileAs(
            [In] string pwszFilename
            );

        new void GetProtocolName(
            [Out] StringBuilder pwszProtocol,
            ref int pcchProtocol
            );

        new void StartAtMarker(
            [In] short wMarkerIndex,
            [In] long cnsDuration,
            [In] float fRate,
            [In] IntPtr pvContext
            );

        new void GetOutputSetting(
            [In] int dwOutputNum,
            [In] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        new void SetOutputSetting(
            [In] int dwOutputNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In] byte[] pValue,
            [In] short cbLength
            );

        new void Preroll(
            [In] long cnsStart,
            [In] long cnsDuration,
            [In] float fRate
            );

        new void SetLogClientID(
            [In, MarshalAs(UnmanagedType.Bool)] bool fLogClientID
            );

        new void GetLogClientID(
            [MarshalAs(UnmanagedType.Bool)] out bool pfLogClientID
            );

        new void StopBuffering();

        new void OpenStream(
            [In] IStream pStream,
            [In] IWMReaderCallback pCallback,
            [In] IntPtr pvContext
            );

        #endregion

        #region IWMReaderAdvanced3 Methods

        new void StopNetStreaming();

        new void StartAtPosition(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.LPStruct)] RA3Union pvOffsetStart,
            [In, MarshalAs(UnmanagedType.LPStruct)] RA3Union pvDuration,
            [In] OffsetFormat dwOffsetFormat,
            [In] float fRate,
            [In] IntPtr pvContext
            );

        #endregion

        #region IWMReaderAdvanced4 Methods

        new void GetLanguageCount(
            [In] int dwOutputNum,
            out short pwLanguageCount
            );

        new void GetLanguage(
            [In] int dwOutputNum,
            [In] short wLanguage,
            [Out] StringBuilder pwszLanguageString,
            ref short pcchLanguageStringLength
            );

        new void GetMaxSpeedFactor(
            out double pdblFactor
            );

        new void IsUsingFastCache(
            [MarshalAs(UnmanagedType.Bool)] out bool pfUsingFastCache
            );

        new void AddLogParam(
            [In] string wszNameSpace,
            [In] string wszName,
            [In] string wszValue
            );

        new void SendLogParams();

        new void CanSaveFileAs(
            [MarshalAs(UnmanagedType.Bool)] out bool pfCanSave
            );

        new void CancelSaveFileAs();

        new void GetURL(
            [Out] StringBuilder pwszURL,
            ref int pcchURL
            );

        #endregion

        #region IWMReaderAdvanced5 Methods

        new void SetPlayerHook(
            int dwOutputNum,
            IWMPlayerHook pHook);

        #endregion

        void SetProtectStreamSamples(
            IntPtr pbCertificate,
            int cbCertificate,
            int dwCertificateType,
            int dwFlags,
            IntPtr pbInitializationVector,
            out int pcbInitializationVector);
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("a4503bec-5508-4148-97ac-bfa75760a70d"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMRegisteredDevice
    {
        void GetDeviceSerialNumber(
            out Val16 pSerialNumber);

        void GetDeviceCertificate(
            out INSSBuffer ppCertificate);

        void GetDeviceType(
            out int pdwType);

        void GetAttributeCount(
            out int pcAttributes);

        void GetAttributeByIndex(
            int dwIndex,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrName,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrValue);

        void GetAttributeByName(
            [MarshalAs(UnmanagedType.BStr)] string bstrName,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrValue);

        void SetAttributeByName(
            [MarshalAs(UnmanagedType.BStr)] string bstrName,
            [MarshalAs(UnmanagedType.BStr)] string bstrValue);

        void Approve(
            [MarshalAs(UnmanagedType.Bool)] bool fApprove);

        void IsValid(
            [MarshalAs(UnmanagedType.Bool)] out bool pfValid);

        void IsApproved(
            [MarshalAs(UnmanagedType.Bool)] out bool pfApproved);

        void IsWmdrmCompliant(
            [MarshalAs(UnmanagedType.Bool)] out bool pfCompliant);

        void IsOpened(
            [MarshalAs(UnmanagedType.Bool)] out bool pfOpened);

        void Open();

        void Close();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("990641B0-739F-4E94-A808-9888DA8F75AF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMCodecVideoAccelerator
    {
        void NegotiateConnection(
            [In, MarshalAs(UnmanagedType.Interface)] object pIAMVA,
            [In, MarshalAs(UnmanagedType.LPStruct)] AMMediaType pMediaType
            );

        void SetPlayerNotify(
            [In] IWMPlayerTimestampHook pHook
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("9F0AA3B6-7267-4D89-88F2-BA915AA5C4C6"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMImageInfo
    {
        void GetImageCount(
            out int pcImages
            );

        void GetImage(
            [In] int wIndex,
            ref short pcchMIMEType,
            [Out] StringBuilder pwszMIMEType,
            ref short pcchDescription,
            [Out] StringBuilder pwszDescription,
            out short pImageType,
            ref int pcbImageData,
            out byte[] pbImageData
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("6F497062-F2E2-4624-8EA7-9DD40D81FC8D"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMWatermarkInfo
    {
        void GetWatermarkEntryCount(
            [In] WaterMarkEntryType wmetType,
            out int pdwCount
            );

        void GetWatermarkEntry(
            [In] WaterMarkEntryType wmetType,
            [In] int dwEntryNum,
            out WaterMarkEntry pEntry
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("61103CA4-2033-11D2-9EF1-006097D2D7CF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMSBufferAllocator
    {
        void AllocateBuffer(
            [In] int dwMaxBufferSize,
            out INSSBuffer ppBuffer
            );

        void AllocatePageSizeBuffer(
            [In] int dwMaxBufferSize,
            out INSSBuffer ppBuffer
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("D98EE251-34E0-4A2D-9312-9B4C788D9FA1"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMCodecAMVideoAccelerator
    {
        void SetAcceleratorInterface(
            [In, MarshalAs(UnmanagedType.Interface)] object pIAMVA
            );

        void NegotiateConnection(
            [In, MarshalAs(UnmanagedType.LPStruct)] AMMediaType pMediaType
            );

        void SetPlayerNotify(
            [In] IWMPlayerTimestampHook pHook
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("28580DDA-D98E-48D0-B7AE-69E473A02825"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMPlayerTimestampHook
    {
        void MapTimestamp(
            [In] long rtIn, out long prtOut
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("BDDC4D08-944D-4D52-A612-46C3FDA07DD4"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderAccelerator
    {
        void GetCodecInterface(
            [In] int dwOutputNum,
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid riid,
            out IntPtr ppvCodecInterface
            );

        void Notify(
            [In] int dwOutputNum,
            [In, MarshalAs(UnmanagedType.LPStruct)] AMMediaType pSubtype
            );
    }

#endif

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("45086030-F7E4-486a-B504-826BB5792A3B"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IConfigAsfWriter
    {
        void ConfigureFilterUsingProfileId(
            [In] int dwProfileId);

        void GetCurrentProfileId(
            [Out] out int pdwProfileId);

        void ConfigureFilterUsingProfileGuid(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidProfile);

        void GetCurrentProfileGuid(
            [Out] out Guid pProfileGuid);

        void ConfigureFilterUsingProfile(
            [In] IWMProfile pProfile);

        void GetCurrentProfileGuid(
            [Out] out IWMProfile ppProfile);

        void SetIndexMode(
            [In, MarshalAs(UnmanagedType.Bool)] bool bIndexFile);

        void GetIndexMode(
            [Out, MarshalAs(UnmanagedType.Bool)] out bool pbIndexFile);
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("E1CD3524-03D7-11D2-9EED-006097D2D7CF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface INSSBuffer
    {
        void GetLength(
            out int pdwLength
            );

        void SetLength(
            [In] int dwLength
            );

        void GetMaxLength(
            out int pdwLength
            );

        void GetBuffer(
            out IntPtr ppdwBuffer
            );

        void GetBufferAndLength(
            out IntPtr ppdwBuffer,
            out int pdwLength
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("BB3C6389-1633-4E92-AF14-9F3173BA39D0"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMAddressAccess
    {
        void GetAccessEntryCount(
            [In] AEType aeType,
            out int pcEntries
            );

        void GetAccessEntry(
            [In] AEType aeType,
            [In] int dwEntryNum,
            out WMAddressAccessEntry pAddrAccessEntry
            );

        void AddAccessEntry(
            [In] AEType aeType,
            [In] ref WMAddressAccessEntry pAddrAccessEntry
           );

        void RemoveAccessEntry(
            [In] AEType aeType,
            [In] int dwEntryNum
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("65A83FC2-3E98-4D4D-81B5-2A742886B33D"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMAddressAccess2 : IWMAddressAccess
    {
        #region IWMAddressAccess Methods

        new void GetAccessEntryCount(
            [In] AEType aeType,
            out int pcEntries
            );

        new void GetAccessEntry(
            [In] AEType aeType,
            [In] int dwEntryNum,
            out WMAddressAccessEntry pAddrAccessEntry
            );

        new void AddAccessEntry(
            [In] AEType aeType,
            [In] ref WMAddressAccessEntry pAddrAccessEntry
            );

        new void RemoveAccessEntry(
            [In] AEType aeType,
            [In] int dwEntryNum
            );

        #endregion

        void GetAccessEntryEx(
            [In] AEType aeType,
            [In] int dwEntryNum,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrAddress,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrMask
            );

        void AddAccessEntryEx(
            [In] AEType aeType,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrAddress,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrMask
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("3C8E0DA6-996F-4FF3-A1AF-4838F9377E2E"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMBackupRestoreProps
    {
        void GetPropCount(
            out short pcProps
            );

        void GetPropByIndex(
            [In] short wIndex,
            [Out] string pwszName,
            ref short pcchNameLen,
            out AttrDataType pType,
            out byte[] pValue,
            ref short pcbLength
            );

        void GetPropByName(
            [In] string pszName,
            out AttrDataType pType,
            out byte[] pValue,
            ref short pcbLength
            );

        void SetProp(
            [In] string pszName,
            [In] AttrDataType Type,
            [In, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            [In] short cbLength
            );

        void RemoveProp(
            [In] string pcwszName
            );

        void RemoveAllProps();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("AD694AF1-F8D9-42F8-BC47-70311B0C4F9E"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMBandwidthSharing : IWMStreamList
    {
        #region IWMStreamList Methods

        new void GetStreams(
            [Out, MarshalAs(UnmanagedType.LPArray)] short[] pwStreamNumArray,
            ref short pcStreams
            );

        new void AddStream(
            [In] short wStreamNum
            );

        new void RemoveStream(
            [In] short wStreamNum
            );

        #endregion

        void GetType(
            out Guid pguidType
            );

        void SetType(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType
            );

        void GetBandwidth(
            out int pdwBitrate,
            out int pmsBufferWindow
            );

        void SetBandwidth(
            [In] int dwBitrate,
            [In] int msBufferWindow
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("73C66010-A299-41DF-B1F0-CCF03B09C1C6"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMClientConnections
    {
        void GetClientCount(
            out int pcClients
            );

        void GetClientProperties(
            [In] int dwClientNum,
            out WMClientProperties pClientProperties
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("4091571E-4701-4593-BB3D-D5F5F0C74246"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMClientConnections2 : IWMClientConnections
    {
        #region IWMClientConnections Methods

        new void GetClientCount(
            out int pcClients
            );

        new void GetClientProperties(
            [In] int dwClientNum,
            out WMClientProperties pClientProperties
            );

        #endregion

        void GetClientInfo(
            [In] int dwClientNum,
            [Out] StringBuilder pwszNetworkAddress,
            ref int pcchNetworkAddress,
            [Out] StringBuilder pwszPort,
            ref int pcchPort,
            [Out] StringBuilder pwszDNSName,
            ref int pcchDNSName
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("A970F41E-34DE-4A98-B3BA-E4B3CA7528F0"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMCodecInfo
    {
        void GetCodecInfoCount(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            out int pcCodecs
            );

        void GetCodecFormatCount(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            [In] int dwCodecIndex,
            out int pcFormat
            );

        void GetCodecFormat(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            [In] int dwCodecIndex,
            [In] int dwFormatIndex,
            out IWMStreamConfig ppIStreamConfig
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("AA65E273-B686-4056-91EC-DD768D4DF710"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMCodecInfo2 : IWMCodecInfo
    {
        #region IWMCodecInfo Methods

        new void GetCodecInfoCount(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            out int pcCodecs
            );

        new void GetCodecFormatCount(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            [In] int dwCodecIndex,
            out int pcFormat
            );

        new void GetCodecFormat(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            [In] int dwCodecIndex,
            [In] int dwFormatIndex,
            out IWMStreamConfig ppIStreamConfig
            );

        #endregion

        void GetCodecName(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            [In] int dwCodecIndex,
            [Out] StringBuilder wszName,
            ref int pcchName
            );

        void GetCodecFormatDesc(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            [In] int dwCodecIndex,
            [In] int dwFormatIndex,
            out IWMStreamConfig ppIStreamConfig,
            [Out] StringBuilder wszDesc,
            ref int pcchDesc
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("7E51F487-4D93-4F98-8AB4-27D0565ADC51"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMCodecInfo3 : IWMCodecInfo2
    {
        #region IWMCodecInfo Methods

        new void GetCodecInfoCount(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            out int pcCodecs
            );

        new void GetCodecFormatCount(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            [In] int dwCodecIndex,
            out int pcFormat
            );

        new void GetCodecFormat(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            [In] int dwCodecIndex,
            [In] int dwFormatIndex,
            out IWMStreamConfig ppIStreamConfig
            );

        #endregion

        #region IWMCodecInfo2 Methods

        new void GetCodecName(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            [In] int dwCodecIndex,
            [Out] StringBuilder wszName,
            ref int pcchName
            );

        new void GetCodecFormatDesc(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            [In] int dwCodecIndex,
            [In] int dwFormatIndex,
            out IWMStreamConfig ppIStreamConfig,
            [Out] StringBuilder wszDesc,
            ref int pcchDesc
            );

        #endregion

        void GetCodecFormatProp(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            [In] int dwCodecIndex,
            [In] int dwFormatIndex,
            [In, MarshalAs(UnmanagedType.LPWStr)] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref int pdwSize
            );

        void GetCodecProp(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            [In] int dwCodecIndex,
            [In, MarshalAs(UnmanagedType.LPWStr)] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref int pdwSize
            );

        void SetCodecEnumerationSetting(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            [In] int dwCodecIndex,
            [In, MarshalAs(UnmanagedType.LPWStr)] string pszName,
            [In] AttrDataType Type,
            [In, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            [In] int dwSize
            );

        void GetCodecEnumerationSetting(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType,
            [In] int dwCodecIndex,
            [In, MarshalAs(UnmanagedType.LPWStr)] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref int pdwSize
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("342E0EB7-E651-450C-975B-2ACE2C90C48E"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMCredentialCallback
    {
        void AcquireCredentials(
            [In, MarshalAs(UnmanagedType.LPWStr)] string pwszRealm,
            [In, MarshalAs(UnmanagedType.LPWStr)] string pwszSite,
            [In, Out, MarshalAs(UnmanagedType.LPArray, SizeParamIndex=3)] char[] pwszUser,
            [In] int cchUser,
            [In, Out, MarshalAs(UnmanagedType.LPArray, SizeParamIndex=5)] char[] pwszPassword,
            [In] int cchPassword,
            [In] int hrStatus,
            ref CredentialFlags pdwFlags
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("FF130EBC-A6C3-42A6-B401-C3382C3E08B3"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMDRMEditor
    {
        void GetDRMProperty(
            [In] string pwstrName,
            out AttrDataType pdwType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BDA-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMHeaderInfo
    {
        void GetAttributeCount(
            [In] short wStreamNum,
            out short pcAttributes
            );

        void GetAttributeByIndex(
            [In] short wIndex,
            [In] ref short pwStreamNum,
            [Out] StringBuilder pwszName,
            ref short pcchNameLen,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        void GetAttributeByName(
            [In] ref short pwStreamNum,
            [In] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        void SetAttribute(
            [In] short wStreamNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            [In] short cbLength
            );

        void GetMarkerCount(
            out short pcMarkers
            );

        void GetMarker(
            [In] short wIndex,
            [Out] StringBuilder pwszMarkerName,
            ref short pcchMarkerNameLen,
            out long pcnsMarkerTime
            );

        void AddMarker(
            [In] string pwszMarkerName,
            [In] long cnsMarkerTime
            );

        void RemoveMarker(
            [In] short wIndex
            );

        void GetScriptCount(
            out short pcScripts
            );

        void GetScript(
            [In] short wIndex,
            [Out] StringBuilder pwszType,
            ref short pcchTypeLen,
            [Out] StringBuilder pwszCommand,
            ref short pcchCommandLen,
            out long pcnsScriptTime
            );

        void AddScript(
            [In] string pwszType,
            [In] string pwszCommand,
            [In] long cnsScriptTime
            );

        void RemoveScript(
            [In] short wIndex
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("15CF9781-454E-482E-B393-85FAE487A810"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMHeaderInfo2 : IWMHeaderInfo
    {
        #region IWMHeaderInfo Methods

        new void GetAttributeCount(
            [In] short wStreamNum,
            out short pcAttributes
            );

        new void GetAttributeByIndex(
            [In] short wIndex,
            ref short pwStreamNum,
            [Out] StringBuilder pwszName,
            ref short pcchNameLen,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        new void GetAttributeByName(
            ref short pwStreamNum,
            [In] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        new void SetAttribute(
            [In] short wStreamNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            [In] short cbLength
            );

        new void GetMarkerCount(
            out short pcMarkers
            );

        new void GetMarker(
            [In] short wIndex,
            [Out] StringBuilder pwszMarkerName,
            ref short pcchMarkerNameLen,
            out long pcnsMarkerTime
            );

        new void AddMarker(
            [In] string pwszMarkerName,
            [In] long cnsMarkerTime
            );

        new void RemoveMarker(
            [In] short wIndex
            );

        new void GetScriptCount(
            out short pcScripts
            );

        new void GetScript(
            [In] short wIndex,
            [Out] StringBuilder pwszType,
            ref short pcchTypeLen,
            [Out] StringBuilder pwszCommand,
            ref short pcchCommandLen,
            out long pcnsScriptTime
            );

        new void AddScript(
            [In] string pwszType,
            [In] string pwszCommand,
            [In] long cnsScriptTime
            );

        new void RemoveScript(
            [In] short wIndex
            );

        #endregion

        void GetCodecInfoCount(
            out int pcCodecInfos
            );

        void GetCodecInfo(
            [In] int wIndex,
            ref short pcchName,
            [Out] StringBuilder pwszName,
            ref short pcchDescription,
            [Out] StringBuilder pwszDescription,
            out CodecInfoType pCodecType,
            ref short pcbCodecInfo,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pbCodecInfo
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("15CC68E3-27CC-4ECD-B222-3F5D02D80BD5"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMHeaderInfo3 : IWMHeaderInfo2
    {
        #region IWMHeaderInfo Methods

        new void GetAttributeCount(
            [In] short wStreamNum,
            out short pcAttributes
            );

        new void GetAttributeByIndex(
            [In] short wIndex,
            ref short pwStreamNum,
            [Out] StringBuilder pwszName,
            ref short pcchNameLen,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        new void GetAttributeByName(
            ref short pwStreamNum,
            [In] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        new void SetAttribute(
            [In] short wStreamNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            [In] short cbLength
            );

        new void GetMarkerCount(
            out short pcMarkers
            );

        new void GetMarker(
            [In] short wIndex,
            [Out] StringBuilder pwszMarkerName,
            ref short pcchMarkerNameLen,
            out long pcnsMarkerTime
            );

        new void AddMarker(
            [In] string pwszMarkerName,
            [In] long cnsMarkerTime
            );

        new void RemoveMarker(
            [In] short wIndex
            );

        new void GetScriptCount(
            out short pcScripts
            );

        new void GetScript(
            [In] short wIndex,
            [Out] StringBuilder pwszType,
            ref short pcchTypeLen,
            [Out] StringBuilder pwszCommand,
            ref short pcchCommandLen,
            out long pcnsScriptTime
            );

        new void AddScript(
            [In] string pwszType,
            [In] string pwszCommand,
            [In] long cnsScriptTime
            );

        new void RemoveScript(
            [In] short wIndex
            );

        #endregion

        #region IWMHeaderInfo2 Methods

        new void GetCodecInfoCount(
            out int pcCodecInfos
            );

        new void GetCodecInfo(
            [In] int wIndex,
            ref short pcchName,
            [Out] StringBuilder pwszName,
            ref short pcchDescription,
            [Out] StringBuilder pwszDescription,
            out CodecInfoType pCodecType,
            ref short pcbCodecInfo,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pbCodecInfo
            );

        #endregion

        void GetAttributeCountEx(
            [In] short wStreamNum,
            out short pcAttributes
            );

        void GetAttributeIndices(
            [In] short wStreamNum,
            [In] string pwszName,
            [In] WmShort pwLangIndex,
            [Out, MarshalAs(UnmanagedType.LPArray)] short[] pwIndices,
            ref short pwCount
            );

        void GetAttributeByIndexEx(
            [In] short wStreamNum,
            [In] short wIndex,
            [Out] StringBuilder pwszName,
            ref short pwNameLen,
            out AttrDataType pType,
            out short pwLangIndex,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref int pdwDataLength
            );

        void ModifyAttribute(
            [In] short wStreamNum,
            [In] short wIndex,
            [In] AttrDataType Type,
            [In] short wLangIndex,
            [In, MarshalAs(UnmanagedType.LPArray)] byte [] pValue,
            [In] int dwLength
            );

        void AddAttribute(
            [In] short wStreamNum,
            [In] string pszName,
            out short pwIndex,
            [In] AttrDataType Type,
            [In] short wLangIndex,
            [In, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            [In] int dwLength
            );

        void DeleteAttribute(
            [In] short wStreamNum,
            [In] short wIndex
            );

        void AddCodecInfo(
            [In] string pwszName,
            [In] string pwszDescription,
            [In] CodecInfoType codecType,
            [In] short cbCodecInfo,
            [In, MarshalAs(UnmanagedType.LPArray)] byte[] pbCodecInfo
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("6D7CDC71-9888-11D3-8EDC-00C04F6109CF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMIndexer
    {
        void StartIndexing(
            [In] string pwszURL,
            [In] IWMStatusCallback pCallback,
            [In] IntPtr pvContext
            );

        void Cancel();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("B70F1E42-6255-4DF0-A6B9-02B212D9E2BB"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMIndexer2 : IWMIndexer
    {
        #region IWMIndexer Methods

        new void StartIndexing(
            [In] string pwszURL,
            [In] IWMStatusCallback pCallback,
            [In] IntPtr pvContext
            );

        new void Cancel();

        #endregion

        void Configure(
            [In] short wStreamNum,
            [In] IndexerType nIndexerType,
            [In] WmInt pvInterval,
            [In] WmShort pvIndexType
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BD5-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMInputMediaProps : IWMMediaProps
    {
        #region IWMMediaProps Methods

        new void GetType(
            out Guid pguidType
            );

        new void GetMediaType(
            [In, Out, MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(MTMarshaler))] AMMediaType pType,
            ref int pcbType
            );

        new void SetMediaType(
            [In, MarshalAs(UnmanagedType.LPStruct)] AMMediaType pType
            );

        #endregion

        void GetConnectionName(
            [Out] StringBuilder pwszName,
            ref short pcchName
            );

        void GetGroupName(
            [Out] StringBuilder pwszName,
            ref short pcchName
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("6816DAD3-2B4B-4C8E-8149-874C3483A753"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMIStreamProps
    {
        void GetProperty(
            [In] StringBuilder pszName,
            out AttrDataType pType,
            IntPtr pValue,
            ref int pdwSize
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("DF683F00-2D49-4D8E-92B7-FB19F6A0DC57"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMLanguageList
    {
        void GetLanguageCount(
            out short pwCount
            );

        void GetLanguageDetails(
            [In] short wIndex,
            [Out] StringBuilder pwszLanguageString,
            ref short pcchLanguageStringLength
            );

        void AddLanguageByRFC1766String(
            [In] string pwszLanguageString,
            out short pwIndex
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("05E5AC9F-3FB6-4508-BB43-A4067BA1EBE8"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMLicenseBackup
    {
        void BackupLicenses(
            [In] BackupRestoreFlags dwFlags,
            [In] IWMStatusCallback pCallback
            );

        void CancelLicenseBackup();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("C70B6334-A22E-4EFB-A245-15E65A004A13"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMLicenseRestore
    {
        void RestoreLicenses(
            [In] BackupRestoreFlags dwFlags,
            [In] IWMStatusCallback pCallback
            );

        void CancelLicenseRestore();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BCE-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMMediaProps
    {
        void GetType(
            out Guid pguidType
            );

        void GetMediaType(
            [In, Out, MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(MTMarshaler))] AMMediaType pType,
            ref int pcbType
            );

        void SetMediaType(
            [In, MarshalAs(UnmanagedType.LPStruct)] AMMediaType pType
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BD9-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMMetadataEditor
    {
        void Open(
            [In] string pwszFilename
            );

        void Close();

        void Flush();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("203CFFE3-2E18-4FDF-B59D-6E71530534CF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMMetadataEditor2 : IWMMetadataEditor
    {
        #region IWMMetadataEditor Methods

        new void Open(
            [In] string pwszFilename
            );

        new void Close();

        new void Flush();

        #endregion

        void OpenEx(
            [In] string pwszFilename,
            [In] MetaDataAccess dwDesiredAccess,
            [In] MetaDataShare dwShareMode
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BDE-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMMutualExclusion : IWMStreamList
    {
        #region IWMStreamList Methods

        new void GetStreams(
            [Out, MarshalAs(UnmanagedType.LPArray)] short[] pwStreamNumArray,
            ref short pcStreams
            );

        new void AddStream(
            [In] short wStreamNum
            );

        new void RemoveStream(
            [In] short wStreamNum
            );

        #endregion

        void GetType(
            out Guid pguidType
            );

        void SetType(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("0302B57D-89D1-4BA2-85C9-166F2C53EB91"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMMutualExclusion2 : IWMMutualExclusion
    {
        #region IWMStreamList Methods

        new void GetStreams(
            [Out, MarshalAs(UnmanagedType.LPArray)] short[] pwStreamNumArray,
            ref short pcStreams
            );

        new void AddStream(
            [In] short wStreamNum
            );

        new void RemoveStream(
            [In] short wStreamNum
            );

        #endregion

        #region IWMMutualExclusion Methods

        new void GetType(
            out Guid pguidType
            );

        new void SetType(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidType
            );

        #endregion

        void GetName(
            [Out] StringBuilder pwszName,
            ref short pcchName
            );

        void SetName(
            [In] string pwszName
            );

        void GetRecordCount(
            out short pwRecordCount
            );

        void AddRecord();

        void RemoveRecord(
            [In] short wRecordNumber
            );

        void GetRecordName(
            [In] short wRecordNumber,
            [Out] StringBuilder pwszRecordName,
            ref short pcchRecordName
            );

        void SetRecordName(
            [In] short wRecordNumber,
            [In] string pwszRecordName
            );

        void GetStreamsForRecord(
            [In] short wRecordNumber,
            [Out, MarshalAs(UnmanagedType.LPArray)] short[] pwStreamNumArray,
            ref short pcStreams
            );

        void AddStreamForRecord(
            [In] short wRecordNumber,
            [In] short wStreamNumber
            );

        void RemoveStreamForRecord(
            [In] short wRecordNumber,
            [In] short wStreamNumber
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BD7-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMOutputMediaProps : IWMMediaProps
    {
        #region IWMMediaProps Methods

        new void GetType(
            out Guid pguidType
            );

        new void GetMediaType(
            [In, Out, MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(MTMarshaler))] AMMediaType pType,
            ref int pcbType
            );

        new void SetMediaType(
            [In, MarshalAs(UnmanagedType.LPStruct)] AMMediaType pType
            );

        #endregion

        void GetStreamGroupName(
            [Out] StringBuilder pwszName,
            ref short pcchName
            );

        void GetConnectionName(
            [Out] StringBuilder pwszName,
            ref short pcchName
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("CDFB97AB-188F-40B3-B643-5B7903975C59"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMPacketSize
    {
        void GetMaxPacketSize(
            out int pdwMaxPacketSize
            );

        void SetMaxPacketSize(
            [In] int dwMaxPacketSize
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("8BFC2B9E-B646-4233-A877-1C6A079669DC"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMPacketSize2 : IWMPacketSize
    {
        #region IWMPacketSize Methods

        new void GetMaxPacketSize(
            out int pdwMaxPacketSize
            );

        new void SetMaxPacketSize(
            [In] int dwMaxPacketSize
            );

        #endregion

        void GetMinPacketSize(
            out int pdwMinPacketSize
            );

        void SetMinPacketSize(
            [In] int dwMinPacketSize
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BDB-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMProfile
    {
        void GetVersion(
            out WMVersion pdwVersion
            );

        void GetName(
            [Out] StringBuilder pwszName,
            ref int pcchName
            );

        void SetName(
            [In] string pwszName
            );

        void GetDescription(
            [Out] StringBuilder pwszDescription,
            ref int pcchDescription
            );

        void SetDescription(
            [In] string pwszDescription
            );

        void GetStreamCount(
            out int pcStreams
            );

        void GetStream(
            [In] int dwStreamIndex,
            out IWMStreamConfig ppConfig
            );

        void GetStreamByNumber(
            [In] short wStreamNum,
            out IWMStreamConfig ppConfig
            );

        void RemoveStream(
            [In] IWMStreamConfig pConfig
            );

        void RemoveStreamByNumber(
            [In] short wStreamNum
            );

        void AddStream(
            [In] IWMStreamConfig pConfig
            );

        void ReconfigStream(
            [In] IWMStreamConfig pConfig
            );

        void CreateNewStream(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidStreamType,
            out IWMStreamConfig ppConfig
            );

        void GetMutualExclusionCount(
            out int pcME
            );

        void GetMutualExclusion(
            [In] int dwMEIndex,
            out IWMMutualExclusion ppME
            );

        void RemoveMutualExclusion(
            [In] IWMMutualExclusion pME
            );

        void AddMutualExclusion(
            [In] IWMMutualExclusion pME
            );

        void CreateNewMutualExclusion(
            out IWMMutualExclusion ppME
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("07E72D33-D94E-4BE7-8843-60AE5FF7E5F5"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMProfile2 : IWMProfile
    {
        #region IWMProfile Methods

        new void GetVersion(
            out WMVersion pdwVersion
            );

        new void GetName(
            [Out] StringBuilder pwszName,
            ref int pcchName
            );

        new void SetName(
            [In] string pwszName
            );

        new void GetDescription(
            [Out] StringBuilder pwszDescription,
            ref int pcchDescription
            );

        new void SetDescription(
            [In] string pwszDescription
            );

        new void GetStreamCount(
            out int pcStreams
            );

        new void GetStream(
            [In] int dwStreamIndex,
            out IWMStreamConfig ppConfig
            );

        new void GetStreamByNumber(
            [In] short wStreamNum,
            out IWMStreamConfig ppConfig
            );

        new void RemoveStream(
            [In] IWMStreamConfig pConfig
            );

        new void RemoveStreamByNumber(
            [In] short wStreamNum
            );

        new void AddStream(
            [In] IWMStreamConfig pConfig
            );

        new void ReconfigStream(
            [In] IWMStreamConfig pConfig
            );

        new void CreateNewStream(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidStreamType,
            out IWMStreamConfig ppConfig
            );

        new void GetMutualExclusionCount(
            out int pcME
            );

        new void GetMutualExclusion(
            [In] int dwMEIndex,
            out IWMMutualExclusion ppME
            );

        new void RemoveMutualExclusion(
            [In] IWMMutualExclusion pME
            );

        new void AddMutualExclusion(
            [In] IWMMutualExclusion pME
            );

        new void CreateNewMutualExclusion(
            out IWMMutualExclusion ppME
            );

        #endregion

        void GetProfileID(
            out Guid pguidID
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("00EF96CC-A461-4546-8BCD-C9A28F0E06F5"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMProfile3 : IWMProfile2
    {
        #region IWMProfile Methods

        new void GetVersion(
            out WMVersion pdwVersion
            );

        new void GetName(
            [Out] StringBuilder pwszName,
            ref int pcchName
            );

        new void SetName(
            [In] string pwszName
            );

        new void GetDescription(
            [Out] StringBuilder pwszDescription,
            ref int pcchDescription
            );

        new void SetDescription(
            [In] string pwszDescription
            );

        new void GetStreamCount(
            out int pcStreams
            );

        new void GetStream(
            [In] int dwStreamIndex,
            out IWMStreamConfig ppConfig
            );

        new void GetStreamByNumber(
            [In] short wStreamNum,
            out IWMStreamConfig ppConfig
            );

        new void RemoveStream(
            [In] IWMStreamConfig pConfig
            );

        new void RemoveStreamByNumber(
            [In] short wStreamNum
            );

        new void AddStream(
            [In] IWMStreamConfig pConfig
            );

        new void ReconfigStream(
            [In] IWMStreamConfig pConfig
            );

        new void CreateNewStream(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidStreamType,
            out IWMStreamConfig ppConfig
            );

        new void GetMutualExclusionCount(
            out int pcME
            );

        new void GetMutualExclusion(
            [In] int dwMEIndex,
            out IWMMutualExclusion ppME
            );

        new void RemoveMutualExclusion(
            [In] IWMMutualExclusion pME
            );

        new void AddMutualExclusion(
            [In] IWMMutualExclusion pME
            );

        new void CreateNewMutualExclusion(
            out IWMMutualExclusion ppME
            );

        #endregion

        #region IWMProfile2 Methods

        new void GetProfileID(
            out Guid pguidID
            );

        #endregion

        void GetStorageFormat(
            out StorageFormat pnStorageFormat
            );

        void SetStorageFormat(
            [In] StorageFormat nStorageFormat
            );

        void GetBandwidthSharingCount(
            out int pcBS
            );

        void GetBandwidthSharing(
            [In] int dwBSIndex,
            out IWMBandwidthSharing ppBS
            );

        void RemoveBandwidthSharing(
            [In] IWMBandwidthSharing pBS
            );

        void AddBandwidthSharing(
            [In] IWMBandwidthSharing pBS
            );

        void CreateNewBandwidthSharing(
            out IWMBandwidthSharing ppBS
            );

        void GetStreamPrioritization(
            out IWMStreamPrioritization ppSP
            );

        void SetStreamPrioritization(
            [In] IWMStreamPrioritization pSP
            );

        void RemoveStreamPrioritization();

        void CreateNewStreamPrioritization(
            out IWMStreamPrioritization ppSP
            );

        void GetExpectedPacketCount(
            [In] long msDuration,
            out long pcPackets
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("D16679F2-6CA0-472D-8D31-2F5D55AEE155"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMProfileManager
    {
        void CreateEmptyProfile(
            [In] WMVersion dwVersion,
            out IWMProfile ppProfile
            );

        void LoadProfileByID(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidProfile,
            out IWMProfile ppProfile
            );

        void LoadProfileByData(
            [In] string pwszProfile,
            out IWMProfile ppProfile
            );

        void SaveProfile(
            [In] IWMProfile pIWMProfile,
            [Out] StringBuilder pwszProfile,
            ref int pdwLength
            );

        void GetSystemProfileCount(
            out int pcProfiles
            );

        void LoadSystemProfile(
            [In] int dwProfileIndex,
            out IWMProfile ppProfile
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("7A924E51-73C1-494D-8019-23D37ED9B89A"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMProfileManager2 : IWMProfileManager
    {
        #region IWMProfileManager Methods

        new void CreateEmptyProfile(
            [In] WMVersion dwVersion,
            out IWMProfile ppProfile
            );

        new void LoadProfileByID(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidProfile,
            out IWMProfile ppProfile
            );

        new void LoadProfileByData(
            [In] string pwszProfile,
            out IWMProfile ppProfile
            );

        new void SaveProfile(
            [In] IWMProfile pIWMProfile,
            [In] StringBuilder pwszProfile,
            ref int pdwLength
            );

        new void GetSystemProfileCount(
            out int pcProfiles
            );

        new void LoadSystemProfile(
            [In] int dwProfileIndex,
            out IWMProfile ppProfile
            );

        #endregion

        void GetSystemProfileVersion(
            out WMVersion pdwVersion
            );

        void SetSystemProfileVersion(
            WMVersion dwVersion
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("BA4DCC78-7EE0-4AB8-B27A-DBCE8BC51454"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMProfileManagerLanguage
    {
        void GetUserLanguageID(
            out short wLangID
            );

        void SetUserLanguageID(
            short wLangID
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("72995A79-5090-42A4-9C8C-D9D0B6D34BE5"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMPropertyVault
    {
        void GetPropertyCount(
            out int pdwCount
            );

        void GetPropertyByName(
            [In] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref int pdwSize
            );

        void SetProperty(
            [In] string pszName,
            [In] AttrDataType pType,
            [In, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            [In] int dwSize
            );

        void GetPropertyByIndex(
            [In] int dwIndex,
            [Out] StringBuilder pszName,
            ref int pdwNameLen,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref int pdwSize
            );

        void CopyPropertiesFrom(
            [In] IWMPropertyVault pIWMPropertyVault
            );

        void Clear();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BD6-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReader
    {
        void Open(
            [In] string pwszURL,
            [In] IWMReaderCallback pCallback,
            [In] IntPtr pvContext
            );

        void Close();

        void GetOutputCount(
            out int pcOutputs
            );

        void GetOutputProps(
            [In] int dwOutputNum,
            out IWMOutputMediaProps ppOutput
            );

        void SetOutputProps(
            [In] int dwOutputNum,
            [In] IWMOutputMediaProps pOutput
            );

        void GetOutputFormatCount(
            [In] int dwOutputNumber,
            out int pcFormats
            );

        void GetOutputFormat(
            [In] int dwOutputNumber,
            [In] int dwFormatNumber,
            out IWMOutputMediaProps ppProps
            );

        void Start(
            [In] long cnsStart,
            [In] long cnsDuration,
            [In] float fRate,
            [In] IntPtr pvContext
            );

        void Stop();

        void Pause();

        void Resume();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BEA-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderAdvanced
    {
        void SetUserProvidedClock(
            [In, MarshalAs(UnmanagedType.Bool)] bool fUserClock
            );

        void GetUserProvidedClock(
            [MarshalAs(UnmanagedType.Bool)] out bool pfUserClock
            );

        void DeliverTime(
            [In] long cnsTime
            );

        void SetManualStreamSelection(
            [In, MarshalAs(UnmanagedType.Bool)] bool fSelection
            );

        void GetManualStreamSelection(
            [MarshalAs(UnmanagedType.Bool)] out bool pfSelection
            );

        void SetStreamsSelected(
            [In] short cStreamCount,
            [In, MarshalAs(UnmanagedType.LPArray)] short[] pwStreamNumbers,
            [In, MarshalAs(UnmanagedType.LPArray)] StreamSelection[] pSelections
            );

        void GetStreamSelected(
            [In] short wStreamNum,
            out StreamSelection pSelection
            );

        void SetReceiveSelectionCallbacks(
            [In, MarshalAs(UnmanagedType.Bool)] bool fGetCallbacks
            );

        void GetReceiveSelectionCallbacks(
            [MarshalAs(UnmanagedType.Bool)] out bool pfGetCallbacks
            );

        void SetReceiveStreamSamples(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fReceiveStreamSamples
            );

        void GetReceiveStreamSamples(
            [In] short wStreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfReceiveStreamSamples
            );

        void SetAllocateForOutput(
            [In] int dwOutputNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fAllocate
            );

        void GetAllocateForOutput(
            [In] int dwOutputNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfAllocate
            );

        void SetAllocateForStream(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fAllocate
            );

        void GetAllocateForStream(
            [In] short dwSreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfAllocate
            );

        void GetStatistics(
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] WMReaderStatistics pStatistics
            );

        void SetClientInfo(
            [In, MarshalAs(UnmanagedType.LPStruct)] WMReaderClientInfo pClientInfo
            );

        void GetMaxOutputSampleSize(
            [In] int dwOutput,
            out int pcbMax
            );

        void GetMaxStreamSampleSize(
            [In] short wStream,
            out int pcbMax
            );

        void NotifyLateDelivery(
            long cnsLateness
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("AE14A945-B90C-4D0D-9127-80D665F7D73E"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderAdvanced2 : IWMReaderAdvanced
    {
        #region IWMReaderAdvanced Methods

        new void SetUserProvidedClock(
            [In, MarshalAs(UnmanagedType.Bool)] bool fUserClock
            );

        new void GetUserProvidedClock(
            [MarshalAs(UnmanagedType.Bool)] out bool pfUserClock
            );

        new void DeliverTime(
            [In] long cnsTime
            );

        new void SetManualStreamSelection(
            [In, MarshalAs(UnmanagedType.Bool)] bool fSelection
            );

        new void GetManualStreamSelection(
            [MarshalAs(UnmanagedType.Bool)] out bool pfSelection
            );

        new void SetStreamsSelected(
            [In] short cStreamCount,
            [In] short[] pwStreamNumbers,
            [In] StreamSelection[] pSelections
            );

        new void GetStreamSelected(
            [In] short wStreamNum,
            out StreamSelection pSelection
            );

        new void SetReceiveSelectionCallbacks(
            [In, MarshalAs(UnmanagedType.Bool)] bool fGetCallbacks
            );

        new void GetReceiveSelectionCallbacks(
            [MarshalAs(UnmanagedType.Bool)] out bool pfGetCallbacks
            );

        new void SetReceiveStreamSamples(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fReceiveStreamSamples
            );

        new void GetReceiveStreamSamples(
            [In] short wStreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfReceiveStreamSamples
            );

        new void SetAllocateForOutput(
            [In] int dwOutputNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fAllocate
            );

        new void GetAllocateForOutput(
            [In] int dwOutputNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfAllocate
            );

        new void SetAllocateForStream(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fAllocate
            );

        new void GetAllocateForStream(
            [In] short dwSreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfAllocate
            );

        new void GetStatistics(
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] WMReaderStatistics pStatistics
            );

        new void SetClientInfo(
            [In, MarshalAs(UnmanagedType.LPStruct)] WMReaderClientInfo pClientInfo
            );

        new void GetMaxOutputSampleSize(
            [In] int dwOutput,
            out int pcbMax
            );

        new void GetMaxStreamSampleSize(
            [In] short wStream,
            out int pcbMax
            );

        new void NotifyLateDelivery(
            long cnsLateness
            );

        #endregion

        void SetPlayMode(
            [In] PlayMode Mode
            );

        void GetPlayMode(
            out PlayMode pMode
            );

        void GetBufferProgress(
            out int pdwPercent,
            out long pcnsBuffering
            );

        void GetDownloadProgress(
            out int pdwPercent,
            out long pqwBytesDownloaded,
            out long pcnsDownload
            );

        void GetSaveAsProgress(
            out int pdwPercent
            );

        void SaveFileAs(
            [In] string pwszFilename
            );

        void GetProtocolName(
            [Out] StringBuilder pwszProtocol,
            ref int pcchProtocol
            );

        void StartAtMarker(
            [In] short wMarkerIndex,
            [In] long cnsDuration,
            [In] float fRate,
            [In] IntPtr pvContext
            );

        void GetOutputSetting(
            [In] int dwOutputNum,
            [In] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        void SetOutputSetting(
            [In] int dwOutputNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            [In] short cbLength
            );

        void Preroll(
            [In] long cnsStart,
            [In] long cnsDuration,
            [In] float fRate
            );

        void SetLogClientID(
            [In, MarshalAs(UnmanagedType.Bool)] bool fLogClientID
            );

        void GetLogClientID(
            [MarshalAs(UnmanagedType.Bool)] out bool pfLogClientID
            );

        void StopBuffering();

        void OpenStream(
            [In] IStream pStream,
            [In] IWMReaderCallback pCallback,
            [In] IntPtr pvContext
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("5DC0674B-F04B-4A4E-9F2A-B1AFDE2C8100"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderAdvanced3 : IWMReaderAdvanced2
    {
        #region IWMReaderAdvanced Methods

        new void SetUserProvidedClock(
            [In, MarshalAs(UnmanagedType.Bool)] bool fUserClock
            );

        new void GetUserProvidedClock(
            [MarshalAs(UnmanagedType.Bool)] out bool pfUserClock
            );

        new void DeliverTime(
            [In] long cnsTime
            );

        new void SetManualStreamSelection(
            [In, MarshalAs(UnmanagedType.Bool)] bool fSelection
            );

        new void GetManualStreamSelection(
            [MarshalAs(UnmanagedType.Bool)] out bool pfSelection
            );

        new void SetStreamsSelected(
            [In] short cStreamCount,
            [In] short[] pwStreamNumbers,
            [In] StreamSelection[] pSelections
            );

        new void GetStreamSelected(
            [In] short wStreamNum,
            out StreamSelection pSelection
            );

        new void SetReceiveSelectionCallbacks(
            [In, MarshalAs(UnmanagedType.Bool)] bool fGetCallbacks
            );

        new void GetReceiveSelectionCallbacks(
            [MarshalAs(UnmanagedType.Bool)] out bool pfGetCallbacks
            );

        new void SetReceiveStreamSamples(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fReceiveStreamSamples
            );

        new void GetReceiveStreamSamples(
            [In] short wStreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfReceiveStreamSamples
            );

        new void SetAllocateForOutput(
            [In] int dwOutputNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fAllocate
            );

        new void GetAllocateForOutput(
            [In] int dwOutputNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfAllocate
            );

        new void SetAllocateForStream(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fAllocate
            );

        new void GetAllocateForStream(
            [In] short dwSreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfAllocate
            );

        new void GetStatistics(
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] WMReaderStatistics pStatistics
            );

        new void SetClientInfo(
            [In, MarshalAs(UnmanagedType.LPStruct)] WMReaderClientInfo pClientInfo
            );

        new void GetMaxOutputSampleSize(
            [In] int dwOutput,
            out int pcbMax
            );

        new void GetMaxStreamSampleSize(
            [In] short wStream,
            out int pcbMax
            );

        new void NotifyLateDelivery(
            long cnsLateness
            );

        #endregion

        #region IWMReaderAdvanced2 Methods

        new void SetPlayMode(
            [In] PlayMode Mode
            );

        new void GetPlayMode(
            out PlayMode pMode
            );

        new void GetBufferProgress(
            out int pdwPercent,
            out long pcnsBuffering
            );

        new void GetDownloadProgress(
            out int pdwPercent,
            out long pqwBytesDownloaded,
            out long pcnsDownload
            );

        new void GetSaveAsProgress(
            out int pdwPercent
            );

        new void SaveFileAs(
            [In] string pwszFilename
            );

        new void GetProtocolName(
            [Out] StringBuilder pwszProtocol,
            ref int pcchProtocol
            );

        new void StartAtMarker(
            [In] short wMarkerIndex,
            [In] long cnsDuration,
            [In] float fRate,
            [In] IntPtr pvContext
            );

        new void GetOutputSetting(
            [In] int dwOutputNum,
            [In] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        new void SetOutputSetting(
            [In] int dwOutputNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In] byte[] pValue,
            [In] short cbLength
            );

        new void Preroll(
            [In] long cnsStart,
            [In] long cnsDuration,
            [In] float fRate
            );

        new void SetLogClientID(
            [In, MarshalAs(UnmanagedType.Bool)] bool fLogClientID
            );

        new void GetLogClientID(
            [MarshalAs(UnmanagedType.Bool)] out bool pfLogClientID
            );

        new void StopBuffering();

        new void OpenStream(
            [In] IStream pStream,
            [In] IWMReaderCallback pCallback,
            [In] IntPtr pvContext
            );

        #endregion

        void StopNetStreaming();

        void StartAtPosition(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.LPStruct)] RA3Union pvOffsetStart,
            [In, MarshalAs(UnmanagedType.LPStruct)] RA3Union pvDuration,
            [In] OffsetFormat dwOffsetFormat,
            [In] float fRate,
            [In] IntPtr pvContext
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("945A76A2-12AE-4D48-BD3C-CD1D90399B85"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderAdvanced4 : IWMReaderAdvanced3
    {
        #region IWMReaderAdvanced Methods

        new void SetUserProvidedClock(
            [In, MarshalAs(UnmanagedType.Bool)] bool fUserClock
            );

        new void GetUserProvidedClock(
            [MarshalAs(UnmanagedType.Bool)] out bool pfUserClock
            );

        new void DeliverTime(
            [In] long cnsTime
            );

        new void SetManualStreamSelection(
            [In, MarshalAs(UnmanagedType.Bool)] bool fSelection
            );

        new void GetManualStreamSelection(
            [MarshalAs(UnmanagedType.Bool)] out bool pfSelection
            );

        new void SetStreamsSelected(
            [In] short cStreamCount,
            [In] short[] pwStreamNumbers,
            [In] StreamSelection[] pSelections
            );

        new void GetStreamSelected(
            [In] short wStreamNum,
            out StreamSelection pSelection
            );

        new void SetReceiveSelectionCallbacks(
            [In, MarshalAs(UnmanagedType.Bool)] bool fGetCallbacks
            );

        new void GetReceiveSelectionCallbacks(
            [MarshalAs(UnmanagedType.Bool)] out bool pfGetCallbacks
            );

        new void SetReceiveStreamSamples(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fReceiveStreamSamples
            );

        new void GetReceiveStreamSamples(
            [In] short wStreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfReceiveStreamSamples
            );

        new void SetAllocateForOutput(
            [In] int dwOutputNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fAllocate
            );

        new void GetAllocateForOutput(
            [In] int dwOutputNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfAllocate
            );

        new void SetAllocateForStream(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fAllocate
            );

        new void GetAllocateForStream(
            [In] short dwSreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfAllocate
            );

        new void GetStatistics(
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] WMReaderStatistics pStatistics
            );

        new void SetClientInfo(
            [In, MarshalAs(UnmanagedType.LPStruct)] WMReaderClientInfo pClientInfo
            );

        new void GetMaxOutputSampleSize(
            [In] int dwOutput,
            out int pcbMax
            );

        new void GetMaxStreamSampleSize(
            [In] short wStream,
            out int pcbMax
            );

        new void NotifyLateDelivery(
            long cnsLateness
            );

        #endregion

        #region IWMReaderAdvanced2 Methods

        new void SetPlayMode(
            [In] PlayMode Mode
            );

        new void GetPlayMode(
            out PlayMode pMode
            );

        new void GetBufferProgress(
            out int pdwPercent,
            out long pcnsBuffering
            );

        new void GetDownloadProgress(
            out int pdwPercent,
            out long pqwBytesDownloaded,
            out long pcnsDownload
            );

        new void GetSaveAsProgress(
            out int pdwPercent
            );

        new void SaveFileAs(
            [In] string pwszFilename
            );

        new void GetProtocolName(
            [Out] StringBuilder pwszProtocol,
            ref int pcchProtocol
            );

        new void StartAtMarker(
            [In] short wMarkerIndex,
            [In] long cnsDuration,
            [In] float fRate,
            [In] IntPtr pvContext
            );

        new void GetOutputSetting(
            [In] int dwOutputNum,
            [In] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        new void SetOutputSetting(
            [In] int dwOutputNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In] byte[] pValue,
            [In] short cbLength
            );

        new void Preroll(
            [In] long cnsStart,
            [In] long cnsDuration,
            [In] float fRate
            );

        new void SetLogClientID(
            [In, MarshalAs(UnmanagedType.Bool)] bool fLogClientID
            );

        new void GetLogClientID(
            [MarshalAs(UnmanagedType.Bool)] out bool pfLogClientID
            );

        new void StopBuffering();

        new void OpenStream(
            [In] IStream pStream,
            [In] IWMReaderCallback pCallback,
            [In] IntPtr pvContext
            );

        #endregion

        #region IWMReaderAdvanced3 Methods

        new void StopNetStreaming();

        new void StartAtPosition(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.LPStruct)] RA3Union pvOffsetStart,
            [In, MarshalAs(UnmanagedType.LPStruct)] RA3Union pvDuration,
            [In] OffsetFormat dwOffsetFormat,
            [In] float fRate,
            [In] IntPtr pvContext
            );

        #endregion

        void GetLanguageCount(
            [In] int dwOutputNum,
            out short pwLanguageCount
            );

        void GetLanguage(
            [In] int dwOutputNum,
            [In] short wLanguage,
            [Out] StringBuilder pwszLanguageString,
            ref short pcchLanguageStringLength
            );

        void GetMaxSpeedFactor(
            out double pdblFactor
            );

        void IsUsingFastCache(
            [MarshalAs(UnmanagedType.Bool)] out bool pfUsingFastCache
            );

        void AddLogParam(
            [In] string wszNameSpace,
            [In] string wszName,
            [In] string wszValue
            );

        void SendLogParams();

        void CanSaveFileAs(
            [MarshalAs(UnmanagedType.Bool)] out bool pfCanSave
            );

        void CancelSaveFileAs();

        void GetURL(
            [Out] StringBuilder pwszURL,
            ref int pcchURL
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("9F762FA7-A22E-428D-93C9-AC82F3AAFE5A"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderAllocatorEx
    {
        void AllocateForStreamEx(
            [In] short wStreamNum,
            [In] int cbBuffer,
            out INSSBuffer ppBuffer,
            [In] SampleFlagEx dwFlags,
            [In] long cnsSampleTime,
            [In] long cnsSampleDuration,
            [In] IntPtr pvContext
            );

        void AllocateForOutputEx(
            [In] int dwOutputNum,
            [In] int cbBuffer,
            out INSSBuffer ppBuffer,
            [In] SampleFlagEx dwFlags,
            [In] long cnsSampleTime,
            [In] long cnsSampleDuration,
            [In] IntPtr pvContext
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BD8-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderCallback : IWMStatusCallback
    {
        #region IWMStatusCallback Methods

        new void OnStatus(
            [In] Status iStatus,
            [In] int hr,
            [In] AttrDataType dwType,
            [In] IntPtr pValue,
            [In] IntPtr pvContext
            );

        #endregion

        void OnSample(
            [In] int dwOutputNum,
            [In] long cnsSampleTime,
            [In] long cnsSampleDuration,
            [In] SampleFlag dwFlags,
            [In] INSSBuffer pSample,
            [In] IntPtr pvContext
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BEB-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderCallbackAdvanced
    {
        void OnStreamSample(
            [In] short wStreamNum,
            [In] long cnsSampleTime,
            [In] long cnsSampleDuration,
            [In] SampleFlag dwFlags,
            [In] INSSBuffer pSample,
            [In] IntPtr pvContext
            );

        void OnTime(
            [In] long cnsCurrentTime,
            [In] IntPtr pvContext
            );

        void OnStreamSelection(
            [In] short wStreamCount,
            [In, MarshalAs(UnmanagedType.LPArray, SizeParamIndex=0)] short[] pStreamNumbers,
            [In, MarshalAs(UnmanagedType.LPArray, SizeParamIndex=0)] StreamSelection[] pSelections,
            [In] IntPtr pvContext
            );

        void OnOutputPropsChanged(
            [In] int dwOutputNum,
            [In, MarshalAs(UnmanagedType.LPStruct)] AMMediaType pMediaType,
            [In] IntPtr pvContext
            );

        void AllocateForStream(
            [In] short wStreamNum,
            [In] int cbBuffer,
            out INSSBuffer ppBuffer,
            [In] IntPtr pvContext
            );

        void AllocateForOutput(
            [In] int dwOutputNum,
            [In] int cbBuffer,
            out INSSBuffer ppBuffer,
            [In] IntPtr pvContext
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BEC-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderNetworkConfig
    {
        void GetBufferingTime(
            out long pcnsBufferingTime
            );

        void SetBufferingTime(
            [In] long cnsBufferingTime
            );

        void GetUDPPortRanges(
            [Out, MarshalAs(UnmanagedType.LPArray)] WMPortNumberRange[] pRangeArray,
            ref int pcRanges
            );

        void SetUDPPortRanges(
            [In, MarshalAs(UnmanagedType.LPArray)] WMPortNumberRange[] pRangeArray,
            [In] int cRanges
            );

        void GetProxySettings(
            [In] string pwszProtocol,
            out ProxySettings pProxySetting
            );

        void SetProxySettings(
            [In] string pwszProtocol,
            [In] ProxySettings ProxySetting
            );

        void GetProxyHostName(
            [In] string pwszProtocol,
            [Out] StringBuilder pwszHostName,
            ref int pcchHostName
            );

        void SetProxyHostName(
            [In] string pwszProtocol,
            [In] string pwszHostName
            );

        void GetProxyPort(
            [In] string pwszProtocol,
            out int pdwPort
            );

        void SetProxyPort(
            [In] string pwszProtocol,
            [In] int dwPort
            );

        void GetProxyExceptionList(
            [In] string pwszProtocol,
            [Out] StringBuilder pwszExceptionList,
            ref int pcchExceptionList
            );

        void SetProxyExceptionList(
            [In] string pwszProtocol,
            [In] string pwszExceptionList
            );

        void GetProxyBypassForLocal(
            [In] string pwszProtocol,
            [MarshalAs(UnmanagedType.Bool)] out bool pfBypassForLocal
            );

        void SetProxyBypassForLocal(
            [In] string pwszProtocol,
            [In, MarshalAs(UnmanagedType.Bool)] bool fBypassForLocal
            );

        void GetForceRerunAutoProxyDetection(
            [MarshalAs(UnmanagedType.Bool)] out bool pfForceRerunDetection
            );

        void SetForceRerunAutoProxyDetection(
            [In, MarshalAs(UnmanagedType.Bool)] bool fForceRerunDetection
            );

        void GetEnableMulticast(
            [MarshalAs(UnmanagedType.Bool)] out bool pfEnableMulticast
            );

        void SetEnableMulticast(
            [In, MarshalAs(UnmanagedType.Bool)] bool fEnableMulticast
            );

        void GetEnableHTTP(
            [MarshalAs(UnmanagedType.Bool)] out bool pfEnableHTTP
            );

        void SetEnableHTTP(
            [In, MarshalAs(UnmanagedType.Bool)] bool fEnableHTTP
            );

        void GetEnableUDP(
            [MarshalAs(UnmanagedType.Bool)] out bool pfEnableUDP
            );

        void SetEnableUDP(
            [In, MarshalAs(UnmanagedType.Bool)] bool fEnableUDP
            );

        void GetEnableTCP(
            [MarshalAs(UnmanagedType.Bool)] out bool pfEnableTCP
            );

        void SetEnableTCP(
            [In, MarshalAs(UnmanagedType.Bool)] bool fEnableTCP
            );

        void ResetProtocolRollover();

        void GetConnectionBandwidth(
            out int pdwConnectionBandwidth
            );

        void SetConnectionBandwidth(
            [In] int dwConnectionBandwidth
            );

        void GetNumProtocolsSupported(
            out int pcProtocols
            );

        void GetSupportedProtocolName(
            [In] int dwProtocolNum,
            [Out] StringBuilder pwszProtocolName,
            ref int pcchProtocolName
            );

        void AddLoggingUrl(
            [In] string pwszURL
            );

        void GetLoggingUrl(
            [In] int dwIndex,
            [Out] StringBuilder pwszURL,
            ref int pcchURL
            );

        void GetLoggingUrlCount(
            out int pdwUrlCount
            );

        void ResetLoggingUrlList();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("D979A853-042B-4050-8387-C939DB22013F"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderNetworkConfig2 : IWMReaderNetworkConfig
    {
        #region IWMReaderNetworkConfig Methods

        new void GetBufferingTime(
            out long pcnsBufferingTime
            );

        new void SetBufferingTime(
            [In] long cnsBufferingTime
            );

        new void GetUDPPortRanges(
            [Out, MarshalAs(UnmanagedType.LPArray)] WMPortNumberRange[] pRangeArray,
            ref int pcRanges
            );

        new void SetUDPPortRanges(
            [In, MarshalAs(UnmanagedType.LPArray)] WMPortNumberRange[] pRangeArray,
            [In] int cRanges
            );

        new void GetProxySettings(
            [In] string pwszProtocol,
            out ProxySettings pProxySetting
            );

        new void SetProxySettings(
            [In] string pwszProtocol,
            [In] ProxySettings ProxySetting
            );

        new void GetProxyHostName(
            [In] string pwszProtocol,
            [Out] StringBuilder pwszHostName,
            ref int pcchHostName
            );

        new void SetProxyHostName(
            [In] string pwszProtocol,
            [In] string pwszHostName
            );

        new void GetProxyPort(
            [In] string pwszProtocol,
            out int pdwPort
            );

        new void SetProxyPort(
            [In] string pwszProtocol,
            [In] int dwPort
            );

        new void GetProxyExceptionList(
            [In] string pwszProtocol,
            [Out] StringBuilder pwszExceptionList,
            ref int pcchExceptionList
            );

        new void SetProxyExceptionList(
            [In] string pwszProtocol,
            [In] string pwszExceptionList
            );

        new void GetProxyBypassForLocal(
            [In] string pwszProtocol,
            [MarshalAs(UnmanagedType.Bool)] out bool pfBypassForLocal
            );

        new void SetProxyBypassForLocal(
            [In] string pwszProtocol,
            [In, MarshalAs(UnmanagedType.Bool)] bool fBypassForLocal
            );

        new void GetForceRerunAutoProxyDetection(
            [MarshalAs(UnmanagedType.Bool)] out bool pfForceRerunDetection
            );

        new void SetForceRerunAutoProxyDetection(
            [In, MarshalAs(UnmanagedType.Bool)] bool fForceRerunDetection
            );

        new void GetEnableMulticast(
            [MarshalAs(UnmanagedType.Bool)] out bool pfEnableMulticast
            );

        new void SetEnableMulticast(
            [In, MarshalAs(UnmanagedType.Bool)] bool fEnableMulticast
            );

        new void GetEnableHTTP(
            [MarshalAs(UnmanagedType.Bool)] out bool pfEnableHTTP
            );

        new void SetEnableHTTP(
            [In, MarshalAs(UnmanagedType.Bool)] bool fEnableHTTP
            );

        new void GetEnableUDP(
            [MarshalAs(UnmanagedType.Bool)] out bool pfEnableUDP
            );

        new void SetEnableUDP(
            [In, MarshalAs(UnmanagedType.Bool)] bool fEnableUDP
            );

        new void GetEnableTCP(
            [MarshalAs(UnmanagedType.Bool)] out bool pfEnableTCP
            );

        new void SetEnableTCP(
            [In, MarshalAs(UnmanagedType.Bool)] bool fEnableTCP
            );

        new void ResetProtocolRollover();

        new void GetConnectionBandwidth(
            out int pdwConnectionBandwidth
            );

        new void SetConnectionBandwidth(
            [In] int dwConnectionBandwidth
            );

        new void GetNumProtocolsSupported(
            out int pcProtocols
            );

        new void GetSupportedProtocolName(
            [In] int dwProtocolNum,
            [Out] StringBuilder pwszProtocolName,
            ref int pcchProtocolName
            );

        new void AddLoggingUrl(
            [In] string pwszURL
            );

        new void GetLoggingUrl(
            [In] int dwIndex,
            [Out] StringBuilder pwszURL,
            ref int pcchURL
            );

        new void GetLoggingUrlCount(
            out int pdwUrlCount
            );

        new void ResetLoggingUrlList();

        #endregion

        void GetEnableContentCaching(
            [MarshalAs(UnmanagedType.Bool)] out bool pfEnableContentCaching
            );

        void SetEnableContentCaching(
            [In, MarshalAs(UnmanagedType.Bool)] bool fEnableContentCaching
            );

        void GetEnableFastCache(
            [MarshalAs(UnmanagedType.Bool)] out bool pfEnableFastCache
            );

        void SetEnableFastCache(
            [In, MarshalAs(UnmanagedType.Bool)] bool fEnableFastCache
            );

        void GetAcceleratedStreamingDuration(
            out long pcnsAccelDuration
            );

        void SetAcceleratedStreamingDuration(
            [In] long cnsAccelDuration
            );

        void GetAutoReconnectLimit(
            out int pdwAutoReconnectLimit
            );

        void SetAutoReconnectLimit(
            [In] int dwAutoReconnectLimit
            );

        void GetEnableResends(
            [MarshalAs(UnmanagedType.Bool)] out bool pfEnableResends
            );

        void SetEnableResends(
            [In, MarshalAs(UnmanagedType.Bool)] bool fEnableResends
            );

        void GetEnableThinning(
            [MarshalAs(UnmanagedType.Bool)] out bool pfEnableThinning
            );

        void SetEnableThinning(
            [In, MarshalAs(UnmanagedType.Bool)] bool fEnableThinning
            );

        void GetMaxNetPacketSize(
            out int pdwMaxNetPacketSize
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BED-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderStreamClock
    {
        void GetTime(
            out long pcnsNow
            );

        void SetTimer(
            [In] long cnsWhen,
            [In] IntPtr pvParam,
            out int pdwTimerId
            );

        void KillTimer(
            [In] int dwTimerId
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("FDBE5592-81A1-41EA-93BD-735CAD1ADC05"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderTypeNegotiation
    {
        void TryOutputProps(
            [In] int dwOutputNum,
            [In] IWMOutputMediaProps pOutput
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("CF4B1F99-4DE2-4E49-A363-252740D99BC1"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMRegisterCallback
    {
        void Advise(
            [In] IWMStatusCallback pCallback,
            [In] IntPtr pvContext
            );

        void Unadvise(
            [In] IWMStatusCallback pCallback,
            [In] IntPtr pvContext
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("6D7CDC70-9888-11D3-8EDC-00C04F6109CF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMStatusCallback
    {
        void OnStatus(
            [In] Status iStatus,
            [In] int hr,
            [In] AttrDataType dwType,
            [In] IntPtr pValue,
            [In] IntPtr pvContext
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BDC-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMStreamConfig
    {
        void GetStreamType(
            out Guid pguidStreamType
            );

        void GetStreamNumber(
            out short pwStreamNum
            );

        void SetStreamNumber(
            [In] short wStreamNum
            );

        void GetStreamName(
            [Out] StringBuilder pwszStreamName,
            ref short pcchStreamName
            );

        void SetStreamName(
            [In] string pwszStreamName
            );

        void GetConnectionName(
            [Out] StringBuilder pwszInputName,
            ref short pcchInputName
            );

        void SetConnectionName(
            [In] string pwszInputName
            );

        void GetBitrate(
            out int pdwBitrate
            );

        void SetBitrate(
            [In] int pdwBitrate
            );

        void GetBufferWindow(
            out int pmsBufferWindow
            );

        void SetBufferWindow(
            [In] int msBufferWindow
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("7688D8CB-FC0D-43BD-9459-5A8DEC200CFA"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMStreamConfig2 : IWMStreamConfig
    {
        #region IWMStreamConfig Methods

        new void GetStreamType(
            out Guid pguidStreamType
            );

        new void GetStreamNumber(
            out short pwStreamNum
            );

        new void SetStreamNumber(
            [In] short wStreamNum
            );

        new void GetStreamName(
            [Out] StringBuilder pwszStreamName,
            ref short pcchStreamName
            );

        new void SetStreamName(
            [In] string pwszStreamName
            );

        new void GetConnectionName(
            [Out] StringBuilder pwszInputName,
            ref short pcchInputName
            );

        new void SetConnectionName(
            [In] string pwszInputName
            );

        new void GetBitrate(
            out int pdwBitrate
            );

        new void SetBitrate(
            [In] int pdwBitrate
            );

        new void GetBufferWindow(
            out int pmsBufferWindow
            );

        new void SetBufferWindow(
            [In] int msBufferWindow
            );

        #endregion

        void GetTransportType(
            out TransportType pnTransportType
            );

        void SetTransportType(
            [In] TransportType nTransportType
            );

        void AddDataUnitExtension(
            [In] Guid guidExtensionSystemID,
            [In] short cbExtensionDataSize,
            [In, MarshalAs(UnmanagedType.LPArray)] byte[] pbExtensionSystemInfo,
            [In] int cbExtensionSystemInfo
            );

        void GetDataUnitExtensionCount(
            out short pcDataUnitExtensions
            );

        void GetDataUnitExtension(
            [In] short wDataUnitExtensionNumber,
            out Guid pguidExtensionSystemID,
            out short pcbExtensionDataSize,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pbExtensionSystemInfo,
            ref int pcbExtensionSystemInfo
            );

        void RemoveAllDataUnitExtensions();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("CB164104-3AA9-45A7-9AC9-4DAEE131D6E1"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMStreamConfig3 : IWMStreamConfig2
    {
        #region IWMStreamConfig Methods

        new void GetStreamType(
            out Guid pguidStreamType
            );

        new void GetStreamNumber(
            out short pwStreamNum
            );

        new void SetStreamNumber(
            [In] short wStreamNum
            );

        new void GetStreamName(
            [Out] StringBuilder pwszStreamName,
            ref short pcchStreamName
            );

        new void SetStreamName(
            [In] string pwszStreamName
            );

        new void GetConnectionName(
            [Out] StringBuilder pwszInputName,
            ref short pcchInputName
            );

        new void SetConnectionName(
            [In] string pwszInputName
            );

        new void GetBitrate(
            out int pdwBitrate
            );

        new void SetBitrate(
            [In] int pdwBitrate
            );

        new void GetBufferWindow(
            out int pmsBufferWindow
            );

        new void SetBufferWindow(
            [In] int msBufferWindow
            );

        #endregion

        #region IWMStreamConfig2 Methods

        new void GetTransportType(
            out TransportType pnTransportType
            );

        new void SetTransportType(
            [In] TransportType nTransportType
            );

        new void AddDataUnitExtension(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidExtensionSystemID,
            [In] short cbExtensionDataSize,
            [In] byte[] pbExtensionSystemInfo,
            [In] int cbExtensionSystemInfo
            );

        new void GetDataUnitExtensionCount(
            out short pcDataUnitExtensions
            );

        new void GetDataUnitExtension(
            [In] short wDataUnitExtensionNumber,
            out Guid pguidExtensionSystemID,
            out short pcbExtensionDataSize,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pbExtensionSystemInfo,
            ref int pcbExtensionSystemInfo
            );

        new void RemoveAllDataUnitExtensions();

        #endregion

        void GetLanguage(
            [Out] StringBuilder pwszLanguageString,
            ref short pcchLanguageStringLength
            );

        void SetLanguage(
            [In] string pwszLanguageString
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BDD-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMStreamList
    {
        void GetStreams(
            [Out, MarshalAs(UnmanagedType.LPArray)] short[] pwStreamNumArray,
            ref short pcStreams
            );

        void AddStream(
            [In] short wStreamNum
            );

        void RemoveStream(
            [In] short wStreamNum
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("8C1C6090-F9A8-4748-8EC3-DD1108BA1E77"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMStreamPrioritization
    {
        void GetPriorityRecords(
            [Out, MarshalAs(UnmanagedType.LPArray)] WMStreamPrioritizationRecord [] pRecordArray,
            ref short pcRecords
            );

        void SetPriorityRecords(
            [In, MarshalAs(UnmanagedType.LPArray)] WMStreamPrioritizationRecord [] pRecordArray,
            [In] short cRecords
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("9397F121-7705-4DC9-B049-98B698188414"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMSyncReader
    {
        void Open(
            [In] string pwszFilename
            );

        void Close();

        void SetRange(
            [In] long cnsStartTime,
            [In] long cnsDuration
            );

        void SetRangeByFrame(
            [In] short wStreamNum,
            [In] long qwFrameNumber,
            [In] long cFramesToRead
            );

        void GetNextSample(
            [In] short wStreamNum,
            out INSSBuffer ppSample,
            out long pcnsSampleTime,
            out long pcnsDuration,
            out SampleFlag pdwFlags,
            out int pdwOutputNum,
            out short pwStreamNum
            );

        void SetStreamsSelected(
            [In] short cStreamCount,
            [In, MarshalAs(UnmanagedType.LPArray)] short[] pwStreamNumbers,
            [In, MarshalAs(UnmanagedType.LPArray)] StreamSelection[] pSelections
            );

        void GetStreamSelected(
            [In] short wStreamNum,
            out StreamSelection pSelection
            );

        void SetReadStreamSamples(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fCompressed
            );

        void GetReadStreamSamples(
            [In] short wStreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfCompressed
            );

        void GetOutputSetting(
            [In] int dwOutputNum,
            [In] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        void SetOutputSetting(
            [In] int dwOutputNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            [In] short cbLength
            );

        void GetOutputCount(
            out int pcOutputs
            );

        void GetOutputProps(
            [In] int dwOutputNum,
            out IWMOutputMediaProps ppOutput
            );

        void SetOutputProps(
            [In] int dwOutputNum,
            [In] IWMOutputMediaProps pOutput
            );

        void GetOutputFormatCount(
            [In] int dwOutputNum,
            out int pcFormats
            );

        void GetOutputFormat(
            [In] int dwOutputNum,
            [In] int dwFormatNum,
            out IWMOutputMediaProps ppProps
            );

        void GetOutputNumberForStream(
            [In] short wStreamNum,
            out int pdwOutputNum
            );

        void GetStreamNumberForOutput(
            [In] int dwOutputNum,
            out short pwStreamNum
            );

        void GetMaxOutputSampleSize(
            [In] int dwOutput,
            out int pcbMax
            );

        void GetMaxStreamSampleSize(
            [In] short wStream,
            out int pcbMax
            );

        void OpenStream(
            [In] IStream pStream
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("FAED3D21-1B6B-4AF7-8CB6-3E189BBC187B"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMSyncReader2 : IWMSyncReader
    {
        #region IWMSyncReader Methods

        new void Open(
            [In] string pwszFilename
            );

        new void Close();

        new void SetRange(
            [In] long cnsStartTime,
            [In] long cnsDuration
            );

        new void SetRangeByFrame(
            [In] short wStreamNum,
            [In] long qwFrameNumber,
            [In] long cFramesToRead
            );

        new void GetNextSample(
            [In] short wStreamNum,
            out INSSBuffer ppSample,
            out long pcnsSampleTime,
            out long pcnsDuration,
            out SampleFlag pdwFlags,
            out int pdwOutputNum,
            out short pwStreamNum
            );

        new void SetStreamsSelected(
            [In] short cStreamCount,
            [In] short[] pwStreamNumbers,
            [In] StreamSelection[] pSelections
            );

        new void GetStreamSelected(
            [In] short wStreamNum,
            out StreamSelection pSelection
            );

        new void SetReadStreamSamples(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fCompressed
            );

        new void GetReadStreamSamples(
            [In] short wStreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfCompressed
            );

        new void GetOutputSetting(
            [In] int dwOutputNum,
            [In] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        new void SetOutputSetting(
            [In] int dwOutputNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In] byte[] pValue,
            [In] short cbLength
            );

        new void GetOutputCount(
            out int pcOutputs
            );

        new void GetOutputProps(
            [In] int dwOutputNum,
            out IWMOutputMediaProps ppOutput
            );

        new void SetOutputProps(
            [In] int dwOutputNum,
            [In] IWMOutputMediaProps pOutput
            );

        new void GetOutputFormatCount(
            [In] int dwOutputNum,
            out int pcFormats
            );

        new void GetOutputFormat(
            [In] int dwOutputNum,
            [In] int dwFormatNum,
            out IWMOutputMediaProps ppProps
            );

        new void GetOutputNumberForStream(
            [In] short wStreamNum,
            out int pdwOutputNum
            );

        new void GetStreamNumberForOutput(
            [In] int dwOutputNum,
            out short pwStreamNum
            );

        new void GetMaxOutputSampleSize(
            [In] int dwOutput,
            out int pcbMax
            );

        new void GetMaxStreamSampleSize(
            [In] short wStream,
            out int pcbMax
            );

        new void OpenStream(
            [In] IStream pStream
            );

        #endregion

        void SetRangeByTimecode(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.LPStruct)] TimeCodeExtensionData pStart,
            [In, MarshalAs(UnmanagedType.LPStruct)] TimeCodeExtensionData pEnd
            );

        void SetRangeByFrameEx(
            [In] short wStreamNum,
            [In] long qwFrameNumber,
            [In] long cFramesToRead,
            out long pcnsStartTime
            );

        void SetAllocateForOutput(
            [In] int dwOutputNum,
            [In] IWMReaderAllocatorEx pAllocator
            );

        void GetAllocateForOutput(
            [In] int dwOutputNum,
            out IWMReaderAllocatorEx ppAllocator
            );

        void SetAllocateForStream(
            [In] short wStreamNum,
            [In] IWMReaderAllocatorEx pAllocator
            );

        void GetAllocateForStream(
            [In] short dwSreamNum,
            out IWMReaderAllocatorEx ppAllocator
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BCF-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMVideoMediaProps : IWMMediaProps
    {
        #region IWMMediaProps Methods

        new void GetType(
            out Guid pguidType
            );

        new void GetMediaType(
            [In, Out, MarshalAs(UnmanagedType.CustomMarshaler, MarshalTypeRef = typeof(MTMarshaler))] AMMediaType pType,
            ref int pcbType
            );

        new void SetMediaType(
            [In, MarshalAs(UnmanagedType.LPStruct)] AMMediaType pType
            );

        #endregion

        void GetMaxKeyFrameSpacing(
            out long pllTime
            );

        void SetMaxKeyFrameSpacing(
            [In] long llTime
            );

        void GetQuality(
            out int pdwQuality
            );

        void SetQuality(
            [In] int dwQuality
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BD4-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMWriter
    {
        void SetProfileByID(
            [In, MarshalAs(UnmanagedType.LPStruct)] Guid guidProfile
            );

        void SetProfile(
            [In] IWMProfile pProfile
            );

        void SetOutputFilename(
            [In] string pwszFilename
            );

        void GetInputCount(
            out int pcInputs
            );

        void GetInputProps(
            [In] int dwInputNum,
            out IWMInputMediaProps ppInput
            );

        void SetInputProps(
            [In] int dwInputNum,
            [In] IWMInputMediaProps pInput
            );

        void GetInputFormatCount(
            [In] int dwInputNumber,
            out int pcFormats
            );

        void GetInputFormat(
            [In] int dwInputNumber,
            [In] int dwFormatNumber,
            out IWMInputMediaProps pProps
            );

        void BeginWriting();

        void EndWriting();

        void AllocateSample(
            [In] int dwSampleSize,
            out INSSBuffer ppSample
            );

        void WriteSample(
            [In] int dwInputNum,
            [In] long cnsSampleTime,
            [In] SampleFlag dwFlags,
            [In] INSSBuffer pSample
            );

        void Flush();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BE3-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMWriterAdvanced
    {
        void GetSinkCount(
            out int pcSinks
            );

        void GetSink(
            [In] int dwSinkNum,
            out IWMWriterSink ppSink
            );

        void AddSink(
            [In] IWMWriterSink pSink
            );

        void RemoveSink(
            [In] IWMWriterSink pSink
            );

        void WriteStreamSample(
            [In] short wStreamNum,
            [In] long cnsSampleTime,
            [In] int msSampleSendTime,
            [In] long cnsSampleDuration,
            [In] SampleFlag dwFlags,
            [In] INSSBuffer pSample
            );

        void SetLiveSource(
            [In, MarshalAs(UnmanagedType.Bool)] bool fIsLiveSource
            );

        void IsRealTime(
            [MarshalAs(UnmanagedType.Bool)] out bool pfRealTime
            );

        void GetWriterTime(
            out long pcnsCurrentTime
            );

        void GetStatistics(
            [In] short wStreamNum,
            out WriterStatistics pStats
            );

        void SetSyncTolerance(
            [In] int msWindow
            );

        void GetSyncTolerance(
            out int pmsWindow
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("962DC1EC-C046-4DB8-9CC7-26CEAE500817"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMWriterAdvanced2 : IWMWriterAdvanced
    {
        #region IWMWriterAdvanced Methods

        new void GetSinkCount(
            out int pcSinks
            );

        new void GetSink(
            [In] int dwSinkNum,
            out IWMWriterSink ppSink
            );

        new void AddSink(
            [In] IWMWriterSink pSink
            );

        new void RemoveSink(
            [In] IWMWriterSink pSink
            );

        new void WriteStreamSample(
            [In] short wStreamNum,
            [In] long cnsSampleTime,
            [In] int msSampleSendTime,
            [In] long cnsSampleDuration,
            [In] SampleFlag dwFlags,
            [In] INSSBuffer pSample
            );

        new void SetLiveSource(
            [In, MarshalAs(UnmanagedType.Bool)] bool fIsLiveSource
            );

        new void IsRealTime(
            [MarshalAs(UnmanagedType.Bool)] out bool pfRealTime
            );

        new void GetWriterTime(
            out long pcnsCurrentTime
            );

        new void GetStatistics(
            [In] short wStreamNum,
            out WriterStatistics pStats
            );

        new void SetSyncTolerance(
            [In] int msWindow
            );

        new void GetSyncTolerance(
            out int pmsWindow
            );

        #endregion

        void GetInputSetting(
            [In] int dwInputNum,
            [In] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        void SetInputSetting(
            [In] int dwInputNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            [In] short cbLength
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("2CD6492D-7C37-4E76-9D3B-59261183A22E"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMWriterAdvanced3 : IWMWriterAdvanced2
    {
        #region IWMWriterAdvanced Methods

        new void GetSinkCount(
            out int pcSinks
            );

        new void GetSink(
            [In] int dwSinkNum,
            out IWMWriterSink ppSink
            );

        new void AddSink(
            [In] IWMWriterSink pSink
            );

        new void RemoveSink(
            [In] IWMWriterSink pSink
            );

        new void WriteStreamSample(
            [In] short wStreamNum,
            [In] long cnsSampleTime,
            [In] int msSampleSendTime,
            [In] long cnsSampleDuration,
            [In] SampleFlag dwFlags,
            [In] INSSBuffer pSample
            );

        new void SetLiveSource(
            [In, MarshalAs(UnmanagedType.Bool)] bool fIsLiveSource
            );

        new void IsRealTime(
            [MarshalAs(UnmanagedType.Bool)] out bool pfRealTime
            );

        new void GetWriterTime(
            out long pcnsCurrentTime
            );

        new void GetStatistics(
            [In] short wStreamNum,
            out WriterStatistics pStats
            );

        new void SetSyncTolerance(
            [In] int msWindow
            );

        new void GetSyncTolerance(
            out int pmsWindow
            );

        new void GetInputSetting(
            [In] int dwInputNum,
            [In] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        #endregion

        #region IWMWriterAdvanced2 Methods

        new void SetInputSetting(
            [In] int dwInputNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In] byte[] pValue,
            [In] short cbLength
            );

        #endregion

        void GetStatisticsEx(
            [In] short wStreamNum,
            out WMWriterStatisticsEx pStats
            );

        void SetNonBlocking();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BE5-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMWriterFileSink : IWMWriterSink
    {
        #region IWMWriterSink Methods

        new void OnHeader(
            [In] INSSBuffer pHeader
            );

        new void IsRealTime(
            [MarshalAs(UnmanagedType.Bool)] out bool pfRealTime
            );

        new void AllocateDataUnit(
            [In] int cbDataUnit,
            out INSSBuffer ppDataUnit
            );

        new void OnDataUnit(
            [In] INSSBuffer pDataUnit
            );

        new void OnEndWriting();

        #endregion

        void Open(
            [In] string pwszFilename
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("14282BA7-4AEF-4205-8CE5-C229035A05BC"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMWriterFileSink2 : IWMWriterFileSink
    {
        #region IWMWriterSink Methods

        new void OnHeader(
            [In] INSSBuffer pHeader
            );

        new void IsRealTime(
            [MarshalAs(UnmanagedType.Bool)] out bool pfRealTime
            );

        new void AllocateDataUnit(
            [In] int cbDataUnit,
            out INSSBuffer ppDataUnit
            );

        new void OnDataUnit(
            [In] INSSBuffer pDataUnit
            );

        new void OnEndWriting();

        #endregion

        #region IWMWriterFileSink Methods

        new void Open(
            [In] string pwszFilename
            );

        #endregion

        void Start(
            [In] long cnsStartTime
            );

        void Stop(
            [In] long cnsStopTime
            );

        void IsStopped(
            [MarshalAs(UnmanagedType.Bool)] out bool pfStopped
            );

        void GetFileDuration(
            out long pcnsDuration
            );

        void GetFileSize(
            out long pcbFile
            );

        void Close();

        void IsClosed(
            [MarshalAs(UnmanagedType.Bool)] out bool pfClosed
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("3FEA4FEB-2945-47A7-A1DD-C53A8FC4C45C"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMWriterFileSink3 : IWMWriterFileSink2
    {
        #region IWMWriterSink Methods

        new void OnHeader(
            [In] INSSBuffer pHeader
            );

        new void IsRealTime(
            [MarshalAs(UnmanagedType.Bool)] out bool pfRealTime
            );

        new void AllocateDataUnit(
            [In] int cbDataUnit,
            out INSSBuffer ppDataUnit
            );

        new void OnDataUnit(
            [In] INSSBuffer pDataUnit
            );

        new void OnEndWriting();

        #endregion

        #region IWMWriterFileSink Methods

        new void Open(
            [In] string pwszFilename
            );

        #endregion

        #region IWMWriterFileSink2 Methods

        new void Start(
            [In] long cnsStartTime
            );

        new void Stop(
            [In] long cnsStopTime
            );

        new void IsStopped(
            [MarshalAs(UnmanagedType.Bool)] out bool pfStopped
            );

        new void GetFileDuration(
            out long pcnsDuration
            );

        new void GetFileSize(
            out long pcbFile
            );

        new void Close();

        new void IsClosed(
            [MarshalAs(UnmanagedType.Bool)] out bool pfClosed
            );

        #endregion

        void SetAutoIndexing(
            [In, MarshalAs(UnmanagedType.Bool)] bool fDoAutoIndexing
            );

        void GetAutoIndexing(
            [MarshalAs(UnmanagedType.Bool)] out bool pfAutoIndexing
            );

        void SetControlStream(
            [In] short wStreamNumber,
            [In, MarshalAs(UnmanagedType.Bool)] bool fShouldControlStartAndStop
            );

        void GetMode(
            out FileSinkMode pdwFileSinkMode
            );

        void OnDataUnitEx(
            [In, MarshalAs(UnmanagedType.LPStruct)] FileSinkDataUnit pFileSinkDataUnit
            );

        void SetUnbufferedIO(
            [In, MarshalAs(UnmanagedType.Bool)] bool fUnbufferedIO,
            [In, MarshalAs(UnmanagedType.Bool)] bool fRestrictMemUsage
            );

        void GetUnbufferedIO(
            [MarshalAs(UnmanagedType.Bool)] out bool pfUnbufferedIO
            );

        void CompleteOperations();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BE7-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMWriterNetworkSink : IWMWriterSink
    {
        #region IWMWriterSink Methods

        new void OnHeader(
            [In] INSSBuffer pHeader
            );

        new void IsRealTime(
            [MarshalAs(UnmanagedType.Bool)] out bool pfRealTime
            );

        new void AllocateDataUnit(
            [In] int cbDataUnit,
            out INSSBuffer ppDataUnit
            );

        new void OnDataUnit(
            [In] INSSBuffer pDataUnit
            );

        new void OnEndWriting();

        #endregion

        void SetMaximumClients(
            [In] int dwMaxClients
            );

        void GetMaximumClients(
            out int pdwMaxClients
            );

        void SetNetworkProtocol(
            [In] NetProtocol protocol
            );

        void GetNetworkProtocol(
            out NetProtocol pProtocol
            );

        void GetHostURL(
            [Out] StringBuilder pwszURL,
            ref int pcchURL
            );

        void Open(
            ref int pdwPortNum
            );

        void Disconnect();

        void Close();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("81E20CE4-75EF-491A-8004-FC53C45BDC3E"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMWriterPostView
    {
        void SetPostViewCallback(
            IWMWriterPostViewCallback pCallback,
            IntPtr pvContext
            );

        void SetReceivePostViewSamples(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fReceivePostViewSamples
            );

        void GetReceivePostViewSamples(
            [In] short wStreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfReceivePostViewSamples
            );

        void GetPostViewProps(
            [In] short wStreamNumber,
            out IWMMediaProps ppOutput
            );

        void SetPostViewProps(
            [In] short wStreamNumber,
            [In] IWMMediaProps pOutput
            );

        void GetPostViewFormatCount(
            [In] short wStreamNumber,
            out int pcFormats
            );

        void GetPostViewFormat(
            [In] short wStreamNumber,
            [In] int dwFormatNumber,
            out IWMMediaProps ppProps
            );

        void SetAllocateForPostView(
            [In] short wStreamNumber,
            [In, MarshalAs(UnmanagedType.Bool)] bool fAllocate
            );

        void GetAllocateForPostView(
            [In] short wStreamNumber,
            [MarshalAs(UnmanagedType.Bool)] out bool pfAllocate
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("D9D6549D-A193-4F24-B308-03123D9B7F8D"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMWriterPostViewCallback : IWMStatusCallback
    {
        #region IWMStatusCallback Methods

        new void OnStatus(
            [In] Status iStatus,
            [In] int hr,
            [In] AttrDataType dwType,
            [In] IntPtr pValue,
            [In] IntPtr pvContext
            );

        #endregion

        void OnPostViewSample(
            [In] short wStreamNumber,
            [In] long cnsSampleTime,
            [In] long cnsSampleDuration,
            [In] SampleFlag dwFlags,
            [In] INSSBuffer pSample,
            [In] IntPtr pvContext
            );

        void AllocateForPostView(
            [In] short wStreamNum,
            [In] int cbBuffer,
            out INSSBuffer ppBuffer,
            [In] IntPtr pvContext
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("FC54A285-38C4-45B5-AA23-85B9F7CB424B"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMWriterPreprocess
    {
        void GetMaxPreprocessingPasses(
            [In] int dwInputNum,
            [In] int dwFlags,
            out int pdwMaxNumPasses
            );

        void SetNumPreprocessingPasses(
            [In] int dwInputNum,
            [In] int dwFlags,
            [In] int dwNumPasses
            );

        void BeginPreprocessingPass(
            [In] int dwInputNum,
            [In] int dwFlags
            );

        void PreprocessSample(
            [In] int dwInputNum,
            [In] long cnsSampleTime,
            [In] int dwFlags,
            [In] INSSBuffer pSample
            );

        void EndPreprocessingPass(
            [In] int dwInputNum,
            [In] int dwFlags
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("DC10E6A5-072C-467D-BF57-6330A9DDE12A"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMWriterPushSink : IWMWriterSink
    {
        #region IWMWriterSink Methods

        new void OnHeader(
            [In] INSSBuffer pHeader
            );

        new void IsRealTime(
            [MarshalAs(UnmanagedType.Bool)] out bool pfRealTime
            );

        new void AllocateDataUnit(
            [In] int cbDataUnit,
            out INSSBuffer ppDataUnit
            );

        new void OnDataUnit(
            [In] INSSBuffer pDataUnit
            );

        new void OnEndWriting();

        #endregion

        void Connect(
            [In] string pwszURL,
            [In] string pwszTemplateURL,
            [In, MarshalAs(UnmanagedType.Bool)] bool fAutoDestroy
            );

        void Disconnect();

        void EndSession();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("96406BE4-2B2B-11D3-B36B-00C04F6108FF"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMWriterSink
    {
        void OnHeader(
            [In] INSSBuffer pHeader
            );

        void IsRealTime(
            [MarshalAs(UnmanagedType.Bool)] out bool pfRealTime
            );

        void AllocateDataUnit(
            [In] int cbDataUnit,
            out INSSBuffer ppDataUnit
            );

        void OnDataUnit(
            [In] INSSBuffer pDataUnit
            );

        void OnEndWriting();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("0C0E4080-9081-11D2-BEEC-0060082F2054"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface INSNetSourceCreator
    {
        void Initialize();

        void CreateNetSource(
            [In] string pszStreamName,
            [In, MarshalAs(UnmanagedType.IUnknown)] object pMonitor,
            [In] byte[] pData,
            [In, MarshalAs(UnmanagedType.IUnknown)] object pUserContext,
            [In, MarshalAs(UnmanagedType.IUnknown)] object pCallback,
            [In] long qwContext
            );

        void GetNetSourceProperties(
            [In] string pszStreamName,
            [MarshalAs(UnmanagedType.IUnknown)] out object ppPropertiesNode
            );

        void GetNetSourceSharedNamespace(
            [MarshalAs(UnmanagedType.IUnknown)] out object ppSharedNamespace
            );

        void GetNetSourceAdminInterface(
            [In] string pszStreamName,
            [MarshalAs(UnmanagedType.Struct)] out object pVal
            );

        void GetNumProtocolsSupported(
            out int pcProtocols
            );

        void GetProtocolName(
            [In] int dwProtocolNum,
            [Out] StringBuilder pwszProtocolName,
            ref short pcchProtocolName
            );

        void Shutdown();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("4F528693-1035-43FE-B428-757561AD3A68"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface INSSBuffer2 : INSSBuffer
    {
        #region INSSBuffer Methods

        new void GetLength(
            out int pdwLength
            );

        new void SetLength(
            [In] int dwLength
            );

        new void GetMaxLength(
            out int pdwLength
            );

        new void GetBuffer(
            out IntPtr ppdwBuffer
            );

        new void GetBufferAndLength(
            out IntPtr ppdwBuffer,
            out int pdwLength
            );

        #endregion

        void GetSampleProperties(
            [In] int cbProperties,
            out byte[] pbProperties
            );

        void SetSampleProperties(
            [In] int cbProperties,
            [In] byte[] pbProperties
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("C87CEAAF-75BE-4BC4-84EB-AC2798507672"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface INSSBuffer3 : INSSBuffer2
    {
        #region INSSBuffer Methods

        new void GetLength(
            out int pdwLength
            );

        new void SetLength(
            [In] int dwLength
            );

        new void GetMaxLength(
            out int pdwLength
            );

        new void GetBuffer(
            out IntPtr ppdwBuffer
            );

        new void GetBufferAndLength(
            out IntPtr ppdwBuffer,
            out int pdwLength
            );

        #endregion

        #region INSSBuffer2 Methods

        new void GetSampleProperties(
            [In] int cbProperties,
            out byte[] pbProperties
            );

        new void SetSampleProperties(
            [In] int cbProperties,
            [In] byte[] pbProperties
            );

        #endregion

        void SetProperty(
            [In] Guid WM_SampleExtensionGUID,
            [In] IntPtr pvBufferProperty,
            [In] int dwBufferPropertySize
            );

        void GetProperty(
            [In] Guid WM_SampleExtensionGUID,
            [Out] IntPtr pvBufferProperty,
            ref int pdwBufferPropertySize
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("B6B8FD5A-32E2-49D4-A910-C26CC85465ED"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface INSSBuffer4 : INSSBuffer3
    {
        #region INSSBuffer Methods

        new void GetLength(
            out int pdwLength
            );

        new void SetLength(
            [In] int dwLength
            );

        new void GetMaxLength(
            out int pdwLength
            );

        new void GetBuffer(
            out IntPtr ppdwBuffer
            );

        new void GetBufferAndLength(
            out IntPtr ppdwBuffer,
            out int pdwLength
            );

        #endregion

        #region INSSBuffer2 Methods

        new void GetSampleProperties(
            [In] int cbProperties,
            out byte[] pbProperties
            );

        new void SetSampleProperties(
            [In] int cbProperties,
            [In] byte[] pbProperties
            );

        #endregion

        #region INSSBuffer3 Methods

        new void SetProperty(
            [In] Guid guidBufferProperty,
            [In] IntPtr pvBufferProperty,
            [In] int dwBufferPropertySize
            );

        new void GetProperty(
            [In] Guid guidBufferProperty,
            [Out] IntPtr pvBufferProperty,
            ref int pdwBufferPropertySize
            );

        #endregion

        void GetPropertyCount(
            out int pcBufferProperties
            );

        void GetPropertyByIndex(
            [In] int dwBufferPropertyIndex,
            out Guid pguidBufferProperty,
            [Out] IntPtr pvBufferProperty,
            ref int pdwBufferPropertySize
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("8BB23E5F-D127-4AFB-8D02-AE5B66D54C78"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMSInternalAdminNetSource
    {
        void Initialize(
            [In, MarshalAs(UnmanagedType.IUnknown)] object pSharedNamespace,
            [In, MarshalAs(UnmanagedType.IUnknown)] object pNamespaceNode,
            [In] INSNetSourceCreator pNetSourceCreator,
            [In] int fEmbeddedInServer
            );

        void GetNetSourceCreator(
            out INSNetSourceCreator ppNetSourceCreator
            );

        void SetCredentials(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrRealm,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrName,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrPassword,
            [In, MarshalAs(UnmanagedType.Bool)] bool fPersist,
            [In, MarshalAs(UnmanagedType.Bool)] bool fConfirmedGood
            );

        void GetCredentials(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrRealm,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrName,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrPassword,
            [MarshalAs(UnmanagedType.Bool)] out bool pfConfirmedGood
            );

        void DeleteCredentials(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrRealm
            );

        void GetCredentialFlags(
            out CredentialFlag lpdwFlags
            );

        void SetCredentialFlags(
            [In] CredentialFlag dwFlags
            );

        [PreserveSig]
        int FindProxyForURL(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrProtocol,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrHost,
            [MarshalAs(UnmanagedType.Bool)] out bool pfProxyEnabled,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrProxyServer,
            out int pdwProxyPort,
            ref int pdwProxyContext
            );

        void RegisterProxyFailure(
            [In] int hrParam,
            [In] int dwProxyContext
            );

        void ShutdownProxyContext(
            [In] int dwProxyContext
            );

        void IsUsingIE(
            [In] int dwProxyContext,
            [MarshalAs(UnmanagedType.Bool)] out bool pfIsUsingIE
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("E74D58C3-CF77-4B51-AF17-744687C43EAE"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMSInternalAdminNetSource2
    {
        void SetCredentialsEx(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrRealm,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrUrl,
            [In, MarshalAs(UnmanagedType.Bool)] bool fProxy,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrName,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrPassword,
            [In, MarshalAs(UnmanagedType.Bool)] bool fPersist,
            [In, MarshalAs(UnmanagedType.Bool)] bool fConfirmedGood
            );

        void GetCredentialsEx(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrRealm,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrUrl,
            [In, MarshalAs(UnmanagedType.Bool)] bool fProxy,
            out NetSourceURLCredPolicySettings pdwUrlPolicy,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrName,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrPassword,
            [MarshalAs(UnmanagedType.Bool)] out bool pfConfirmedGood
            );

        void DeleteCredentialsEx(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrRealm,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrUrl,
            [In, MarshalAs(UnmanagedType.Bool)] bool fProxy
            );

        void FindProxyForURLEx(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrProtocol,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrHost,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrUrl,
            [MarshalAs(UnmanagedType.Bool)] out bool pfProxyEnabled,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrProxyServer,
            out int pdwProxyPort,
            ref int pdwProxyContext
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("6B63D08E-4590-44AF-9EB3-57FF1E73BF80"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMSInternalAdminNetSource3 : IWMSInternalAdminNetSource2
    {
        #region IWMSInternalAdminNetSource2 Methods

        new void SetCredentialsEx(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrRealm,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrUrl,
            [In, MarshalAs(UnmanagedType.Bool)] bool fProxy,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrName,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrPassword,
            [In, MarshalAs(UnmanagedType.Bool)] bool fPersist,
            [In, MarshalAs(UnmanagedType.Bool)] bool fConfirmedGood
            );

        new void GetCredentialsEx(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrRealm,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrUrl,
            [In, MarshalAs(UnmanagedType.Bool)] bool fProxy,
            out NetSourceURLCredPolicySettings pdwUrlPolicy,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrName,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrPassword,
            [MarshalAs(UnmanagedType.Bool)] out bool pfConfirmedGood
            );

        new void DeleteCredentialsEx(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrRealm,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrUrl,
            [In, MarshalAs(UnmanagedType.Bool)] bool fProxy
            );

        new void FindProxyForURLEx(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrProtocol,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrHost,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrUrl,
            [MarshalAs(UnmanagedType.Bool)] out bool pfProxyEnabled,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrProxyServer,
            out int pdwProxyPort,
            ref int pdwProxyContext
            );

        #endregion

        void GetNetSourceCreator2(
            [MarshalAs(UnmanagedType.IUnknown)] out object ppNetSourceCreator
            );

        [PreserveSig]
        int FindProxyForURLEx2(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrProtocol,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrHost,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrUrl,
            [MarshalAs(UnmanagedType.Bool)] out bool pfProxyEnabled,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrProxyServer,
            out int pdwProxyPort,
            ref long pqwProxyContext
            );

        void RegisterProxyFailure2(
            [In] int hrParam,
            [In] long qwProxyContext
            );

        void ShutdownProxyContext2(
            [In] long qwProxyContext
            );

        void IsUsingIE2(
            [In] long qwProxyContext,
            [MarshalAs(UnmanagedType.Bool)] out bool pfIsUsingIE
            );

        void SetCredentialsEx2(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrRealm,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrUrl,
            [In, MarshalAs(UnmanagedType.Bool)] bool fProxy,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrName,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrPassword,
            [In, MarshalAs(UnmanagedType.Bool)] bool fPersist,
            [In, MarshalAs(UnmanagedType.Bool)] bool fConfirmedGood,
            [In, MarshalAs(UnmanagedType.Bool)] bool fClearTextAuthentication
            );

        void GetCredentialsEx2(
            [In, MarshalAs(UnmanagedType.BStr)] string bstrRealm,
            [In, MarshalAs(UnmanagedType.BStr)] string bstrUrl,
            [In, MarshalAs(UnmanagedType.Bool)] bool fProxy,
            [In, MarshalAs(UnmanagedType.Bool)] bool fClearTextAuthentication,
            out NetSourceURLCredPolicySettings pdwUrlPolicy,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrName,
            [MarshalAs(UnmanagedType.BStr)] out string pbstrPassword,
            [MarshalAs(UnmanagedType.Bool)] out bool pfConfirmedGood
            );
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("e5b7ca9a-0f1c-4f66-9002-74ec50d8b304"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMPlayerHook
    {
        void PreDecode();
    }

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("24C44DB0-55D1-49ae-A5CC-F13815E36363"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderAdvanced5 : IWMReaderAdvanced4
    {
        #region IWMReaderAdvanced Methods

        new void SetUserProvidedClock(
            [In, MarshalAs(UnmanagedType.Bool)] bool fUserClock
            );

        new void GetUserProvidedClock(
            [MarshalAs(UnmanagedType.Bool)] out bool pfUserClock
            );

        new void DeliverTime(
            [In] long cnsTime
            );

        new void SetManualStreamSelection(
            [In, MarshalAs(UnmanagedType.Bool)] bool fSelection
            );

        new void GetManualStreamSelection(
            [MarshalAs(UnmanagedType.Bool)] out bool pfSelection
            );

        new void SetStreamsSelected(
            [In] short cStreamCount,
            [In] short[] pwStreamNumbers,
            [In] StreamSelection[] pSelections
            );

        new void GetStreamSelected(
            [In] short wStreamNum,
            out StreamSelection pSelection
            );

        new void SetReceiveSelectionCallbacks(
            [In, MarshalAs(UnmanagedType.Bool)] bool fGetCallbacks
            );

        new void GetReceiveSelectionCallbacks(
            [MarshalAs(UnmanagedType.Bool)] out bool pfGetCallbacks
            );

        new void SetReceiveStreamSamples(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fReceiveStreamSamples
            );

        new void GetReceiveStreamSamples(
            [In] short wStreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfReceiveStreamSamples
            );

        new void SetAllocateForOutput(
            [In] int dwOutputNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fAllocate
            );

        new void GetAllocateForOutput(
            [In] int dwOutputNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfAllocate
            );

        new void SetAllocateForStream(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.Bool)] bool fAllocate
            );

        new void GetAllocateForStream(
            [In] short dwSreamNum,
            [MarshalAs(UnmanagedType.Bool)] out bool pfAllocate
            );

        new void GetStatistics(
            [In, Out, MarshalAs(UnmanagedType.LPStruct)] WMReaderStatistics pStatistics
            );

        new void SetClientInfo(
            [In, MarshalAs(UnmanagedType.LPStruct)] WMReaderClientInfo pClientInfo
            );

        new void GetMaxOutputSampleSize(
            [In] int dwOutput,
            out int pcbMax
            );

        new void GetMaxStreamSampleSize(
            [In] short wStream,
            out int pcbMax
            );

        new void NotifyLateDelivery(
            long cnsLateness
            );

        #endregion

        #region IWMReaderAdvanced2 Methods

        new void SetPlayMode(
            [In] PlayMode Mode
            );

        new void GetPlayMode(
            out PlayMode pMode
            );

        new void GetBufferProgress(
            out int pdwPercent,
            out long pcnsBuffering
            );

        new void GetDownloadProgress(
            out int pdwPercent,
            out long pqwBytesDownloaded,
            out long pcnsDownload
            );

        new void GetSaveAsProgress(
            out int pdwPercent
            );

        new void SaveFileAs(
            [In] string pwszFilename
            );

        new void GetProtocolName(
            [Out] StringBuilder pwszProtocol,
            ref int pcchProtocol
            );

        new void StartAtMarker(
            [In] short wMarkerIndex,
            [In] long cnsDuration,
            [In] float fRate,
            [In] IntPtr pvContext
            );

        new void GetOutputSetting(
            [In] int dwOutputNum,
            [In] string pszName,
            out AttrDataType pType,
            [Out, MarshalAs(UnmanagedType.LPArray)] byte[] pValue,
            ref short pcbLength
            );

        new void SetOutputSetting(
            [In] int dwOutputNum,
            [In] string pszName,
            [In] AttrDataType Type,
            [In] byte[] pValue,
            [In] short cbLength
            );

        new void Preroll(
            [In] long cnsStart,
            [In] long cnsDuration,
            [In] float fRate
            );

        new void SetLogClientID(
            [In, MarshalAs(UnmanagedType.Bool)] bool fLogClientID
            );

        new void GetLogClientID(
            [MarshalAs(UnmanagedType.Bool)] out bool pfLogClientID
            );

        new void StopBuffering();

        new void OpenStream(
            [In] IStream pStream,
            [In] IWMReaderCallback pCallback,
            [In] IntPtr pvContext
            );

        #endregion

        #region IWMReaderAdvanced3 Methods

        new void StopNetStreaming();

        new void StartAtPosition(
            [In] short wStreamNum,
            [In, MarshalAs(UnmanagedType.LPStruct)] RA3Union pvOffsetStart,
            [In, MarshalAs(UnmanagedType.LPStruct)] RA3Union pvDuration,
            [In] OffsetFormat dwOffsetFormat,
            [In] float fRate,
            [In] IntPtr pvContext
            );

        #endregion

        #region IWMReaderAdvanced4 Methods

        new void GetLanguageCount(
            [In] int dwOutputNum,
            out short pwLanguageCount
            );

        new void GetLanguage(
            [In] int dwOutputNum,
            [In] short wLanguage,
            [Out] StringBuilder pwszLanguageString,
            ref short pcchLanguageStringLength
            );

        new void GetMaxSpeedFactor(
            out double pdblFactor
            );

        new void IsUsingFastCache(
            [MarshalAs(UnmanagedType.Bool)] out bool pfUsingFastCache
            );

        new void AddLogParam(
            [In] string wszNameSpace,
            [In] string wszName,
            [In] string wszValue
            );

        new void SendLogParams();

        new void CanSaveFileAs(
            [MarshalAs(UnmanagedType.Bool)] out bool pfCanSave
            );

        new void CancelSaveFileAs();

        new void GetURL(
            [Out] StringBuilder pwszURL,
            ref int pcchURL
            );

        #endregion

        void SetPlayerHook(
            int dwOutputNum,
            IWMPlayerHook pHook);
    };

    [ComImport, SuppressUnmanagedCodeSecurity,
    Guid("f28c0300-9baa-4477-a846-1744d9cbf533"),
    InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    public interface IWMReaderPlaylistBurn
    {
        void InitPlaylistBurn(
            int cFiles,
            [In, MarshalAs(UnmanagedType.LPArray, ArraySubType=UnmanagedType.LPWStr)] string[] ppwszFilenames,
            IWMStatusCallback pCallback,
            IntPtr pvContext);

        void GetInitResults(
            int cFiles,
            [Out, MarshalAs(UnmanagedType.LPArray)] int[] phrStati);

        void Cancel();

        void EndPlaylistBurn(
            int hrBurnResult);
    };

    #endregion
}
