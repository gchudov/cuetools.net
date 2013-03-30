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
using System.Diagnostics;

using WindowsMediaLib.Defs;

namespace WindowsMediaLib
{
    static public class WMUtils
    {
        /// <summary>
        ///  Free the nested structures and release any
        ///  COM objects within an WMMediaType struct.
        /// </summary>
        public static void FreeWMMediaType(AMMediaType mediaType)
        {
            if (mediaType != null)
            {
                if (mediaType.formatSize != 0)
                {
                    Marshal.FreeCoTaskMem(mediaType.formatPtr);
                    mediaType.formatSize = 0;
                    mediaType.formatPtr = IntPtr.Zero;
                }
                if (mediaType.unkPtr != IntPtr.Zero)
                {
                    Marshal.Release(mediaType.unkPtr);
                    mediaType.unkPtr = IntPtr.Zero;
                }
            }
        }

        [DllImport("WMVCore.dll", PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern void WMValidateData(
            byte[] pbData,
            ref int pdwDataSize
            );

        [DllImport("WMVCore.dll", ExactSpelling = true, CharSet = CharSet.Unicode, PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern int WMCheckURLExtension(
            string pwszURL
            );

        [DllImport("WMVCore.dll", ExactSpelling = true, CharSet = CharSet.Unicode, PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern void WMCheckURLScheme(
            string pwszURLScheme
            );

        [DllImport("WMVCore.dll", ExactSpelling = true, CharSet = CharSet.Unicode, PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern void WMIsAvailableOffline(
            string pwszURL,
            string pwszLanguage,
            [MarshalAs(UnmanagedType.Bool)] out bool pfIsAvailableOffline
            );

        [DllImport("WMVCore.dll", PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern void WMCreateEditor(
            out IWMMetadataEditor ppMetadataEditor
            );

        [DllImport("WMVCore.dll", PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern void WMCreateWriterNetworkSink(
            out IWMWriterNetworkSink ppSink
            );

        [DllImport("WMVCore.dll", PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern void WMCreateWriter(
            IntPtr pUnkCert,
            out IWMWriter ppWriter
            );

        [DllImport("WMVCore.dll", PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern void WMCreateProfileManager(
            out IWMProfileManager ppProfileManager
            );

        [DllImport("WMVCore.dll", ExactSpelling = true, CharSet = CharSet.Unicode, PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern void WMIsContentProtected(
            string pwszFileName,
            [MarshalAs(UnmanagedType.Bool)] out bool pfIsProtected
            );

        [DllImport("WMVCore.dll", PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern void WMCreateReader(
            IntPtr pUnkCert,
            Rights dwRights,
            out IWMReader ppReader
            );

        [DllImport("WMVCore.dll", PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern void WMCreateSyncReader(
            IntPtr pUnkCert,
            Rights dwRights,
            out IWMSyncReader ppSyncReader);

        [DllImport("WMVCore.dll", PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern void WMCreateIndexer(
            out IWMIndexer ppIndexer
            );

        [DllImport("WMVCore.dll", PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern void WMCreateBackupRestorer(
            IWMStatusCallback pCallback,
            out IWMLicenseBackup ppBackup
            );

        [DllImport("WMVCore.dll", PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern void WMCreateWriterFileSink(
            out IWMWriterFileSink ppSink
            );

        [DllImport("WMVCore.dll", PreserveSig = false), SuppressUnmanagedCodeSecurity]
        public static extern void WMCreateWriterPushSink(
            out IWMWriterPushSink ppSink
            );

    }

    abstract public class COMBase
    {
        protected const int S_Ok = 0;
        protected const int S_False = 0;

        protected const int E_NotImplemented = unchecked((int)0x80004001);
        protected const int E_NoInterface = unchecked((int)0x80004002);
        protected const int E_Pointer = unchecked((int)0x80004003);
        protected const int E_Abort = unchecked((int)0x80004004);
        protected const int E_Fail = unchecked((int)0x80004005);
        protected const int E_Unexpected = unchecked((int)0x8000FFFF);
        protected const int E_OutOfMemory = unchecked((int)0x8007000E);
        protected const int E_InvalidArgument = unchecked((int)0x80070057);
        protected const int E_BufferTooSmall = unchecked((int)0x8007007a);

        protected static bool Succeeded(int hr)
        {
            return hr >= 0;
        }

        protected static bool Failed(int hr)
        {
            return hr < 0;
        }

        protected static void SafeRelease(object o)
        {
            if (o != null)
            {
                if (Marshal.IsComObject(o))
                {
                    Marshal.ReleaseComObject(o);
                }
                else
                {
                    IDisposable iDis = o as IDisposable;
                    if (iDis != null)
                    {
                        iDis.Dispose();
                    }
                }
            }
        }

        protected static void TRACE(string s)
        {
            Debug.WriteLine(s);
        }
    }

    static public class NSResults
    {
        public const int S_CALLPENDING = unchecked((int)0x000D0000);
        public const int S_CALLABORTED = unchecked((int)0x000D0001);
        public const int S_STREAM_TRUNCATED = unchecked((int)0x000D0002);
        public const int W_SERVER_BANDWIDTH_LIMIT = unchecked((int)0x800D0003);
        public const int W_FILE_BANDWIDTH_LIMIT = unchecked((int)0x800D0004);
        public const int E_NOCONNECTION = unchecked((int)0xC00D0005);
        public const int E_CANNOTCONNECT = unchecked((int)0xC00D0006);
        public const int E_CANNOTDESTROYTITLE = unchecked((int)0xC00D0007);
        public const int E_CANNOTRENAMETITLE = unchecked((int)0xC00D0008);
        public const int E_CANNOTOFFLINEDISK = unchecked((int)0xC00D0009);
        public const int E_CANNOTONLINEDISK = unchecked((int)0xC00D000A);
        public const int E_NOREGISTEREDWALKER = unchecked((int)0xC00D000B);
        public const int E_NOFUNNEL = unchecked((int)0xC00D000C);
        public const int E_NO_LOCALPLAY = unchecked((int)0xC00D000D);
        public const int E_NETWORK_BUSY = unchecked((int)0xC00D000E);
        public const int E_TOO_MANY_SESS = unchecked((int)0xC00D000F);
        public const int E_ALREADY_CONNECTED = unchecked((int)0xC00D0010);
        public const int E_INVALID_INDEX = unchecked((int)0xC00D0011);
        public const int E_PROTOCOL_MISMATCH = unchecked((int)0xC00D0012);
        public const int E_TIMEOUT = unchecked((int)0xC00D0013);
        public const int E_NET_WRITE = unchecked((int)0xC00D0014);
        public const int E_NET_READ = unchecked((int)0xC00D0015);
        public const int E_DISK_WRITE = unchecked((int)0xC00D0016);
        public const int E_DISK_READ = unchecked((int)0xC00D0017);
        public const int E_FILE_WRITE = unchecked((int)0xC00D0018);
        public const int E_FILE_READ = unchecked((int)0xC00D0019);
        public const int E_FILE_NOT_FOUND = unchecked((int)0xC00D001A);
        public const int E_FILE_EXISTS = unchecked((int)0xC00D001B);
        public const int E_INVALID_NAME = unchecked((int)0xC00D001C);
        public const int E_FILE_OPEN_FAILED = unchecked((int)0xC00D001D);
        public const int E_FILE_ALLOCATION_FAILED = unchecked((int)0xC00D001E);
        public const int E_FILE_INIT_FAILED = unchecked((int)0xC00D001F);
        public const int E_FILE_PLAY_FAILED = unchecked((int)0xC00D0020);
        public const int E_SET_DISK_UID_FAILED = unchecked((int)0xC00D0021);
        public const int E_INDUCED = unchecked((int)0xC00D0022);
        public const int E_CCLINK_DOWN = unchecked((int)0xC00D0023);
        public const int E_INTERNAL = unchecked((int)0xC00D0024);
        public const int E_BUSY = unchecked((int)0xC00D0025);
        public const int E_UNRECOGNIZED_STREAM_TYPE = unchecked((int)0xC00D0026);
        public const int E_NETWORK_SERVICE_FAILURE = unchecked((int)0xC00D0027);
        public const int E_NETWORK_RESOURCE_FAILURE = unchecked((int)0xC00D0028);
        public const int E_CONNECTION_FAILURE = unchecked((int)0xC00D0029);
        public const int E_SHUTDOWN = unchecked((int)0xC00D002A);
        public const int E_INVALID_REQUEST = unchecked((int)0xC00D002B);
        public const int E_INSUFFICIENT_BANDWIDTH = unchecked((int)0xC00D002C);
        public const int E_NOT_REBUILDING = unchecked((int)0xC00D002D);
        public const int E_LATE_OPERATION = unchecked((int)0xC00D002E);
        public const int E_INVALID_DATA = unchecked((int)0xC00D002F);
        public const int E_FILE_BANDWIDTH_LIMIT = unchecked((int)0xC00D0030);
        public const int E_OPEN_FILE_LIMIT = unchecked((int)0xC00D0031);
        public const int E_BAD_CONTROL_DATA = unchecked((int)0xC00D0032);
        public const int E_NO_STREAM = unchecked((int)0xC00D0033);
        public const int E_STREAM_END = unchecked((int)0xC00D0034);
        public const int E_SERVER_NOT_FOUND = unchecked((int)0xC00D0035);
        public const int E_DUPLICATE_NAME = unchecked((int)0xC00D0036);
        public const int E_DUPLICATE_ADDRESS = unchecked((int)0xC00D0037);
        public const int E_BAD_MULTICAST_ADDRESS = unchecked((int)0xC00D0038);
        public const int E_BAD_ADAPTER_ADDRESS = unchecked((int)0xC00D0039);
        public const int E_BAD_DELIVERY_MODE = unchecked((int)0xC00D003A);
        public const int E_INVALID_CHANNEL = unchecked((int)0xC00D003B);
        public const int E_INVALID_STREAM = unchecked((int)0xC00D003C);
        public const int E_INVALID_ARCHIVE = unchecked((int)0xC00D003D);
        public const int E_NOTITLES = unchecked((int)0xC00D003E);
        public const int E_INVALID_CLIENT = unchecked((int)0xC00D003F);
        public const int E_INVALID_BLACKHOLE_ADDRESS = unchecked((int)0xC00D0040);
        public const int E_INCOMPATIBLE_FORMAT = unchecked((int)0xC00D0041);
        public const int E_INVALID_KEY = unchecked((int)0xC00D0042);
        public const int E_INVALID_PORT = unchecked((int)0xC00D0043);
        public const int E_INVALID_TTL = unchecked((int)0xC00D0044);
        public const int E_STRIDE_REFUSED = unchecked((int)0xC00D0045);
        public const int E_MMSAUTOSERVER_CANTFINDWALKER = unchecked((int)0xC00D0046);
        public const int E_MAX_BITRATE = unchecked((int)0xC00D0047);
        public const int E_LOGFILEPERIOD = unchecked((int)0xC00D0048);
        public const int E_MAX_CLIENTS = unchecked((int)0xC00D0049);
        public const int E_LOG_FILE_SIZE = unchecked((int)0xC00D004A);
        public const int E_MAX_FILERATE = unchecked((int)0xC00D004B);
        public const int E_WALKER_UNKNOWN = unchecked((int)0xC00D004C);
        public const int E_WALKER_SERVER = unchecked((int)0xC00D004D);
        public const int E_WALKER_USAGE = unchecked((int)0xC00D004E);
        public const int I_TIGER_START = unchecked((int)0x400D004F);
        public const int E_TIGER_FAIL = unchecked((int)0xC00D0050);
        public const int I_CUB_START = unchecked((int)0x400D0051);
        public const int I_CUB_RUNNING = unchecked((int)0x400D0052);
        public const int E_CUB_FAIL = unchecked((int)0xC00D0053);
        public const int I_DISK_START = unchecked((int)0x400D0054);
        public const int E_DISK_FAIL = unchecked((int)0xC00D0055);
        public const int I_DISK_REBUILD_STARTED = unchecked((int)0x400D0056);
        public const int I_DISK_REBUILD_FINISHED = unchecked((int)0x400D0057);
        public const int I_DISK_REBUILD_ABORTED = unchecked((int)0x400D0058);
        public const int I_LIMIT_FUNNELS = unchecked((int)0x400D0059);
        public const int I_START_DISK = unchecked((int)0x400D005A);
        public const int I_STOP_DISK = unchecked((int)0x400D005B);
        public const int I_STOP_CUB = unchecked((int)0x400D005C);
        public const int I_KILL_USERSESSION = unchecked((int)0x400D005D);
        public const int I_KILL_CONNECTION = unchecked((int)0x400D005E);
        public const int I_REBUILD_DISK = unchecked((int)0x400D005F);
        public const int W_UNKNOWN_EVENT = unchecked((int)0x800D0060);
        public const int E_MAX_FUNNELS_ALERT = unchecked((int)0xC00D0060);
        public const int E_ALLOCATE_FILE_FAIL = unchecked((int)0xC00D0061);
        public const int E_PAGING_ERROR = unchecked((int)0xC00D0062);
        public const int E_BAD_BLOCK0_VERSION = unchecked((int)0xC00D0063);
        public const int E_BAD_DISK_UID = unchecked((int)0xC00D0064);
        public const int E_BAD_FSMAJOR_VERSION = unchecked((int)0xC00D0065);
        public const int E_BAD_STAMPNUMBER = unchecked((int)0xC00D0066);
        public const int E_PARTIALLY_REBUILT_DISK = unchecked((int)0xC00D0067);
        public const int E_ENACTPLAN_GIVEUP = unchecked((int)0xC00D0068);
        public const int E_NO_FORMATS = unchecked((int)0xC00D006B);
        public const int E_NO_REFERENCES = unchecked((int)0xC00D006C);
        public const int E_WAVE_OPEN = unchecked((int)0xC00D006D);
        public const int I_LOGGING_FAILED = unchecked((int)0x400D006E);
        public const int E_CANNOTCONNECTEVENTS = unchecked((int)0xC00D006F);
        public const int I_LIMIT_BANDWIDTH = unchecked((int)0x400D0070);
        public const int E_NO_DEVICE = unchecked((int)0xC00D0071);
        public const int E_NO_SPECIFIED_DEVICE = unchecked((int)0xC00D0072);
        //public const int E_NOTFOUND = unchecked((int)0xC00D07F0);
        public const int E_NOTHING_TO_DO = unchecked((int)0xC00D07F1);
        public const int E_NO_MULTICAST = unchecked((int)0xC00D07F2);
        public const int E_MONITOR_GIVEUP = unchecked((int)0xC00D00C8);
        public const int E_REMIRRORED_DISK = unchecked((int)0xC00D00C9);
        public const int E_INSUFFICIENT_DATA = unchecked((int)0xC00D00CA);
        public const int E_ASSERT = unchecked((int)0xC00D00CB);
        public const int E_BAD_ADAPTER_NAME = unchecked((int)0xC00D00CC);
        public const int E_NOT_LICENSED = unchecked((int)0xC00D00CD);
        public const int E_NO_SERVER_CONTACT = unchecked((int)0xC00D00CE);
        public const int E_TOO_MANY_TITLES = unchecked((int)0xC00D00CF);
        public const int E_TITLE_SIZE_EXCEEDED = unchecked((int)0xC00D00D0);
        public const int E_UDP_DISABLED = unchecked((int)0xC00D00D1);
        public const int E_TCP_DISABLED = unchecked((int)0xC00D00D2);
        public const int E_HTTP_DISABLED = unchecked((int)0xC00D00D3);
        public const int E_LICENSE_EXPIRED = unchecked((int)0xC00D00D4);
        public const int E_TITLE_BITRATE = unchecked((int)0xC00D00D5);
        public const int E_EMPTY_PROGRAM_NAME = unchecked((int)0xC00D00D6);
        public const int E_MISSING_CHANNEL = unchecked((int)0xC00D00D7);
        public const int E_NO_CHANNELS = unchecked((int)0xC00D00D8);
        public const int E_INVALID_INDEX2 = unchecked((int)0xC00D00D9);
        public const int E_CUB_FAIL_LINK = unchecked((int)0xC00D0190);
        public const int I_CUB_UNFAIL_LINK = unchecked((int)0x400D0191);
        public const int E_BAD_CUB_UID = unchecked((int)0xC00D0192);
        public const int I_RESTRIPE_START = unchecked((int)0x400D0193);
        public const int I_RESTRIPE_DONE = unchecked((int)0x400D0194);
        public const int E_GLITCH_MODE = unchecked((int)0xC00D0195);
        public const int I_RESTRIPE_DISK_OUT = unchecked((int)0x400D0196);
        public const int I_RESTRIPE_CUB_OUT = unchecked((int)0x400D0197);
        public const int I_DISK_STOP = unchecked((int)0x400D0198);
        public const int I_CATATONIC_FAILURE = unchecked((int)0x800D0199);
        public const int I_CATATONIC_AUTO_UNFAIL = unchecked((int)0x800D019A);
        public const int E_NO_MEDIA_PROTOCOL = unchecked((int)0xC00D019B);
        public const int E_INVALID_INPUT_FORMAT = unchecked((int)0xC00D0BB8);
        public const int E_MSAUDIO_NOT_INSTALLED = unchecked((int)0xC00D0BB9);
        public const int E_UNEXPECTED_MSAUDIO_ERROR = unchecked((int)0xC00D0BBA);
        public const int E_INVALID_OUTPUT_FORMAT = unchecked((int)0xC00D0BBB);
        public const int E_NOT_CONFIGURED = unchecked((int)0xC00D0BBC);
        public const int E_PROTECTED_CONTENT = unchecked((int)0xC00D0BBD);
        public const int E_LICENSE_REQUIRED = unchecked((int)0xC00D0BBE);
        public const int E_TAMPERED_CONTENT = unchecked((int)0xC00D0BBF);
        public const int E_LICENSE_OUTOFDATE = unchecked((int)0xC00D0BC0);
        public const int E_LICENSE_INCORRECT_RIGHTS = unchecked((int)0xC00D0BC1);
        public const int E_AUDIO_CODEC_NOT_INSTALLED = unchecked((int)0xC00D0BC2);
        public const int E_AUDIO_CODEC_ERROR = unchecked((int)0xC00D0BC3);
        public const int E_VIDEO_CODEC_NOT_INSTALLED = unchecked((int)0xC00D0BC4);
        public const int E_VIDEO_CODEC_ERROR = unchecked((int)0xC00D0BC5);
        public const int E_INVALIDPROFILE = unchecked((int)0xC00D0BC6);
        public const int E_INCOMPATIBLE_VERSION = unchecked((int)0xC00D0BC7);
        public const int S_REBUFFERING = unchecked((int)0x000D0BC8);
        public const int S_DEGRADING_QUALITY = unchecked((int)0x000D0BC9);
        public const int E_OFFLINE_MODE = unchecked((int)0xC00D0BCA);
        public const int E_NOT_CONNECTED = unchecked((int)0xC00D0BCB);
        public const int E_TOO_MUCH_DATA = unchecked((int)0xC00D0BCC);
        public const int E_UNSUPPORTED_PROPERTY = unchecked((int)0xC00D0BCD);
        public const int E_8BIT_WAVE_UNSUPPORTED = unchecked((int)0xC00D0BCE);
        public const int E_NO_MORE_SAMPLES = unchecked((int)0xC00D0BCF);
        public const int E_INVALID_SAMPLING_RATE = unchecked((int)0xC00D0BD0);
        public const int E_MAX_PACKET_SIZE_TOO_SMALL = unchecked((int)0xC00D0BD1);
        public const int E_LATE_PACKET = unchecked((int)0xC00D0BD2);
        public const int E_DUPLICATE_PACKET = unchecked((int)0xC00D0BD3);
        public const int E_SDK_BUFFERTOOSMALL = unchecked((int)0xC00D0BD4);
        public const int E_INVALID_NUM_PASSES = unchecked((int)0xC00D0BD5);
        public const int E_ATTRIBUTE_READ_ONLY = unchecked((int)0xC00D0BD6);
        public const int E_ATTRIBUTE_NOT_ALLOWED = unchecked((int)0xC00D0BD7);
        public const int E_INVALID_EDL = unchecked((int)0xC00D0BD8);
        public const int E_DATA_UNIT_EXTENSION_TOO_LARGE = unchecked((int)0xC00D0BD9);
        public const int E_CODEC_DMO_ERROR = unchecked((int)0xC00D0BDA);
        public const int E_NO_CD = unchecked((int)0xC00D0FA0);
        public const int E_CANT_READ_DIGITAL = unchecked((int)0xC00D0FA1);
        public const int E_DEVICE_DISCONNECTED = unchecked((int)0xC00D0FA2);
        public const int E_DEVICE_NOT_SUPPORT_FORMAT = unchecked((int)0xC00D0FA3);
        public const int E_SLOW_READ_DIGITAL = unchecked((int)0xC00D0FA4);
        public const int E_MIXER_INVALID_LINE = unchecked((int)0xC00D0FA5);
        public const int E_MIXER_INVALID_CONTROL = unchecked((int)0xC00D0FA6);
        public const int E_MIXER_INVALID_VALUE = unchecked((int)0xC00D0FA7);
        public const int E_MIXER_UNKNOWN_MMRESULT = unchecked((int)0xC00D0FA8);
        public const int E_USER_STOP = unchecked((int)0xC00D0FA9);
        public const int E_MP3_FORMAT_NOT_FOUND = unchecked((int)0xC00D0FAA);
        public const int E_CD_READ_ERROR_NO_CORRECTION = unchecked((int)0xC00D0FAB);
        public const int E_CD_READ_ERROR = unchecked((int)0xC00D0FAC);
        public const int E_CD_SLOW_COPY = unchecked((int)0xC00D0FAD);
        public const int E_CD_COPYTO_CD = unchecked((int)0xC00D0FAE);
        public const int E_MIXER_NODRIVER = unchecked((int)0xC00D0FAF);
        public const int E_REDBOOK_ENABLED_WHILE_COPYING = unchecked((int)0xC00D0FB0);
        public const int E_CD_REFRESH = unchecked((int)0xC00D0FB1);
        public const int E_CD_DRIVER_PROBLEM = unchecked((int)0xC00D0FB2);
        public const int E_WONT_DO_DIGITAL = unchecked((int)0xC00D0FB3);
        public const int E_WMPXML_NOERROR = unchecked((int)0xC00D0FB4);
        public const int E_WMPXML_ENDOFDATA = unchecked((int)0xC00D0FB5);
        public const int E_WMPXML_PARSEERROR = unchecked((int)0xC00D0FB6);
        public const int E_WMPXML_ATTRIBUTENOTFOUND = unchecked((int)0xC00D0FB7);
        public const int E_WMPXML_PINOTFOUND = unchecked((int)0xC00D0FB8);
        public const int E_WMPXML_EMPTYDOC = unchecked((int)0xC00D0FB9);
        public const int E_WMP_WINDOWSAPIFAILURE = unchecked((int)0xC00D0FC8);
        public const int E_WMP_RECORDING_NOT_ALLOWED = unchecked((int)0xC00D0FC9);
        public const int E_DEVICE_NOT_READY = unchecked((int)0xC00D0FCA);
        public const int E_DAMAGED_FILE = unchecked((int)0xC00D0FCB);
        public const int E_MPDB_GENERIC = unchecked((int)0xC00D0FCC);
        public const int E_FILE_FAILED_CHECKS = unchecked((int)0xC00D0FCD);
        public const int E_MEDIA_LIBRARY_FAILED = unchecked((int)0xC00D0FCE);
        public const int E_SHARING_VIOLATION = unchecked((int)0xC00D0FCF);
        public const int E_NO_ERROR_STRING_FOUND = unchecked((int)0xC00D0FD0);
        public const int E_WMPOCX_NO_REMOTE_CORE = unchecked((int)0xC00D0FD1);
        public const int E_WMPOCX_NO_ACTIVE_CORE = unchecked((int)0xC00D0FD2);
        public const int E_WMPOCX_NOT_RUNNING_REMOTELY = unchecked((int)0xC00D0FD3);
        public const int E_WMPOCX_NO_REMOTE_WINDOW = unchecked((int)0xC00D0FD4);
        public const int E_WMPOCX_ERRORMANAGERNOTAVAILABLE = unchecked((int)0xC00D0FD5);
        public const int E_PLUGIN_NOTSHUTDOWN = unchecked((int)0xC00D0FD6);
        public const int E_WMP_CANNOT_FIND_FOLDER = unchecked((int)0xC00D0FD7);
        public const int E_WMP_STREAMING_RECORDING_NOT_ALLOWED = unchecked((int)0xC00D0FD8);
        public const int E_WMP_PLUGINDLL_NOTFOUND = unchecked((int)0xC00D0FD9);
        public const int E_NEED_TO_ASK_USER = unchecked((int)0xC00D0FDA);
        public const int E_WMPOCX_PLAYER_NOT_DOCKED = unchecked((int)0xC00D0FDB);
        public const int E_WMP_EXTERNAL_NOTREADY = unchecked((int)0xC00D0FDC);
        public const int E_WMP_MLS_STALE_DATA = unchecked((int)0xC00D0FDD);
        public const int E_WMP_UI_SUBCONTROLSNOTSUPPORTED = unchecked((int)0xC00D0FDE);
        public const int E_WMP_UI_VERSIONMISMATCH = unchecked((int)0xC00D0FDF);
        public const int E_WMP_UI_NOTATHEMEFILE = unchecked((int)0xC00D0FE0);
        public const int E_WMP_UI_SUBELEMENTNOTFOUND = unchecked((int)0xC00D0FE1);
        public const int E_WMP_UI_VERSIONPARSE = unchecked((int)0xC00D0FE2);
        public const int E_WMP_UI_VIEWIDNOTFOUND = unchecked((int)0xC00D0FE3);
        public const int E_WMP_UI_PASSTHROUGH = unchecked((int)0xC00D0FE4);
        public const int E_WMP_UI_OBJECTNOTFOUND = unchecked((int)0xC00D0FE5);
        public const int E_WMP_UI_SECONDHANDLER = unchecked((int)0xC00D0FE6);
        public const int E_WMP_UI_NOSKININZIP = unchecked((int)0xC00D0FE7);
        public const int S_WMP_UI_VERSIONMISMATCH = unchecked((int)0x000D0FE8);
        public const int S_WMP_EXCEPTION = unchecked((int)0x000D0FE9);
        public const int E_WMP_URLDOWNLOADFAILED = unchecked((int)0xC00D0FEA);
        public const int E_WMPOCX_UNABLE_TO_LOAD_SKIN = unchecked((int)0xC00D0FEB);
        public const int E_WMP_INVALID_SKIN = unchecked((int)0xC00D0FEC);
        public const int E_WMP_SENDMAILFAILED = unchecked((int)0xC00D0FED);
        public const int E_WMP_SAVEAS_READONLY = unchecked((int)0xC00D0FF0);
        public const int E_WMP_RBC_JPGMAPPINGIMAGE = unchecked((int)0xC00D1004);
        public const int E_WMP_JPGTRANSPARENCY = unchecked((int)0xC00D1005);
        public const int E_WMP_INVALID_MAX_VAL = unchecked((int)0xC00D1009);
        public const int E_WMP_INVALID_MIN_VAL = unchecked((int)0xC00D100A);
        public const int E_WMP_CS_JPGPOSITIONIMAGE = unchecked((int)0xC00D100E);
        public const int E_WMP_CS_NOTEVENLYDIVISIBLE = unchecked((int)0xC00D100F);
        public const int E_WMPZIP_NOTAZIPFILE = unchecked((int)0xC00D1018);
        public const int E_WMPZIP_CORRUPT = unchecked((int)0xC00D1019);
        public const int E_WMPZIP_FILENOTFOUND = unchecked((int)0xC00D101A);
        public const int E_WMP_IMAGE_FILETYPE_UNSUPPORTED = unchecked((int)0xC00D1022);
        public const int E_WMP_IMAGE_INVALID_FORMAT = unchecked((int)0xC00D1023);
        public const int E_WMP_GIF_UNEXPECTED_ENDOFFILE = unchecked((int)0xC00D1024);
        public const int E_WMP_GIF_INVALID_FORMAT = unchecked((int)0xC00D1025);
        public const int E_WMP_GIF_BAD_VERSION_NUMBER = unchecked((int)0xC00D1026);
        public const int E_WMP_GIF_NO_IMAGE_IN_FILE = unchecked((int)0xC00D1027);
        public const int E_WMP_PNG_INVALIDFORMAT = unchecked((int)0xC00D1028);
        public const int E_WMP_PNG_UNSUPPORTED_BITDEPTH = unchecked((int)0xC00D1029);
        public const int E_WMP_PNG_UNSUPPORTED_COMPRESSION = unchecked((int)0xC00D102A);
        public const int E_WMP_PNG_UNSUPPORTED_FILTER = unchecked((int)0xC00D102B);
        public const int E_WMP_PNG_UNSUPPORTED_INTERLACE = unchecked((int)0xC00D102C);
        public const int E_WMP_PNG_UNSUPPORTED_BAD_CRC = unchecked((int)0xC00D102D);
        public const int E_WMP_BMP_INVALID_BITMASK = unchecked((int)0xC00D102E);
        public const int E_WMP_BMP_TOPDOWN_DIB_UNSUPPORTED = unchecked((int)0xC00D102F);
        public const int E_WMP_BMP_BITMAP_NOT_CREATED = unchecked((int)0xC00D1030);
        public const int E_WMP_BMP_COMPRESSION_UNSUPPORTED = unchecked((int)0xC00D1031);
        public const int E_WMP_BMP_INVALID_FORMAT = unchecked((int)0xC00D1032);
        public const int E_WMP_JPG_JERR_ARITHCODING_NOTIMPL = unchecked((int)0xC00D1033);
        public const int E_WMP_JPG_INVALID_FORMAT = unchecked((int)0xC00D1034);
        public const int E_WMP_JPG_BAD_DCTSIZE = unchecked((int)0xC00D1035);
        public const int E_WMP_JPG_BAD_VERSION_NUMBER = unchecked((int)0xC00D1036);
        public const int E_WMP_JPG_BAD_PRECISION = unchecked((int)0xC00D1037);
        public const int E_WMP_JPG_CCIR601_NOTIMPL = unchecked((int)0xC00D1038);
        public const int E_WMP_JPG_NO_IMAGE_IN_FILE = unchecked((int)0xC00D1039);
        public const int E_WMP_JPG_READ_ERROR = unchecked((int)0xC00D103A);
        public const int E_WMP_JPG_FRACT_SAMPLE_NOTIMPL = unchecked((int)0xC00D103B);
        public const int E_WMP_JPG_IMAGE_TOO_BIG = unchecked((int)0xC00D103C);
        public const int E_WMP_JPG_UNEXPECTED_ENDOFFILE = unchecked((int)0xC00D103D);
        public const int E_WMP_JPG_SOF_UNSUPPORTED = unchecked((int)0xC00D103E);
        public const int E_WMP_JPG_UNKNOWN_MARKER = unchecked((int)0xC00D103F);
        public const int S_WMP_LOADED_GIF_IMAGE = unchecked((int)0x000D1040);
        public const int S_WMP_LOADED_PNG_IMAGE = unchecked((int)0x000D1041);
        public const int S_WMP_LOADED_BMP_IMAGE = unchecked((int)0x000D1042);
        public const int S_WMP_LOADED_JPG_IMAGE = unchecked((int)0x000D1043);
        public const int E_WMG_RATEUNAVAILABLE = unchecked((int)0xC00D104A);
        public const int E_WMG_PLUGINUNAVAILABLE = unchecked((int)0xC00D104B);
        public const int E_WMG_CANNOTQUEUE = unchecked((int)0xC00D104C);
        public const int E_WMG_PREROLLLICENSEACQUISITIONNOTALLOWED = unchecked((int)0xC00D104D);
        public const int E_WMG_UNEXPECTEDPREROLLSTATUS = unchecked((int)0xC00D104E);
        public const int E_WMG_INVALIDSTATE = unchecked((int)0xC00D1054);
        public const int E_WMG_SINKALREADYEXISTS = unchecked((int)0xC00D1055);
        public const int E_WMG_NOSDKINTERFACE = unchecked((int)0xC00D1056);
        public const int E_WMG_NOTALLOUTPUTSRENDERED = unchecked((int)0xC00D1057);
        public const int E_WMG_FILETRANSFERNOTALLOWED = unchecked((int)0xC00D1058);
        public const int E_WMR_UNSUPPORTEDSTREAM = unchecked((int)0xC00D1059);
        public const int E_WMR_PINNOTFOUND = unchecked((int)0xC00D105A);
        public const int E_WMR_WAITINGONFORMATSWITCH = unchecked((int)0xC00D105B);
        public const int E_WMR_NOSOURCEFILTER = unchecked((int)0xC00D105C);
        public const int E_WMR_PINTYPENOMATCH = unchecked((int)0xC00D105D);
        public const int E_WMR_NOCALLBACKAVAILABLE = unchecked((int)0xC00D105E);
        public const int S_WMR_ALREADYRENDERED = unchecked((int)0x000D105F);
        public const int S_WMR_PINTYPEPARTIALMATCH = unchecked((int)0x000D1060);
        public const int S_WMR_PINTYPEFULLMATCH = unchecked((int)0x000D1061);
        public const int E_WMR_SAMPLEPROPERTYNOTSET = unchecked((int)0xC00D1062);
        public const int E_WMR_CANNOT_RENDER_BINARY_STREAM = unchecked((int)0xC00D1063);
        public const int E_WMG_LICENSE_TAMPERED = unchecked((int)0xC00D1064);
        public const int E_WMR_WILLNOT_RENDER_BINARY_STREAM = unchecked((int)0xC00D1065);
        public const int E_WMX_UNRECOGNIZED_PLAYLIST_FORMAT = unchecked((int)0xC00D1068);
        public const int E_ASX_INVALIDFORMAT = unchecked((int)0xC00D1069);
        public const int E_ASX_INVALIDVERSION = unchecked((int)0xC00D106A);
        public const int E_ASX_INVALID_REPEAT_BLOCK = unchecked((int)0xC00D106B);
        public const int E_ASX_NOTHING_TO_WRITE = unchecked((int)0xC00D106C);
        public const int E_URLLIST_INVALIDFORMAT = unchecked((int)0xC00D106D);
        public const int E_WMX_ATTRIBUTE_DOES_NOT_EXIST = unchecked((int)0xC00D106E);
        public const int E_WMX_ATTRIBUTE_ALREADY_EXISTS = unchecked((int)0xC00D106F);
        public const int E_WMX_ATTRIBUTE_UNRETRIEVABLE = unchecked((int)0xC00D1070);
        public const int E_WMX_ITEM_DOES_NOT_EXIST = unchecked((int)0xC00D1071);
        public const int E_WMX_ITEM_TYPE_ILLEGAL = unchecked((int)0xC00D1072);
        public const int E_WMX_ITEM_UNSETTABLE = unchecked((int)0xC00D1073);
        public const int E_WMX_PLAYLIST_EMPTY = unchecked((int)0xC00D1074);
        public const int E_MLS_SMARTPLAYLIST_FILTER_NOT_REGISTERED = unchecked((int)0xC00D1075);
        public const int E_WMX_INVALID_FORMAT_OVER_NESTING = unchecked((int)0xC00D1076);
        public const int E_WMPCORE_NOSOURCEURLSTRING = unchecked((int)0xC00D107C);
        public const int E_WMPCORE_COCREATEFAILEDFORGITOBJECT = unchecked((int)0xC00D107D);
        public const int E_WMPCORE_FAILEDTOGETMARSHALLEDEVENTHANDLERINTERFACE = unchecked((int)0xC00D107E);
        public const int E_WMPCORE_BUFFERTOOSMALL = unchecked((int)0xC00D107F);
        public const int E_WMPCORE_UNAVAILABLE = unchecked((int)0xC00D1080);
        public const int E_WMPCORE_INVALIDPLAYLISTMODE = unchecked((int)0xC00D1081);
        public const int E_WMPCORE_ITEMNOTINPLAYLIST = unchecked((int)0xC00D1086);
        public const int E_WMPCORE_PLAYLISTEMPTY = unchecked((int)0xC00D1087);
        public const int E_WMPCORE_NOBROWSER = unchecked((int)0xC00D1088);
        public const int E_WMPCORE_UNRECOGNIZED_MEDIA_URL = unchecked((int)0xC00D1089);
        public const int E_WMPCORE_GRAPH_NOT_IN_LIST = unchecked((int)0xC00D108A);
        public const int E_WMPCORE_PLAYLIST_EMPTY_OR_SINGLE_MEDIA = unchecked((int)0xC00D108B);
        public const int E_WMPCORE_ERRORSINKNOTREGISTERED = unchecked((int)0xC00D108C);
        public const int E_WMPCORE_ERRORMANAGERNOTAVAILABLE = unchecked((int)0xC00D108D);
        public const int E_WMPCORE_WEBHELPFAILED = unchecked((int)0xC00D108E);
        public const int E_WMPCORE_MEDIA_ERROR_RESUME_FAILED = unchecked((int)0xC00D108F);
        public const int E_WMPCORE_NO_REF_IN_ENTRY = unchecked((int)0xC00D1090);
        public const int E_WMPCORE_WMX_LIST_ATTRIBUTE_NAME_EMPTY = unchecked((int)0xC00D1091);
        public const int E_WMPCORE_WMX_LIST_ATTRIBUTE_NAME_ILLEGAL = unchecked((int)0xC00D1092);
        public const int E_WMPCORE_WMX_LIST_ATTRIBUTE_VALUE_EMPTY = unchecked((int)0xC00D1093);
        public const int E_WMPCORE_WMX_LIST_ATTRIBUTE_VALUE_ILLEGAL = unchecked((int)0xC00D1094);
        public const int E_WMPCORE_WMX_LIST_ITEM_ATTRIBUTE_NAME_EMPTY = unchecked((int)0xC00D1095);
        public const int E_WMPCORE_WMX_LIST_ITEM_ATTRIBUTE_NAME_ILLEGAL = unchecked((int)0xC00D1096);
        public const int E_WMPCORE_WMX_LIST_ITEM_ATTRIBUTE_VALUE_EMPTY = unchecked((int)0xC00D1097);
        public const int E_WMPCORE_LIST_ENTRY_NO_REF = unchecked((int)0xC00D1098);
        public const int E_WMPCORE_MISNAMED_FILE = unchecked((int)0xC00D1099);
        public const int E_WMPCORE_CODEC_NOT_TRUSTED = unchecked((int)0xC00D109A);
        public const int E_WMPCORE_CODEC_NOT_FOUND = unchecked((int)0xC00D109B);
        public const int E_WMPCORE_CODEC_DOWNLOAD_NOT_ALLOWED = unchecked((int)0xC00D109C);
        public const int E_WMPCORE_ERROR_DOWNLOADING_PLAYLIST = unchecked((int)0xC00D109D);
        public const int E_WMPCORE_FAILED_TO_BUILD_PLAYLIST = unchecked((int)0xC00D109E);
        public const int E_WMPCORE_PLAYLIST_ITEM_ALTERNATE_NONE = unchecked((int)0xC00D109F);
        public const int E_WMPCORE_PLAYLIST_ITEM_ALTERNATE_EXHAUSTED = unchecked((int)0xC00D10A0);
        public const int E_WMPCORE_PLAYLIST_ITEM_ALTERNATE_NAME_NOT_FOUND = unchecked((int)0xC00D10A1);
        public const int E_WMPCORE_PLAYLIST_ITEM_ALTERNATE_MORPH_FAILED = unchecked((int)0xC00D10A2);
        public const int E_WMPCORE_PLAYLIST_ITEM_ALTERNATE_INIT_FAILED = unchecked((int)0xC00D10A3);
        public const int E_WMPCORE_MEDIA_ALTERNATE_REF_EMPTY = unchecked((int)0xC00D10A4);
        public const int E_WMPCORE_PLAYLIST_NO_EVENT_NAME = unchecked((int)0xC00D10A5);
        public const int E_WMPCORE_PLAYLIST_EVENT_ATTRIBUTE_ABSENT = unchecked((int)0xC00D10A6);
        public const int E_WMPCORE_PLAYLIST_EVENT_EMPTY = unchecked((int)0xC00D10A7);
        public const int E_WMPCORE_PLAYLIST_STACK_EMPTY = unchecked((int)0xC00D10A8);
        public const int E_WMPCORE_CURRENT_MEDIA_NOT_ACTIVE = unchecked((int)0xC00D10A9);
        public const int E_WMPCORE_USER_CANCEL = unchecked((int)0xC00D10AB);
        public const int E_WMPCORE_PLAYLIST_REPEAT_EMPTY = unchecked((int)0xC00D10AC);
        public const int E_WMPCORE_PLAYLIST_REPEAT_START_MEDIA_NONE = unchecked((int)0xC00D10AD);
        public const int E_WMPCORE_PLAYLIST_REPEAT_END_MEDIA_NONE = unchecked((int)0xC00D10AE);
        public const int E_WMPCORE_INVALID_PLAYLIST_URL = unchecked((int)0xC00D10AF);
        public const int E_WMPCORE_MISMATCHED_RUNTIME = unchecked((int)0xC00D10B0);
        public const int E_WMPCORE_PLAYLIST_IMPORT_FAILED_NO_ITEMS = unchecked((int)0xC00D10B1);
        public const int E_WMPCORE_VIDEO_TRANSFORM_FILTER_INSERTION = unchecked((int)0xC00D10B2);
        public const int E_WMPCORE_MEDIA_UNAVAILABLE = unchecked((int)0xC00D10B3);
        public const int E_WMPCORE_WMX_ENTRYREF_NO_REF = unchecked((int)0xC00D10B4);
        public const int E_WMPCORE_NO_PLAYABLE_MEDIA_IN_PLAYLIST = unchecked((int)0xC00D10B5);
        public const int E_WMPCORE_PLAYLIST_EMPTY_NESTED_PLAYLIST_SKIPPED_ITEMS = unchecked((int)0xC00D10B6);
        public const int E_WMPCORE_BUSY = unchecked((int)0xC00D10B7);
        public const int E_WMPCORE_MEDIA_CHILD_PLAYLIST_UNAVAILABLE = unchecked((int)0xC00D10B8);
        public const int E_WMPCORE_MEDIA_NO_CHILD_PLAYLIST = unchecked((int)0xC00D10B9);
        public const int E_WMPCORE_FILE_NOT_FOUND = unchecked((int)0xC00D10BA);
        public const int E_WMPCORE_TEMP_FILE_NOT_FOUND = unchecked((int)0xC00D10BB);
        public const int E_WMDM_REVOKED = unchecked((int)0xC00D10BC);
        public const int E_DDRAW_GENERIC = unchecked((int)0xC00D10BD);
        public const int E_DISPLAY_MODE_CHANGE_FAILED = unchecked((int)0xC00D10BE);
        public const int E_PLAYLIST_CONTAINS_ERRORS = unchecked((int)0xC00D10BF);
        public const int E_CHANGING_PROXY_NAME = unchecked((int)0xC00D10C0);
        public const int E_CHANGING_PROXY_PORT = unchecked((int)0xC00D10C1);
        public const int E_CHANGING_PROXY_EXCEPTIONLIST = unchecked((int)0xC00D10C2);
        public const int E_CHANGING_PROXYBYPASS = unchecked((int)0xC00D10C3);
        public const int E_CHANGING_PROXY_PROTOCOL_NOT_FOUND = unchecked((int)0xC00D10C4);
        public const int E_GRAPH_NOAUDIOLANGUAGE = unchecked((int)0xC00D10C5);
        public const int E_GRAPH_NOAUDIOLANGUAGESELECTED = unchecked((int)0xC00D10C6);
        public const int E_CORECD_NOTAMEDIACD = unchecked((int)0xC00D10C7);
        public const int E_WMPCORE_MEDIA_URL_TOO_LONG = unchecked((int)0xC00D10C8);
        public const int E_WMPFLASH_CANT_FIND_COM_SERVER = unchecked((int)0xC00D10C9);
        public const int E_WMPFLASH_INCOMPATIBLEVERSION = unchecked((int)0xC00D10CA);
        public const int E_WMPOCXGRAPH_IE_DISALLOWS_ACTIVEX_CONTROLS = unchecked((int)0xC00D10CB);
        public const int E_NEED_CORE_REFERENCE = unchecked((int)0xC00D10CC);
        public const int E_MEDIACD_READ_ERROR = unchecked((int)0xC00D10CD);
        public const int E_IE_DISALLOWS_ACTIVEX_CONTROLS = unchecked((int)0xC00D10CE);
        public const int E_FLASH_PLAYBACK_NOT_ALLOWED = unchecked((int)0xC00D10CF);
        public const int E_UNABLE_TO_CREATE_RIP_LOCATION = unchecked((int)0xC00D10D0);
        public const int E_WMPCORE_SOME_CODECS_MISSING = unchecked((int)0xC00D10D1);
        public const int S_WMPCORE_PLAYLISTCLEARABORT = unchecked((int)0x000D10FE);
        public const int S_WMPCORE_PLAYLISTREMOVEITEMABORT = unchecked((int)0x000D10FF);
        public const int S_WMPCORE_PLAYLIST_CREATION_PENDING = unchecked((int)0x000D1102);
        public const int S_WMPCORE_MEDIA_VALIDATION_PENDING = unchecked((int)0x000D1103);
        public const int S_WMPCORE_PLAYLIST_REPEAT_SECONDARY_SEGMENTS_IGNORED = unchecked((int)0x000D1104);
        public const int S_WMPCORE_COMMAND_NOT_AVAILABLE = unchecked((int)0x000D1105);
        public const int S_WMPCORE_PLAYLIST_NAME_AUTO_GENERATED = unchecked((int)0x000D1106);
        public const int S_WMPCORE_PLAYLIST_IMPORT_MISSING_ITEMS = unchecked((int)0x000D1107);
        public const int S_WMPCORE_PLAYLIST_COLLAPSED_TO_SINGLE_MEDIA = unchecked((int)0x000D1108);
        public const int S_WMPCORE_MEDIA_CHILD_PLAYLIST_OPEN_PENDING = unchecked((int)0x000D1109);
        public const int S_WMPCORE_MORE_NODES_AVAIABLE = unchecked((int)0x000D110A);
        public const int E_WMPIM_USEROFFLINE = unchecked((int)0xC00D1126);
        public const int E_WMPIM_USERCANCELED = unchecked((int)0xC00D1127);
        public const int E_WMPIM_DIALUPFAILED = unchecked((int)0xC00D1128);
        public const int E_WINSOCK_ERROR_STRING = unchecked((int)0xC00D1129);
        public const int E_WMPBR_NOLISTENER = unchecked((int)0xC00D1130);
        public const int E_WMPBR_BACKUPCANCEL = unchecked((int)0xC00D1131);
        public const int E_WMPBR_RESTORECANCEL = unchecked((int)0xC00D1132);
        public const int E_WMPBR_ERRORWITHURL = unchecked((int)0xC00D1133);
        public const int E_WMPBR_NAMECOLLISION = unchecked((int)0xC00D1134);
        public const int S_WMPBR_SUCCESS = unchecked((int)0x000D1135);
        public const int S_WMPBR_PARTIALSUCCESS = unchecked((int)0x000D1136);
        public const int E_WMPBR_DRIVE_INVALID = unchecked((int)0xC00D1137);
        public const int S_WMPEFFECT_TRANSPARENT = unchecked((int)0x000D1144);
        public const int S_WMPEFFECT_OPAQUE = unchecked((int)0x000D1145);
        public const int S_OPERATION_PENDING = unchecked((int)0x000D114E);
        public const int E_DVD_NO_SUBPICTURE_STREAM = unchecked((int)0xC00D1162);
        public const int E_DVD_COPY_PROTECT = unchecked((int)0xC00D1163);
        public const int E_DVD_AUTHORING_PROBLEM = unchecked((int)0xC00D1164);
        public const int E_DVD_INVALID_DISC_REGION = unchecked((int)0xC00D1165);
        public const int E_DVD_COMPATIBLE_VIDEO_CARD = unchecked((int)0xC00D1166);
        public const int E_DVD_MACROVISION = unchecked((int)0xC00D1167);
        public const int E_DVD_SYSTEM_DECODER_REGION = unchecked((int)0xC00D1168);
        public const int E_DVD_DISC_DECODER_REGION = unchecked((int)0xC00D1169);
        public const int E_DVD_NO_VIDEO_STREAM = unchecked((int)0xC00D116A);
        public const int E_DVD_NO_AUDIO_STREAM = unchecked((int)0xC00D116B);
        public const int E_DVD_GRAPH_BUILDING = unchecked((int)0xC00D116C);
        public const int E_DVD_NO_DECODER = unchecked((int)0xC00D116D);
        public const int E_DVD_PARENTAL = unchecked((int)0xC00D116E);
        public const int E_DVD_CANNOT_JUMP = unchecked((int)0xC00D116F);
        public const int E_DVD_DEVICE_CONTENTION = unchecked((int)0xC00D1170);
        public const int E_DVD_NO_VIDEO_MEMORY = unchecked((int)0xC00D1171);
        public const int E_DVD_CANNOT_COPY_PROTECTED = unchecked((int)0xC00D1172);
        public const int E_DVD_REQUIRED_PROPERTY_NOT_SET = unchecked((int)0xC00D1173);
        public const int E_DVD_INVALID_TITLE_CHAPTER = unchecked((int)0xC00D1174);
        public const int E_NO_CD_BURNER = unchecked((int)0xC00D1176);
        public const int E_DEVICE_IS_NOT_READY = unchecked((int)0xC00D1177);
        public const int E_PDA_UNSUPPORTED_FORMAT = unchecked((int)0xC00D1178);
        public const int E_NO_PDA = unchecked((int)0xC00D1179);
        public const int E_PDA_UNSPECIFIED_ERROR = unchecked((int)0xC00D117A);
        public const int E_MEMSTORAGE_BAD_DATA = unchecked((int)0xC00D117B);
        public const int E_PDA_FAIL_SELECT_DEVICE = unchecked((int)0xC00D117C);
        public const int E_PDA_FAIL_READ_WAVE_FILE = unchecked((int)0xC00D117D);
        public const int E_IMAPI_LOSSOFSTREAMING = unchecked((int)0xC00D117E);
        public const int E_PDA_DEVICE_FULL = unchecked((int)0xC00D117F);
        public const int E_FAIL_LAUNCH_ROXIO_PLUGIN = unchecked((int)0xC00D1180);
        public const int E_PDA_DEVICE_FULL_IN_SESSION = unchecked((int)0xC00D1181);
        public const int E_IMAPI_MEDIUM_INVALIDTYPE = unchecked((int)0xC00D1182);
        public const int E_WMP_PROTOCOL_PROBLEM = unchecked((int)0xC00D1194);
        public const int E_WMP_NO_DISK_SPACE = unchecked((int)0xC00D1195);
        public const int E_WMP_LOGON_FAILURE = unchecked((int)0xC00D1196);
        public const int E_WMP_CANNOT_FIND_FILE = unchecked((int)0xC00D1197);
        public const int E_WMP_SERVER_INACCESSIBLE = unchecked((int)0xC00D1198);
        public const int E_WMP_UNSUPPORTED_FORMAT = unchecked((int)0xC00D1199);
        public const int E_WMP_DSHOW_UNSUPPORTED_FORMAT = unchecked((int)0xC00D119A);
        public const int E_WMP_PLAYLIST_EXISTS = unchecked((int)0xC00D119B);
        public const int E_WMP_NONMEDIA_FILES = unchecked((int)0xC00D119C);
        public const int E_WMP_INVALID_ASX = unchecked((int)0xC00D119D);
        public const int E_WMP_ALREADY_IN_USE = unchecked((int)0xC00D119E);
        public const int E_WMP_IMAPI_FAILURE = unchecked((int)0xC00D119F);
        public const int E_WMP_WMDM_FAILURE = unchecked((int)0xC00D11A0);
        public const int E_WMP_CODEC_NEEDED_WITH_4CC = unchecked((int)0xC00D11A1);
        public const int E_WMP_CODEC_NEEDED_WITH_FORMATTAG = unchecked((int)0xC00D11A2);
        public const int E_WMP_MSSAP_NOT_AVAILABLE = unchecked((int)0xC00D11A3);
        public const int E_WMP_WMDM_INTERFACEDEAD = unchecked((int)0xC00D11A4);
        public const int E_WMP_WMDM_NOTCERTIFIED = unchecked((int)0xC00D11A5);
        public const int E_WMP_WMDM_LICENSE_NOTEXIST = unchecked((int)0xC00D11A6);
        public const int E_WMP_WMDM_LICENSE_EXPIRED = unchecked((int)0xC00D11A7);
        public const int E_WMP_WMDM_BUSY = unchecked((int)0xC00D11A8);
        public const int E_WMP_WMDM_NORIGHTS = unchecked((int)0xC00D11A9);
        public const int E_WMP_WMDM_INCORRECT_RIGHTS = unchecked((int)0xC00D11AA);
        public const int E_WMP_IMAPI_GENERIC = unchecked((int)0xC00D11AB);
        public const int E_WMP_IMAPI_DEVICE_NOTPRESENT = unchecked((int)0xC00D11AD);
        public const int E_WMP_IMAPI_STASHINUSE = unchecked((int)0xC00D11AE);
        public const int E_WMP_IMAPI_LOSS_OF_STREAMING = unchecked((int)0xC00D11AF);
        public const int E_WMP_SERVER_UNAVAILABLE = unchecked((int)0xC00D11B0);
        public const int E_WMP_FILE_OPEN_FAILED = unchecked((int)0xC00D11B1);
        public const int E_WMP_VERIFY_ONLINE = unchecked((int)0xC00D11B2);
        public const int E_WMP_SERVER_NOT_RESPONDING = unchecked((int)0xC00D11B3);
        public const int E_WMP_DRM_CORRUPT_BACKUP = unchecked((int)0xC00D11B4);
        public const int E_WMP_DRM_LICENSE_SERVER_UNAVAILABLE = unchecked((int)0xC00D11B5);
        public const int E_WMP_NETWORK_FIREWALL = unchecked((int)0xC00D11B6);
        public const int E_WMP_NO_REMOVABLE_MEDIA = unchecked((int)0xC00D11B7);
        public const int E_WMP_PROXY_CONNECT_TIMEOUT = unchecked((int)0xC00D11B8);
        public const int E_WMP_NEED_UPGRADE = unchecked((int)0xC00D11B9);
        public const int E_WMP_AUDIO_HW_PROBLEM = unchecked((int)0xC00D11BA);
        public const int E_WMP_INVALID_PROTOCOL = unchecked((int)0xC00D11BB);
        public const int E_WMP_INVALID_LIBRARY_ADD = unchecked((int)0xC00D11BC);
        public const int E_WMP_MMS_NOT_SUPPORTED = unchecked((int)0xC00D11BD);
        public const int E_WMP_NO_PROTOCOLS_SELECTED = unchecked((int)0xC00D11BE);
        public const int E_WMP_GOFULLSCREEN_FAILED = unchecked((int)0xC00D11BF);
        public const int E_WMP_NETWORK_ERROR = unchecked((int)0xC00D11C0);
        public const int E_WMP_CONNECT_TIMEOUT = unchecked((int)0xC00D11C1);
        public const int E_WMP_MULTICAST_DISABLED = unchecked((int)0xC00D11C2);
        public const int E_WMP_SERVER_DNS_TIMEOUT = unchecked((int)0xC00D11C3);
        public const int E_WMP_PROXY_NOT_FOUND = unchecked((int)0xC00D11C4);
        public const int E_WMP_TAMPERED_CONTENT = unchecked((int)0xC00D11C5);
        public const int E_WMP_OUTOFMEMORY = unchecked((int)0xC00D11C6);
        public const int E_WMP_AUDIO_CODEC_NOT_INSTALLED = unchecked((int)0xC00D11C7);
        public const int E_WMP_VIDEO_CODEC_NOT_INSTALLED = unchecked((int)0xC00D11C8);
        public const int E_WMP_IMAPI_DEVICE_INVALIDTYPE = unchecked((int)0xC00D11C9);
        public const int E_WMP_DRM_DRIVER_AUTH_FAILURE = unchecked((int)0xC00D11CA);
        public const int E_WMP_NETWORK_RESOURCE_FAILURE = unchecked((int)0xC00D11CB);
        public const int E_WMP_UPGRADE_APPLICATION = unchecked((int)0xC00D11CC);
        public const int E_WMP_UNKNOWN_ERROR = unchecked((int)0xC00D11CD);
        public const int E_WMP_INVALID_KEY = unchecked((int)0xC00D11CE);
        public const int E_WMP_CD_ANOTHER_USER = unchecked((int)0xC00D11CF);
        public const int E_WMP_DRM_NEEDS_AUTHORIZATION = unchecked((int)0xC00D11D0);
        public const int E_WMP_BAD_DRIVER = unchecked((int)0xC00D11D1);
        public const int E_WMP_ACCESS_DENIED = unchecked((int)0xC00D11D2);
        public const int E_WMP_LICENSE_RESTRICTS = unchecked((int)0xC00D11D3);
        public const int E_WMP_INVALID_REQUEST = unchecked((int)0xC00D11D4);
        public const int E_WMP_CD_STASH_NO_SPACE = unchecked((int)0xC00D11D5);
        public const int E_WMP_DRM_NEW_HARDWARE = unchecked((int)0xC00D11D6);
        public const int E_WMP_DRM_INVALID_SIG = unchecked((int)0xC00D11D7);
        public const int E_WMP_DRM_CANNOT_RESTORE = unchecked((int)0xC00D11D8);
        public const int E_CD_NO_BUFFERS_READ = unchecked((int)0xC00D11F8);
        public const int E_CD_EMPTY_TRACK_QUEUE = unchecked((int)0xC00D11F9);
        public const int E_CD_NO_READER = unchecked((int)0xC00D11FA);
        public const int E_CD_ISRC_INVALID = unchecked((int)0xC00D11FB);
        public const int E_CD_MEDIA_CATALOG_NUMBER_INVALID = unchecked((int)0xC00D11FC);
        public const int E_SLOW_READ_DIGITAL_WITH_ERRORCORRECTION = unchecked((int)0xC00D11FD);
        public const int E_CD_SPEEDDETECT_NOT_ENOUGH_READS = unchecked((int)0xC00D11FE);
        public const int E_CD_QUEUEING_DISABLED = unchecked((int)0xC00D11FF);
        public const int E_WMP_POLICY_VALUE_NOT_CONFIGURED = unchecked((int)0xC00D122A);
        public const int E_WMP_HWND_NOTFOUND = unchecked((int)0xC00D125C);
        public const int E_BKGDOWNLOAD_WRONG_NO_FILES = unchecked((int)0xC00D125D);
        public const int E_BKGDOWNLOAD_COMPLETECANCELLEDJOB = unchecked((int)0xC00D125E);
        public const int E_BKGDOWNLOAD_CANCELCOMPLETEDJOB = unchecked((int)0xC00D125F);
        public const int E_BKGDOWNLOAD_NOJOBPOINTER = unchecked((int)0xC00D1260);
        public const int E_BKGDOWNLOAD_INVALIDJOBSIGNATURE = unchecked((int)0xC00D1261);
        public const int E_BKGDOWNLOAD_FAILED_TO_CREATE_TEMPFILE = unchecked((int)0xC00D1262);
        public const int E_BKGDOWNLOAD_PLUGIN_FAILEDINITIALIZE = unchecked((int)0xC00D1263);
        public const int E_BKGDOWNLOAD_PLUGIN_FAILEDTOMOVEFILE = unchecked((int)0xC00D1264);
        public const int E_BKGDOWNLOAD_CALLFUNCFAILED = unchecked((int)0xC00D1265);
        public const int E_BKGDOWNLOAD_CALLFUNCTIMEOUT = unchecked((int)0xC00D1266);
        public const int E_BKGDOWNLOAD_CALLFUNCENDED = unchecked((int)0xC00D1267);
        public const int E_BKGDOWNLOAD_WMDUNPACKFAILED = unchecked((int)0xC00D1268);
        public const int E_BKGDOWNLOAD_FAILEDINITIALIZE = unchecked((int)0xC00D1269);
        public const int E_INTERFACE_NOT_REGISTERED_IN_GIT = unchecked((int)0xC00D126A);
        public const int E_BKGDOWNLOAD_INVALID_FILE_NAME = unchecked((int)0xC00D126B);
        public const int E_IMAGE_DOWNLOAD_FAILED = unchecked((int)0xC00D128E);
        public const int E_WMP_UDRM_NOUSERLIST = unchecked((int)0xC00D12C0);
        public const int E_WMP_DRM_NOT_ACQUIRING = unchecked((int)0xC00D12C1);
        public const int E_WMP_BSTR_TOO_LONG = unchecked((int)0xC00D12F2);
        public const int E_WMP_AUTOPLAY_INVALID_STATE = unchecked((int)0xC00D12FC);
        public const int E_CURL_NOTSAFE = unchecked((int)0xC00D1324);
        public const int E_CURL_INVALIDCHAR = unchecked((int)0xC00D1325);
        public const int E_CURL_INVALIDHOSTNAME = unchecked((int)0xC00D1326);
        public const int E_CURL_INVALIDPATH = unchecked((int)0xC00D1327);
        public const int E_CURL_INVALIDSCHEME = unchecked((int)0xC00D1328);
        public const int E_CURL_INVALIDURL = unchecked((int)0xC00D1329);
        public const int E_CURL_CANTWALK = unchecked((int)0xC00D132B);
        public const int E_CURL_INVALIDPORT = unchecked((int)0xC00D132C);
        public const int E_CURLHELPER_NOTADIRECTORY = unchecked((int)0xC00D132D);
        public const int E_CURLHELPER_NOTAFILE = unchecked((int)0xC00D132E);
        public const int E_CURL_CANTDECODE = unchecked((int)0xC00D132F);
        public const int E_CURLHELPER_NOTRELATIVE = unchecked((int)0xC00D1330);
        public const int E_CURL_INVALIDBUFFERSIZE = unchecked((int)0xC00D1355);
        public const int E_SUBSCRIPTIONSERVICE_PLAYBACK_DISALLOWED = unchecked((int)0xC00D1356);
        public const int E_ADVANCEDEDIT_TOO_MANY_PICTURES = unchecked((int)0xC00D136A);
        public const int E_REDIRECT = unchecked((int)0xC00D1388);
        public const int E_STALE_PRESENTATION = unchecked((int)0xC00D1389);
        public const int E_NAMESPACE_WRONG_PERSIST = unchecked((int)0xC00D138A);
        public const int E_NAMESPACE_WRONG_TYPE = unchecked((int)0xC00D138B);
        public const int E_NAMESPACE_NODE_CONFLICT = unchecked((int)0xC00D138C);
        public const int E_NAMESPACE_NODE_NOT_FOUND = unchecked((int)0xC00D138D);
        public const int E_NAMESPACE_BUFFER_TOO_SMALL = unchecked((int)0xC00D138E);
        public const int E_NAMESPACE_TOO_MANY_CALLBACKS = unchecked((int)0xC00D138F);
        public const int E_NAMESPACE_DUPLICATE_CALLBACK = unchecked((int)0xC00D1390);
        public const int E_NAMESPACE_CALLBACK_NOT_FOUND = unchecked((int)0xC00D1391);
        public const int E_NAMESPACE_NAME_TOO_LONG = unchecked((int)0xC00D1392);
        public const int E_NAMESPACE_DUPLICATE_NAME = unchecked((int)0xC00D1393);
        public const int E_NAMESPACE_EMPTY_NAME = unchecked((int)0xC00D1394);
        public const int E_NAMESPACE_INDEX_TOO_LARGE = unchecked((int)0xC00D1395);
        public const int E_NAMESPACE_BAD_NAME = unchecked((int)0xC00D1396);
        public const int E_NAMESPACE_WRONG_SECURITY = unchecked((int)0xC00D1397);
        public const int E_CACHE_ARCHIVE_CONFLICT = unchecked((int)0xC00D13EC);
        public const int E_CACHE_ORIGIN_SERVER_NOT_FOUND = unchecked((int)0xC00D13ED);
        public const int E_CACHE_ORIGIN_SERVER_TIMEOUT = unchecked((int)0xC00D13EE);
        public const int E_CACHE_NOT_BROADCAST = unchecked((int)0xC00D13EF);
        public const int E_CACHE_CANNOT_BE_CACHED = unchecked((int)0xC00D13F0);
        public const int E_CACHE_NOT_MODIFIED = unchecked((int)0xC00D13F1);
        public const int E_CANNOT_REMOVE_PUBLISHING_POINT = unchecked((int)0xC00D1450);
        public const int E_CANNOT_REMOVE_PLUGIN = unchecked((int)0xC00D1451);
        public const int E_WRONG_PUBLISHING_POINT_TYPE = unchecked((int)0xC00D1452);
        public const int E_UNSUPPORTED_LOAD_TYPE = unchecked((int)0xC00D1453);
        public const int E_INVALID_PLUGIN_LOAD_TYPE_CONFIGURATION = unchecked((int)0xC00D1454);
        public const int E_INVALID_PUBLISHING_POINT_NAME = unchecked((int)0xC00D1455);
        public const int E_TOO_MANY_MULTICAST_SINKS = unchecked((int)0xC00D1456);
        public const int E_PUBLISHING_POINT_INVALID_REQUEST_WHILE_STARTED = unchecked((int)0xC00D1457);
        public const int E_MULTICAST_PLUGIN_NOT_ENABLED = unchecked((int)0xC00D1458);
        public const int E_INVALID_OPERATING_SYSTEM_VERSION = unchecked((int)0xC00D1459);
        public const int E_PUBLISHING_POINT_REMOVED = unchecked((int)0xC00D145A);
        public const int E_INVALID_PUSH_PUBLISHING_POINT_START_REQUEST = unchecked((int)0xC00D145B);
        public const int E_UNSUPPORTED_LANGUAGE = unchecked((int)0xC00D145C);
        public const int E_WRONG_OS_VERSION = unchecked((int)0xC00D145D);
        public const int E_PUBLISHING_POINT_STOPPED = unchecked((int)0xC00D145E);
        public const int E_PLAYLIST_ENTRY_ALREADY_PLAYING = unchecked((int)0xC00D14B4);
        public const int E_EMPTY_PLAYLIST = unchecked((int)0xC00D14B5);
        public const int E_PLAYLIST_PARSE_FAILURE = unchecked((int)0xC00D14B6);
        public const int E_PLAYLIST_UNSUPPORTED_ENTRY = unchecked((int)0xC00D14B7);
        public const int E_PLAYLIST_ENTRY_NOT_IN_PLAYLIST = unchecked((int)0xC00D14B8);
        public const int E_PLAYLIST_ENTRY_SEEK = unchecked((int)0xC00D14B9);
        public const int E_PLAYLIST_RECURSIVE_PLAYLISTS = unchecked((int)0xC00D14BA);
        public const int E_PLAYLIST_TOO_MANY_NESTED_PLAYLISTS = unchecked((int)0xC00D14BB);
        public const int E_PLAYLIST_SHUTDOWN = unchecked((int)0xC00D14BC);
        public const int E_PLAYLIST_END_RECEDING = unchecked((int)0xC00D14BD);
        public const int I_PLAYLIST_CHANGE_RECEDING = unchecked((int)0x400D14BE);
        public const int E_DATAPATH_NO_SINK = unchecked((int)0xC00D1518);
        public const int S_PUBLISHING_POINT_STARTED_WITH_FAILED_SINKS = unchecked((int)0x000D1519);
        public const int E_INVALID_PUSH_TEMPLATE = unchecked((int)0xC00D151A);
        public const int E_INVALID_PUSH_PUBLISHING_POINT = unchecked((int)0xC00D151B);
        public const int E_CRITICAL_ERROR = unchecked((int)0xC00D151C);
        public const int E_NO_NEW_CONNECTIONS = unchecked((int)0xC00D151D);
        public const int E_WSX_INVALID_VERSION = unchecked((int)0xC00D151E);
        public const int E_HEADER_MISMATCH = unchecked((int)0xC00D151F);
        public const int E_PUSH_DUPLICATE_PUBLISHING_POINT_NAME = unchecked((int)0xC00D1520);
        public const int E_NO_SCRIPT_ENGINE = unchecked((int)0xC00D157C);
        public const int E_PLUGIN_ERROR_REPORTED = unchecked((int)0xC00D157D);
        public const int E_SOURCE_PLUGIN_NOT_FOUND = unchecked((int)0xC00D157E);
        public const int E_PLAYLIST_PLUGIN_NOT_FOUND = unchecked((int)0xC00D157F);
        public const int E_DATA_SOURCE_ENUMERATION_NOT_SUPPORTED = unchecked((int)0xC00D1580);
        public const int E_MEDIA_PARSER_INVALID_FORMAT = unchecked((int)0xC00D1581);
        public const int E_SCRIPT_DEBUGGER_NOT_INSTALLED = unchecked((int)0xC00D1582);
        public const int E_FEATURE_REQUIRES_ENTERPRISE_SERVER = unchecked((int)0xC00D1583);
        public const int E_WIZARD_RUNNING = unchecked((int)0xC00D1584);
        public const int E_INVALID_LOG_URL = unchecked((int)0xC00D1585);
        public const int E_INVALID_MTU_RANGE = unchecked((int)0xC00D1586);
        public const int E_INVALID_PLAY_STATISTICS = unchecked((int)0xC00D1587);
        public const int E_LOG_NEED_TO_BE_SKIPPED = unchecked((int)0xC00D1588);
        public const int E_HTTP_TEXT_DATACONTAINER_SIZE_LIMIT_EXCEEDED = unchecked((int)0xC00D1589);
        public const int E_PORT_IN_USE = unchecked((int)0xC00D158A);
        public const int E_PORT_IN_USE_HTTP = unchecked((int)0xC00D158B);
        public const int E_HTTP_TEXT_DATACONTAINER_INVALID_SERVER_RESPONSE = unchecked((int)0xC00D158C);
        public const int E_ARCHIVE_REACH_QUOTA = unchecked((int)0xC00D158D);
        public const int E_ARCHIVE_ABORT_DUE_TO_BCAST = unchecked((int)0xC00D158E);
        public const int E_ARCHIVE_GAP_DETECTED = unchecked((int)0xC00D158F);
        public const int E_BAD_MARKIN = unchecked((int)0xC00D1B58);
        public const int E_BAD_MARKOUT = unchecked((int)0xC00D1B59);
        public const int E_NOMATCHING_MEDIASOURCE = unchecked((int)0xC00D1B5A);
        public const int E_UNSUPPORTED_SOURCETYPE = unchecked((int)0xC00D1B5B);
        public const int E_TOO_MANY_AUDIO = unchecked((int)0xC00D1B5C);
        public const int E_TOO_MANY_VIDEO = unchecked((int)0xC00D1B5D);
        public const int E_NOMATCHING_ELEMENT = unchecked((int)0xC00D1B5E);
        public const int E_MISMATCHED_MEDIACONTENT = unchecked((int)0xC00D1B5F);
        public const int E_CANNOT_DELETE_ACTIVE_SOURCEGROUP = unchecked((int)0xC00D1B60);
        public const int E_AUDIODEVICE_BUSY = unchecked((int)0xC00D1B61);
        public const int E_AUDIODEVICE_UNEXPECTED = unchecked((int)0xC00D1B62);
        public const int E_AUDIODEVICE_BADFORMAT = unchecked((int)0xC00D1B63);
        public const int E_VIDEODEVICE_BUSY = unchecked((int)0xC00D1B64);
        public const int E_VIDEODEVICE_UNEXPECTED = unchecked((int)0xC00D1B65);
        public const int E_INVALIDCALL_WHILE_ENCODER_RUNNING = unchecked((int)0xC00D1B66);
        public const int E_NO_PROFILE_IN_SOURCEGROUP = unchecked((int)0xC00D1B67);
        public const int E_VIDEODRIVER_UNSTABLE = unchecked((int)0xC00D1B68);
        public const int E_VIDCAPSTARTFAILED = unchecked((int)0xC00D1B69);
        public const int E_VIDSOURCECOMPRESSION = unchecked((int)0xC00D1B6A);
        public const int E_VIDSOURCESIZE = unchecked((int)0xC00D1B6B);
        public const int E_ICMQUERYFORMAT = unchecked((int)0xC00D1B6C);
        public const int E_VIDCAPCREATEWINDOW = unchecked((int)0xC00D1B6D);
        public const int E_VIDCAPDRVINUSE = unchecked((int)0xC00D1B6E);
        public const int E_NO_MEDIAFORMAT_IN_SOURCE = unchecked((int)0xC00D1B6F);
        public const int E_NO_VALID_OUTPUT_STREAM = unchecked((int)0xC00D1B70);
        public const int E_NO_VALID_SOURCE_PLUGIN = unchecked((int)0xC00D1B71);
        public const int E_NO_ACTIVE_SOURCEGROUP = unchecked((int)0xC00D1B72);
        public const int E_NO_SCRIPT_STREAM = unchecked((int)0xC00D1B73);
        public const int E_INVALIDCALL_WHILE_ARCHIVAL_RUNNING = unchecked((int)0xC00D1B74);
        public const int E_INVALIDPACKETSIZE = unchecked((int)0xC00D1B75);
        public const int E_PLUGIN_CLSID_INVALID = unchecked((int)0xC00D1B76);
        public const int E_UNSUPPORTED_ARCHIVETYPE = unchecked((int)0xC00D1B77);
        public const int E_UNSUPPORTED_ARCHIVEOPERATION = unchecked((int)0xC00D1B78);
        public const int E_ARCHIVE_FILENAME_NOTSET = unchecked((int)0xC00D1B79);
        public const int E_SOURCEGROUP_NOTPREPARED = unchecked((int)0xC00D1B7A);
        public const int E_PROFILE_MISMATCH = unchecked((int)0xC00D1B7B);
        public const int E_INCORRECTCLIPSETTINGS = unchecked((int)0xC00D1B7C);
        public const int E_NOSTATSAVAILABLE = unchecked((int)0xC00D1B7D);
        public const int E_NOTARCHIVING = unchecked((int)0xC00D1B7E);
        public const int E_INVALIDCALL_WHILE_ENCODER_STOPPED = unchecked((int)0xC00D1B7F);
        public const int E_NOSOURCEGROUPS = unchecked((int)0xC00D1B80);
        public const int E_INVALIDINPUTFPS = unchecked((int)0xC00D1B81);
        public const int E_NO_DATAVIEW_SUPPORT = unchecked((int)0xC00D1B82);
        public const int E_CODEC_UNAVAILABLE = unchecked((int)0xC00D1B83);
        public const int E_ARCHIVE_SAME_AS_INPUT = unchecked((int)0xC00D1B84);
        public const int E_SOURCE_NOTSPECIFIED = unchecked((int)0xC00D1B85);
        public const int E_NO_REALTIME_TIMECOMPRESSION = unchecked((int)0xC00D1B86);
        public const int E_UNSUPPORTED_ENCODER_DEVICE = unchecked((int)0xC00D1B87);
        public const int E_UNEXPECTED_DISPLAY_SETTINGS = unchecked((int)0xC00D1B88);
        public const int E_NO_AUDIODATA = unchecked((int)0xC00D1B89);
        public const int E_INPUTSOURCE_PROBLEM = unchecked((int)0xC00D1B8A);
        public const int E_WME_VERSION_MISMATCH = unchecked((int)0xC00D1B8B);
        public const int E_NO_REALTIME_PREPROCESS = unchecked((int)0xC00D1B8C);
        public const int E_NO_REPEAT_PREPROCESS = unchecked((int)0xC00D1B8D);
        public const int E_CANNOT_PAUSE_LIVEBROADCAST = unchecked((int)0xC00D1B8E);
        public const int E_DRM_PROFILE_NOT_SET = unchecked((int)0xC00D1B8F);
        public const int E_DUPLICATE_DRMPROFILE = unchecked((int)0xC00D1B90);
        public const int E_INVALID_DEVICE = unchecked((int)0xC00D1B91);
        public const int E_SPEECHEDL_ON_NON_MIXEDMODE = unchecked((int)0xC00D1B92);
        public const int E_DRM_PASSWORD_TOO_LONG = unchecked((int)0xC00D1B93);
        public const int E_DEVCONTROL_FAILED_SEEK = unchecked((int)0xC00D1B94);
        public const int E_INTERLACE_REQUIRE_SAMESIZE = unchecked((int)0xC00D1B95);
        public const int E_TOO_MANY_DEVICECONTROL = unchecked((int)0xC00D1B96);
        public const int E_NO_MULTIPASS_FOR_LIVEDEVICE = unchecked((int)0xC00D1B97);
        public const int E_MISSING_AUDIENCE = unchecked((int)0xC00D1B98);
        public const int E_AUDIENCE_CONTENTTYPE_MISMATCH = unchecked((int)0xC00D1B99);
        public const int E_MISSING_SOURCE_INDEX = unchecked((int)0xC00D1B9A);
        public const int E_NUM_LANGUAGE_MISMATCH = unchecked((int)0xC00D1B9B);
        public const int E_LANGUAGE_MISMATCH = unchecked((int)0xC00D1B9C);
        public const int E_VBRMODE_MISMATCH = unchecked((int)0xC00D1B9D);
        public const int E_INVALID_INPUT_AUDIENCE_INDEX = unchecked((int)0xC00D1B9E);
        public const int E_INVALID_INPUT_LANGUAGE = unchecked((int)0xC00D1B9F);
        public const int E_INVALID_INPUT_STREAM = unchecked((int)0xC00D1BA0);
        public const int E_EXPECT_MONO_WAV_INPUT = unchecked((int)0xC00D1BA1);
        public const int E_INPUT_WAVFORMAT_MISMATCH = unchecked((int)0xC00D1BA2);
        public const int E_RECORDQ_DISK_FULL = unchecked((int)0xC00D1BA3);
        public const int E_NO_PAL_INVERSE_TELECINE = unchecked((int)0xC00D1BA4);
        public const int E_ACTIVE_SG_DEVICE_DISCONNECTED = unchecked((int)0xC00D1BA5);
        public const int E_ACTIVE_SG_DEVICE_CONTROL_DISCONNECTED = unchecked((int)0xC00D1BA6);
        public const int E_NO_FRAMES_SUBMITTED_TO_ANALYZER = unchecked((int)0xC00D1BA7);
        public const int E_INPUT_DOESNOT_SUPPORT_SMPTE = unchecked((int)0xC00D1BA8);
        public const int E_NO_SMPTE_WITH_MULTIPLE_SOURCEGROUPS = unchecked((int)0xC00D1BA9);
        public const int E_BAD_CONTENTEDL = unchecked((int)0xC00D1BAA);
        public const int E_INTERLACEMODE_MISMATCH = unchecked((int)0xC00D1BAB);
        public const int E_NONSQUAREPIXELMODE_MISMATCH = unchecked((int)0xC00D1BAC);
        public const int E_SMPTEMODE_MISMATCH = unchecked((int)0xC00D1BAD);
        public const int E_END_OF_TAPE = unchecked((int)0xC00D1BAE);
        public const int E_NO_MEDIA_IN_AUDIENCE = unchecked((int)0xC00D1BAF);
        public const int E_NO_AUDIENCES = unchecked((int)0xC00D1BB0);
        public const int E_NO_AUDIO_COMPAT = unchecked((int)0xC00D1BB1);
        public const int E_INVALID_VBR_COMPAT = unchecked((int)0xC00D1BB2);
        public const int E_NO_PROFILE_NAME = unchecked((int)0xC00D1BB3);
        public const int E_INVALID_VBR_WITH_UNCOMP = unchecked((int)0xC00D1BB4);
        public const int E_MULTIPLE_VBR_AUDIENCES = unchecked((int)0xC00D1BB5);
        public const int E_UNCOMP_COMP_COMBINATION = unchecked((int)0xC00D1BB6);
        public const int E_MULTIPLE_AUDIO_CODECS = unchecked((int)0xC00D1BB7);
        public const int E_MULTIPLE_AUDIO_FORMATS = unchecked((int)0xC00D1BB8);
        public const int E_AUDIO_BITRATE_STEPDOWN = unchecked((int)0xC00D1BB9);
        public const int E_INVALID_AUDIO_PEAKRATE = unchecked((int)0xC00D1BBA);
        public const int E_INVALID_AUDIO_PEAKRATE_2 = unchecked((int)0xC00D1BBB);
        public const int E_INVALID_AUDIO_BUFFERMAX = unchecked((int)0xC00D1BBC);
        public const int E_MULTIPLE_VIDEO_CODECS = unchecked((int)0xC00D1BBD);
        public const int E_MULTIPLE_VIDEO_SIZES = unchecked((int)0xC00D1BBE);
        public const int E_INVALID_VIDEO_BITRATE = unchecked((int)0xC00D1BBF);
        public const int E_VIDEO_BITRATE_STEPDOWN = unchecked((int)0xC00D1BC0);
        public const int E_INVALID_VIDEO_PEAKRATE = unchecked((int)0xC00D1BC1);
        public const int E_INVALID_VIDEO_PEAKRATE_2 = unchecked((int)0xC00D1BC2);
        public const int E_INVALID_VIDEO_WIDTH = unchecked((int)0xC00D1BC3);
        public const int E_INVALID_VIDEO_HEIGHT = unchecked((int)0xC00D1BC4);
        public const int E_INVALID_VIDEO_FPS = unchecked((int)0xC00D1BC5);
        public const int E_INVALID_VIDEO_KEYFRAME = unchecked((int)0xC00D1BC6);
        public const int E_INVALID_VIDEO_IQUALITY = unchecked((int)0xC00D1BC7);
        public const int E_INVALID_VIDEO_CQUALITY = unchecked((int)0xC00D1BC8);
        public const int E_INVALID_VIDEO_BUFFER = unchecked((int)0xC00D1BC9);
        public const int E_INVALID_VIDEO_BUFFERMAX = unchecked((int)0xC00D1BCA);
        public const int E_INVALID_VIDEO_BUFFERMAX_2 = unchecked((int)0xC00D1BCB);
        public const int E_INVALID_VIDEO_WIDTH_ALIGN = unchecked((int)0xC00D1BCC);
        public const int E_INVALID_VIDEO_HEIGHT_ALIGN = unchecked((int)0xC00D1BCD);
        public const int E_MULTIPLE_SCRIPT_BITRATES = unchecked((int)0xC00D1BCE);
        public const int E_INVALID_SCRIPT_BITRATE = unchecked((int)0xC00D1BCF);
        public const int E_MULTIPLE_FILE_BITRATES = unchecked((int)0xC00D1BD0);
        public const int E_INVALID_FILE_BITRATE = unchecked((int)0xC00D1BD1);
        public const int E_SAME_AS_INPUT_COMBINATION = unchecked((int)0xC00D1BD2);
        public const int E_SOURCE_CANNOT_LOOP = unchecked((int)0xC00D1BD3);
        public const int E_INVALID_FOLDDOWN_COEFFICIENTS = unchecked((int)0xC00D1BD4);
        public const int E_DRMPROFILE_NOTFOUND = unchecked((int)0xC00D1BD5);
        public const int E_INVALID_TIMECODE = unchecked((int)0xC00D1BD6);
        public const int E_NO_AUDIO_TIMECOMPRESSION = unchecked((int)0xC00D1BD7);
        public const int E_NO_TWOPASS_TIMECOMPRESSION = unchecked((int)0xC00D1BD8);
        public const int E_TIMECODE_REQUIRES_VIDEOSTREAM = unchecked((int)0xC00D1BD9);
        public const int E_NO_MBR_WITH_TIMECODE = unchecked((int)0xC00D1BDA);
        public const int E_INVALID_INTERLACEMODE = unchecked((int)0xC00D1BDB);
        public const int E_INVALID_INTERLACE_COMPAT = unchecked((int)0xC00D1BDC);
        public const int E_INVALID_NONSQUAREPIXEL_COMPAT = unchecked((int)0xC00D1BDD);
        public const int E_INVALID_SOURCE_WITH_DEVICE_CONTROL = unchecked((int)0xC00D1BDE);
        public const int E_CANNOT_GENERATE_BROADCAST_INFO_FOR_QUALITYVBR = unchecked((int)0xC00D1BDF);
        public const int E_EXCEED_MAX_DRM_PROFILE_LIMIT = unchecked((int)0xC00D1BE0);
        public const int E_DEVICECONTROL_UNSTABLE = unchecked((int)0xC00D1BE1);
        public const int E_INVALID_PIXEL_ASPECT_RATIO = unchecked((int)0xC00D1BE2);
        public const int E_AUDIENCE__LANGUAGE_CONTENTTYPE_MISMATCH = unchecked((int)0xC00D1BE3);
        public const int E_INVALID_PROFILE_CONTENTTYPE = unchecked((int)0xC00D1BE4);
        public const int E_TRANSFORM_PLUGIN_NOT_FOUND = unchecked((int)0xC00D1BE5);
        public const int E_TRANSFORM_PLUGIN_INVALID = unchecked((int)0xC00D1BE6);
        public const int E_EDL_REQUIRED_FOR_DEVICE_MULTIPASS = unchecked((int)0xC00D1BE7);
        public const int E_INVALID_VIDEO_WIDTH_FOR_INTERLACED_ENCODING = unchecked((int)0xC00D1BE8);
        public const int E_DRM_INVALID_APPLICATION = unchecked((int)0xC00D2711);
        public const int E_DRM_LICENSE_STORE_ERROR = unchecked((int)0xC00D2712);
        public const int E_DRM_SECURE_STORE_ERROR = unchecked((int)0xC00D2713);
        public const int E_DRM_LICENSE_STORE_SAVE_ERROR = unchecked((int)0xC00D2714);
        public const int E_DRM_SECURE_STORE_UNLOCK_ERROR = unchecked((int)0xC00D2715);
        public const int E_DRM_INVALID_CONTENT = unchecked((int)0xC00D2716);
        public const int E_DRM_UNABLE_TO_OPEN_LICENSE = unchecked((int)0xC00D2717);
        public const int E_DRM_INVALID_LICENSE = unchecked((int)0xC00D2718);
        public const int E_DRM_INVALID_MACHINE = unchecked((int)0xC00D2719);
        public const int E_DRM_ENUM_LICENSE_FAILED = unchecked((int)0xC00D271B);
        public const int E_DRM_INVALID_LICENSE_REQUEST = unchecked((int)0xC00D271C);
        public const int E_DRM_UNABLE_TO_INITIALIZE = unchecked((int)0xC00D271D);
        public const int E_DRM_UNABLE_TO_ACQUIRE_LICENSE = unchecked((int)0xC00D271E);
        public const int E_DRM_INVALID_LICENSE_ACQUIRED = unchecked((int)0xC00D271F);
        public const int E_DRM_NO_RIGHTS = unchecked((int)0xC00D2720);
        public const int E_DRM_KEY_ERROR = unchecked((int)0xC00D2721);
        public const int E_DRM_ENCRYPT_ERROR = unchecked((int)0xC00D2722);
        public const int E_DRM_DECRYPT_ERROR = unchecked((int)0xC00D2723);
        public const int E_DRM_LICENSE_INVALID_XML = unchecked((int)0xC00D2725);
        public const int S_DRM_LICENSE_ACQUIRED = unchecked((int)0x000D2726);
        public const int S_DRM_INDIVIDUALIZED = unchecked((int)0x000D2727);
        public const int E_DRM_NEEDS_INDIVIDUALIZATION = unchecked((int)0xC00D2728);
        public const int E_DRM_ALREADY_INDIVIDUALIZED = unchecked((int)0xC00D2729);
        public const int E_DRM_ACTION_NOT_QUERIED = unchecked((int)0xC00D272A);
        public const int E_DRM_ACQUIRING_LICENSE = unchecked((int)0xC00D272B);
        public const int E_DRM_INDIVIDUALIZING = unchecked((int)0xC00D272C);
        public const int E_DRM_PARAMETERS_MISMATCHED = unchecked((int)0xC00D272F);
        public const int E_DRM_UNABLE_TO_CREATE_LICENSE_OBJECT = unchecked((int)0xC00D2730);
        public const int E_DRM_UNABLE_TO_CREATE_INDI_OBJECT = unchecked((int)0xC00D2731);
        public const int E_DRM_UNABLE_TO_CREATE_ENCRYPT_OBJECT = unchecked((int)0xC00D2732);
        public const int E_DRM_UNABLE_TO_CREATE_DECRYPT_OBJECT = unchecked((int)0xC00D2733);
        public const int E_DRM_UNABLE_TO_CREATE_PROPERTIES_OBJECT = unchecked((int)0xC00D2734);
        public const int E_DRM_UNABLE_TO_CREATE_BACKUP_OBJECT = unchecked((int)0xC00D2735);
        public const int E_DRM_INDIVIDUALIZE_ERROR = unchecked((int)0xC00D2736);
        public const int E_DRM_LICENSE_OPEN_ERROR = unchecked((int)0xC00D2737);
        public const int E_DRM_LICENSE_CLOSE_ERROR = unchecked((int)0xC00D2738);
        public const int E_DRM_GET_LICENSE_ERROR = unchecked((int)0xC00D2739);
        public const int E_DRM_QUERY_ERROR = unchecked((int)0xC00D273A);
        public const int E_DRM_REPORT_ERROR = unchecked((int)0xC00D273B);
        public const int E_DRM_GET_LICENSESTRING_ERROR = unchecked((int)0xC00D273C);
        public const int E_DRM_GET_CONTENTSTRING_ERROR = unchecked((int)0xC00D273D);
        public const int E_DRM_MONITOR_ERROR = unchecked((int)0xC00D273E);
        public const int E_DRM_UNABLE_TO_SET_PARAMETER = unchecked((int)0xC00D273F);
        public const int E_DRM_INVALID_APPDATA = unchecked((int)0xC00D2740);
        public const int E_DRM_INVALID_APPDATA_VERSION = unchecked((int)0xC00D2741);
        public const int E_DRM_BACKUP_EXISTS = unchecked((int)0xC00D2742);
        public const int E_DRM_BACKUP_CORRUPT = unchecked((int)0xC00D2743);
        public const int E_DRM_BACKUPRESTORE_BUSY = unchecked((int)0xC00D2744);
        public const int S_DRM_MONITOR_CANCELLED = unchecked((int)0x000D2746);
        public const int S_DRM_ACQUIRE_CANCELLED = unchecked((int)0x000D2747);
        public const int E_DRM_LICENSE_UNUSABLE = unchecked((int)0xC00D2748);
        public const int E_DRM_INVALID_PROPERTY = unchecked((int)0xC00D2749);
        public const int E_DRM_SECURE_STORE_NOT_FOUND = unchecked((int)0xC00D274A);
        public const int E_DRM_CACHED_CONTENT_ERROR = unchecked((int)0xC00D274B);
        public const int E_DRM_INDIVIDUALIZATION_INCOMPLETE = unchecked((int)0xC00D274C);
        public const int E_DRM_DRIVER_AUTH_FAILURE = unchecked((int)0xC00D274D);
        public const int E_DRM_NEED_UPGRADE_MSSAP = unchecked((int)0xC00D274E);
        public const int E_DRM_REOPEN_CONTENT = unchecked((int)0xC00D274F);
        public const int E_DRM_DRIVER_DIGIOUT_FAILURE = unchecked((int)0xC00D2750);
        public const int E_DRM_INVALID_SECURESTORE_PASSWORD = unchecked((int)0xC00D2751);
        public const int E_DRM_APPCERT_REVOKED = unchecked((int)0xC00D2752);
        public const int E_DRM_RESTORE_FRAUD = unchecked((int)0xC00D2753);
        public const int E_DRM_HARDWARE_INCONSISTENT = unchecked((int)0xC00D2754);
        public const int E_DRM_SDMI_TRIGGER = unchecked((int)0xC00D2755);
        public const int E_DRM_SDMI_NOMORECOPIES = unchecked((int)0xC00D2756);
        public const int E_DRM_UNABLE_TO_CREATE_HEADER_OBJECT = unchecked((int)0xC00D2757);
        public const int E_DRM_UNABLE_TO_CREATE_KEYS_OBJECT = unchecked((int)0xC00D2758);
        public const int E_DRM_LICENSE_NOTACQUIRED = unchecked((int)0xC00D2759);
        public const int E_DRM_UNABLE_TO_CREATE_CODING_OBJECT = unchecked((int)0xC00D275A);
        public const int E_DRM_UNABLE_TO_CREATE_STATE_DATA_OBJECT = unchecked((int)0xC00D275B);
        public const int E_DRM_BUFFER_TOO_SMALL = unchecked((int)0xC00D275C);
        public const int E_DRM_UNSUPPORTED_PROPERTY = unchecked((int)0xC00D275D);
        public const int E_DRM_ERROR_BAD_NET_RESP = unchecked((int)0xC00D275E);
        public const int E_DRM_STORE_NOTALLSTORED = unchecked((int)0xC00D275F);
        public const int E_DRM_SECURITY_COMPONENT_SIGNATURE_INVALID = unchecked((int)0xC00D2760);
        public const int E_DRM_INVALID_DATA = unchecked((int)0xC00D2761);
        public const int E_DRM_UNABLE_TO_CONTACT_SERVER = unchecked((int)0xC00D2762);
        public const int E_DRM_UNABLE_TO_CREATE_AUTHENTICATION_OBJECT = unchecked((int)0xC00D2763);
        public const int E_DRM_NOT_CONFIGURED = unchecked((int)0xC00D2764);
        public const int E_DRM_DEVICE_ACTIVATION_CANCELED = unchecked((int)0xC00D2765);
        public const int E_DRM_LICENSE_EXPIRED = unchecked((int)0xC00D27D8);
        public const int E_DRM_LICENSE_NOTENABLED = unchecked((int)0xC00D27D9);
        public const int E_DRM_LICENSE_APPSECLOW = unchecked((int)0xC00D27DA);
        public const int E_DRM_STORE_NEEDINDI = unchecked((int)0xC00D27DB);
        public const int E_DRM_STORE_NOTALLOWED = unchecked((int)0xC00D27DC);
        public const int E_DRM_LICENSE_APP_NOTALLOWED = unchecked((int)0xC00D27DD);
        public const int S_DRM_NEEDS_INDIVIDUALIZATION = unchecked((int)0x000D27DE);
        public const int E_DRM_LICENSE_CERT_EXPIRED = unchecked((int)0xC00D27DF);
        public const int E_DRM_LICENSE_SECLOW = unchecked((int)0xC00D27E0);
        public const int E_DRM_LICENSE_CONTENT_REVOKED = unchecked((int)0xC00D27E1);
        public const int E_DRM_LICENSE_NOSAP = unchecked((int)0xC00D280A);
        public const int E_DRM_LICENSE_NOSVP = unchecked((int)0xC00D280B);
        public const int E_DRM_LICENSE_NOWDM = unchecked((int)0xC00D280C);
        public const int E_DRM_LICENSE_NOTRUSTEDCODEC = unchecked((int)0xC00D280D);
        public const int E_DRM_NEEDS_UPGRADE_TEMPFILE = unchecked((int)0xC00D283D);
        public const int E_DRM_NEED_UPGRADE_PD = unchecked((int)0xC00D283E);
        public const int E_DRM_SIGNATURE_FAILURE = unchecked((int)0xC00D283F);
        public const int E_DRM_LICENSE_SERVER_INFO_MISSING = unchecked((int)0xC00D2840);
        public const int E_DRM_BUSY = unchecked((int)0xC00D2841);
        public const int E_DRM_PD_TOO_MANY_DEVICES = unchecked((int)0xC00D2842);
        public const int E_DRM_INDIV_FRAUD = unchecked((int)0xC00D2843);
        public const int E_DRM_INDIV_NO_CABS = unchecked((int)0xC00D2844);
        public const int E_DRM_INDIV_SERVICE_UNAVAILABLE = unchecked((int)0xC00D2845);
        public const int E_DRM_RESTORE_SERVICE_UNAVAILABLE = unchecked((int)0xC00D2846);
        public const int S_REBOOT_RECOMMENDED = unchecked((int)0x000D2AF8);
        public const int S_REBOOT_REQUIRED = unchecked((int)0x000D2AF9);
        public const int E_REBOOT_RECOMMENDED = unchecked((int)0xC00D2AFA);
        public const int E_REBOOT_REQUIRED = unchecked((int)0xC00D2AFB);
        public const int E_UNKNOWN_PROTOCOL = unchecked((int)0xC00D2EE0);
        public const int E_REDIRECT_TO_PROXY = unchecked((int)0xC00D2EE1);
        public const int E_INTERNAL_SERVER_ERROR = unchecked((int)0xC00D2EE2);
        public const int E_BAD_REQUEST = unchecked((int)0xC00D2EE3);
        public const int E_ERROR_FROM_PROXY = unchecked((int)0xC00D2EE4);
        public const int E_PROXY_TIMEOUT = unchecked((int)0xC00D2EE5);
        public const int E_SERVER_UNAVAILABLE = unchecked((int)0xC00D2EE6);
        public const int E_REFUSED_BY_SERVER = unchecked((int)0xC00D2EE7);
        public const int E_INCOMPATIBLE_SERVER = unchecked((int)0xC00D2EE8);
        public const int E_MULTICAST_DISABLED = unchecked((int)0xC00D2EE9);
        public const int E_INVALID_REDIRECT = unchecked((int)0xC00D2EEA);
        public const int E_ALL_PROTOCOLS_DISABLED = unchecked((int)0xC00D2EEB);
        public const int E_MSBD_NO_LONGER_SUPPORTED = unchecked((int)0xC00D2EEC);
        public const int E_PROXY_NOT_FOUND = unchecked((int)0xC00D2EED);
        public const int E_CANNOT_CONNECT_TO_PROXY = unchecked((int)0xC00D2EEE);
        public const int E_SERVER_DNS_TIMEOUT = unchecked((int)0xC00D2EEF);
        public const int E_PROXY_DNS_TIMEOUT = unchecked((int)0xC00D2EF0);
        public const int E_CLOSED_ON_SUSPEND = unchecked((int)0xC00D2EF1);
        public const int E_CANNOT_READ_PLAYLIST_FROM_MEDIASERVER = unchecked((int)0xC00D2EF2);
        public const int E_SESSION_NOT_FOUND = unchecked((int)0xC00D2EF3);
        public const int E_REQUIRE_STREAMING_CLIENT = unchecked((int)0xC00D2EF4);
        public const int E_PLAYLIST_ENTRY_HAS_CHANGED = unchecked((int)0xC00D2EF5);
        public const int E_PROXY_ACCESSDENIED = unchecked((int)0xC00D2EF6);
        public const int E_PROXY_SOURCE_ACCESSDENIED = unchecked((int)0xC00D2EF7);
        public const int E_NETWORK_SINK_WRITE = unchecked((int)0xC00D2EF8);
        public const int E_FIREWALL = unchecked((int)0xC00D2EF9);
        public const int E_MMS_NOT_SUPPORTED = unchecked((int)0xC00D2EFA);
        public const int E_SERVER_ACCESSDENIED = unchecked((int)0xC00D2EFB);
        public const int E_RESOURCE_GONE = unchecked((int)0xC00D2EFC);
        public const int E_NO_EXISTING_PACKETIZER = unchecked((int)0xC00D2EFD);
        public const int E_BAD_SYNTAX_IN_SERVER_RESPONSE = unchecked((int)0xC00D2EFE);
        public const int I_RECONNECTED = unchecked((int)0x400D2EFF);
        public const int E_RESET_SOCKET_CONNECTION = unchecked((int)0xC00D2F00);
        public const int I_NOLOG_STOP = unchecked((int)0x400D2F01);
        public const int E_TOO_MANY_HOPS = unchecked((int)0xC00D2F02);
        public const int I_EXISTING_PACKETIZER = unchecked((int)0x400D2F03);
        public const int I_MANUAL_PROXY = unchecked((int)0x400D2F04);
        public const int E_TOO_MUCH_DATA_FROM_SERVER = unchecked((int)0xC00D2F05);
        public const int E_CONNECT_TIMEOUT = unchecked((int)0xC00D2F06);
        public const int E_PROXY_CONNECT_TIMEOUT = unchecked((int)0xC00D2F07);
        public const int E_SESSION_INVALID = unchecked((int)0xC00D2F08);
        public const int S_EOSRECEDING = unchecked((int)0x000D2F09);
        public const int E_PACKETSINK_UNKNOWN_FEC_STREAM = unchecked((int)0xC00D2F0A);
        public const int E_PUSH_CANNOTCONNECT = unchecked((int)0xC00D2F0B);
        public const int E_INCOMPATIBLE_PUSH_SERVER = unchecked((int)0xC00D2F0C);
        public const int S_CHANGENOTICE = unchecked((int)0x000D2F0D);
        public const int E_END_OF_PLAYLIST = unchecked((int)0xC00D32C8);
        public const int E_USE_FILE_SOURCE = unchecked((int)0xC00D32C9);
        public const int E_PROPERTY_NOT_FOUND = unchecked((int)0xC00D32CA);
        public const int E_PROPERTY_READ_ONLY = unchecked((int)0xC00D32CC);
        public const int E_TABLE_KEY_NOT_FOUND = unchecked((int)0xC00D32CD);
        public const int E_INVALID_QUERY_OPERATOR = unchecked((int)0xC00D32CF);
        public const int E_INVALID_QUERY_PROPERTY = unchecked((int)0xC00D32D0);
        public const int E_PROPERTY_NOT_SUPPORTED = unchecked((int)0xC00D32D2);
        public const int E_SCHEMA_CLASSIFY_FAILURE = unchecked((int)0xC00D32D4);
        public const int E_METADATA_FORMAT_NOT_SUPPORTED = unchecked((int)0xC00D32D5);
        public const int E_METADATA_NO_EDITING_CAPABILITY = unchecked((int)0xC00D32D6);
        public const int E_METADATA_CANNOT_SET_LOCALE = unchecked((int)0xC00D32D7);
        public const int E_METADATA_LANGUAGE_NOT_SUPORTED = unchecked((int)0xC00D32D8);
        public const int E_METADATA_NO_RFC1766_NAME_FOR_LOCALE = unchecked((int)0xC00D32D9);
        public const int E_METADATA_NOT_AVAILABLE = unchecked((int)0xC00D32DA);
        public const int E_METADATA_CACHE_DATA_NOT_AVAILABLE = unchecked((int)0xC00D32DB);
        public const int E_METADATA_INVALID_DOCUMENT_TYPE = unchecked((int)0xC00D32DC);
        public const int E_METADATA_IDENTIFIER_NOT_AVAILABLE = unchecked((int)0xC00D32DD);
        public const int E_METADATA_CANNOT_RETRIEVE_FROM_OFFLINE_CACHE = unchecked((int)0xC00D32DE);

        public const int I_NO_EVENTS = unchecked((int)0x400D0069);
        public const int E_REGKEY_NOT_FOUND = unchecked((int)0xC00D006A);
    }

    static public class WMError
    {
        #region Private methods

        /// <summary>
        /// From #defines in WinBase.h
        /// </summary>
        [Flags]
        private enum LoadLibraryExFlags
        {
            DontResolveDllReferences = 0x00000001,
            LoadLibraryAsDataFile = 0x00000002,
            LoadWithAlteredSearchPath = 0x00000008,
            LoadIgnoreCodeAuthzLevel = 0x00000010
        }

        /// <summary>
        /// From FORMAT_MESSAGE_* defines in WinBase.h
        /// </summary>
        [Flags]
        private enum FormatMessageFlags
        {
            AllocateBuffer = 0x00000100,
            IgnoreInserts = 0x00000200,
            FromString = 0x00000400,
            FromHmodule = 0x00000800,
            FromSystem = 0x00001000,
            ArgumentArray = 0x00002000,
            MaxWidthMask = 0x000000FF
        }

        [DllImport("kernel32.dll", ExactSpelling = true, CharSet = CharSet.Unicode, EntryPoint = "FormatMessageW"), SuppressUnmanagedCodeSecurity]
        private static extern int FormatMessage(FormatMessageFlags dwFlags, IntPtr lpSource,
            int dwMessageId, int dwLanguageId, ref IntPtr lpBuffer, int nSize, IntPtr Arguments);

        [DllImport("kernel32.dll", ExactSpelling = true, CharSet = CharSet.Unicode, EntryPoint = "LoadLibraryExW"), SuppressUnmanagedCodeSecurity]
        private static extern IntPtr LoadLibraryEx(string lpFileName, IntPtr hFile, LoadLibraryExFlags dwFlags);

        [DllImport("kernel32.dll"), SuppressUnmanagedCodeSecurity]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool FreeLibrary(IntPtr hFile);

        [DllImport("kernel32.dll", SetLastError = true), SuppressUnmanagedCodeSecurity]
        private static extern IntPtr LocalFree(IntPtr hMem);

        #endregion

        public static string GetErrorText(int hr)
        {
            string sRet = null;
            IntPtr hModule;
            FormatMessageFlags dwFormatFlags = FormatMessageFlags.AllocateBuffer | FormatMessageFlags.IgnoreInserts | FormatMessageFlags.FromSystem;
            int dwBufferLength;
            IntPtr ip = IntPtr.Zero;

            // Load the Windows Media error message dll
            hModule = LoadLibraryEx("wmerror.dll", IntPtr.Zero, LoadLibraryExFlags.LoadLibraryAsDataFile);
            if (hModule != IntPtr.Zero)
            {
                // If the load succeeds, make sure we look in it
                dwFormatFlags |= FormatMessageFlags.FromHmodule;
            }

            // Scan both the Windows Media library, and the system library looking for the message
            dwBufferLength = FormatMessage(
                dwFormatFlags,
                hModule, // module to get message from (NULL == system)
                hr, // error number to get message for
                0, // default language
                ref ip,
                0,
                IntPtr.Zero
                );

            try
            {
                // Convert the returned buffer to a string.  If ip is null (due to not finding
                // the message), no exception is thrown.  sRet just stays null.  The
                // try/finally is for the (remote) possibility that we run out of memory
                // creating the string.
                sRet = Marshal.PtrToStringUni(ip);
            }
            finally
            {
                // Cleanup
                FreeLibrary(hModule);
                LocalFree(ip);
            }

            return sRet;
        }

        /// <summary>
        /// If hr has a "failed" status code (E_*), throw an exception.  Note that status
        /// messages (S_*) are not considered failure codes.  If Windows Media error text
        /// is available, it is used to build the exception, otherwise a generic com error
        /// is thrown.
        /// </summary>
        /// <param name="hr">The HRESULT to check</param>
        public static void ThrowExceptionForHR(int hr)
        {
            // If an error has occurred
            if (hr < 0)
            {
                // If a string is returned, build a com error from it
                string buf = GetErrorText(hr);
                if (buf != null)
                {
                    throw new COMException(buf, hr);
                }
                else
                {
                    // No string, just use standard com error
                    Marshal.ThrowExceptionForHR(hr);
                }
            }
        }

    }

    [AttributeUsage(AttributeTargets.Enum | AttributeTargets.Struct | AttributeTargets.Class)]
    internal class UnmanagedNameAttribute : Attribute
    {
        private string m_Name;

        public UnmanagedNameAttribute(string s)
        {
            m_Name = s;
        }

        public override string ToString()
        {
            return m_Name;
        }
    }

    static public class Constants
    {
        ////////////////////////////////////////////////////////////////
        //
        // These are the special case attributes that give information
        // about the Windows Media file.
        //
        public const int g_dwWMSpecialAttributes = 20;
        public const string g_wszWMDuration = "Duration";
        public const string g_wszWMBitrate = "Bitrate";
        public const string g_wszWMSeekable = "Seekable";
        public const string g_wszWMStridable = "Stridable";
        public const string g_wszWMBroadcast = "Broadcast";
        public const string g_wszWMProtected = "Is_Protected";
        public const string g_wszWMTrusted = "Is_Trusted";
        public const string g_wszWMSignature_Name = "Signature_Name";
        public const string g_wszWMHasAudio = "HasAudio";
        public const string g_wszWMHasImage = "HasImage";
        public const string g_wszWMHasScript = "HasScript";
        public const string g_wszWMHasVideo = "HasVideo";
        public const string g_wszWMCurrentBitrate = "CurrentBitrate";
        public const string g_wszWMOptimalBitrate = "OptimalBitrate";
        public const string g_wszWMHasAttachedImages = "HasAttachedImages";
        public const string g_wszWMSkipBackward = "Can_Skip_Backward";
        public const string g_wszWMSkipForward = "Can_Skip_Forward";
        public const string g_wszWMNumberOfFrames = "NumberOfFrames";
        public const string g_wszWMFileSize = "FileSize";
        public const string g_wszWMHasArbitraryDataStream = "HasArbitraryDataStream";
        public const string g_wszWMHasFileTransferStream = "HasFileTransferStream";
        public const string g_wszWMContainerFormat = "WM/ContainerFormat";

        ////////////////////////////////////////////////////////////////
        //
        // The content description object supports 5 basic attributes.
        //
        public const int g_dwWMContentAttributes = 5;
        public const string g_wszWMTitle = "Title";
        public const string g_wszWMAuthor = "Author";
        public const string g_wszWMDescription = "Description";
        public const string g_wszWMRating = "Rating";
        public const string g_wszWMCopyright = "Copyright";

        ////////////////////////////////////////////////////////////////
        //
        // These attributes are used to configure and query DRM settings in the reader and writer.
        //
        public const string g_wszWMUse_DRM = "Use_DRM";
        public const string g_wszWMDRM_Flags = "DRM_Flags";
        public const string g_wszWMDRM_Level = "DRM_Level";
        public const string g_wszWMUse_Advanced_DRM = "Use_Advanced_DRM";
        public const string g_wszWMDRM_KeySeed = "DRM_KeySeed";
        public const string g_wszWMDRM_KeyID = "DRM_KeyID";
        public const string g_wszWMDRM_ContentID = "DRM_ContentID";
        public const string g_wszWMDRM_SourceID = "DRM_SourceID";
        public const string g_wszWMDRM_IndividualizedVersion = "DRM_IndividualizedVersion";
        public const string g_wszWMDRM_LicenseAcqURL = "DRM_LicenseAcqURL";
        public const string g_wszWMDRM_V1LicenseAcqURL = "DRM_V1LicenseAcqURL";
        public const string g_wszWMDRM_HeaderSignPrivKey = "DRM_HeaderSignPrivKey";
        public const string g_wszWMDRM_LASignaturePrivKey = "DRM_LASignaturePrivKey";
        public const string g_wszWMDRM_LASignatureCert = "DRM_LASignatureCert";
        public const string g_wszWMDRM_LASignatureLicSrvCert = "DRM_LASignatureLicSrvCert";
        public const string g_wszWMDRM_LASignatureRootCert = "DRM_LASignatureRootCert";

        ////////////////////////////////////////////////////////////////
        //
        // These are the additional attributes defined in the WM attribute
        // namespace that give information about the content.
        //
        public const string g_wszWMAlbumTitle = "WM/AlbumTitle";
        public const string g_wszWMTrack = "WM/Track";
        public const string g_wszWMPromotionURL = "WM/PromotionURL";
        public const string g_wszWMAlbumCoverURL = "WM/AlbumCoverURL";
        public const string g_wszWMGenre = "WM/Genre";
        public const string g_wszWMYear = "WM/Year";
        public const string g_wszWMGenreID = "WM/GenreID";
        public const string g_wszWMMCDI = "WM/MCDI";
        public const string g_wszWMComposer = "WM/Composer";
        public const string g_wszWMLyrics = "WM/Lyrics";
        public const string g_wszWMTrackNumber = "WM/TrackNumber";
        public const string g_wszWMToolName = "WM/ToolName";
        public const string g_wszWMToolVersion = "WM/ToolVersion";
        public const string g_wszWMIsVBR = "IsVBR";
        public const string g_wszWMAlbumArtist = "WM/AlbumArtist";

        ////////////////////////////////////////////////////////////////
        //
        // These optional attributes may be used to give information
        // about the branding of the content.
        //
        public const string g_wszWMBannerImageType = "BannerImageType";
        public const string g_wszWMBannerImageData = "BannerImageData";
        public const string g_wszWMBannerImageURL = "BannerImageURL";
        public const string g_wszWMCopyrightURL = "CopyrightURL";
        ////////////////////////////////////////////////////////////////
        //
        // Optional attributes, used to give information
        // about video stream properties.
        //
        public const string g_wszWMAspectRatioX = "AspectRatioX";
        public const string g_wszWMAspectRatioY = "AspectRatioY";
        ////////////////////////////////////////////////////////////////
        //
        // Optional attributes, used to give information
        // about the overall streaming properties of VBR files.
        // This attribute takes the format:
        //  WORD wReserved (must be 0)
        //  WM_LEAKY_BUCKET_PAIR pair1
        //  WM_LEAKY_BUCKET_PAIR pair2
        //  ...
        //
        public const string g_wszASFLeakyBucketPairs = "ASFLeakyBucketPairs";
        ////////////////////////////////////////////////////////////////
        //
        // The NSC file supports the following attributes.
        //
        public const int g_dwWMNSCAttributes = 5;
        public const string g_wszWMNSCName = "NSC_Name";
        public const string g_wszWMNSCAddress = "NSC_Address";
        public const string g_wszWMNSCPhone = "NSC_Phone";
        public const string g_wszWMNSCEmail = "NSC_Email";
        public const string g_wszWMNSCDescription = "NSC_Description";

        ////////////////////////////////////////////////////////////////
        //
        // Attributes introduced in V9
        //
        public const string g_wszWMWriter = "WM/Writer";
        public const string g_wszWMConductor = "WM/Conductor";
        public const string g_wszWMProducer = "WM/Producer";
        public const string g_wszWMDirector = "WM/Director";
        public const string g_wszWMContentGroupDescription = "WM/ContentGroupDescription";
        public const string g_wszWMSubTitle = "WM/SubTitle";
        public const string g_wszWMPartOfSet = "WM/PartOfSet";
        public const string g_wszWMProtectionType = "WM/ProtectionType";
        public const string g_wszWMVideoHeight = "WM/VideoHeight";
        public const string g_wszWMVideoWidth = "WM/VideoWidth";
        public const string g_wszWMVideoFrameRate = "WM/VideoFrameRate";
        public const string g_wszWMMediaClassPrimaryID = "WM/MediaClassPrimaryID";
        public const string g_wszWMMediaClassSecondaryID = "WM/MediaClassSecondaryID";
        public const string g_wszWMPeriod = "WM/Period";
        public const string g_wszWMCategory = "WM/Category";
        public const string g_wszWMPicture = "WM/Picture";
        public const string g_wszWMLyrics_Synchronised = "WM/Lyrics_Synchronised";
        public const string g_wszWMOriginalLyricist = "WM/OriginalLyricist";
        public const string g_wszWMOriginalArtist = "WM/OriginalArtist";
        public const string g_wszWMOriginalAlbumTitle = "WM/OriginalAlbumTitle";
        public const string g_wszWMOriginalReleaseYear = "WM/OriginalReleaseYear";
        public const string g_wszWMOriginalFilename = "WM/OriginalFilename";
        public const string g_wszWMPublisher = "WM/Publisher";
        public const string g_wszWMEncodedBy = "WM/EncodedBy";
        public const string g_wszWMEncodingSettings = "WM/EncodingSettings";
        public const string g_wszWMEncodingTime = "WM/EncodingTime";
        public const string g_wszWMAuthorURL = "WM/AuthorURL";
        public const string g_wszWMUserWebURL = "WM/UserWebURL";
        public const string g_wszWMAudioFileURL = "WM/AudioFileURL";
        public const string g_wszWMAudioSourceURL = "WM/AudioSourceURL";
        public const string g_wszWMLanguage = "WM/Language";
        public const string g_wszWMParentalRating = "WM/ParentalRating";
        public const string g_wszWMBeatsPerMinute = "WM/BeatsPerMinute";
        public const string g_wszWMInitialKey = "WM/InitialKey";
        public const string g_wszWMMood = "WM/Mood";
        public const string g_wszWMText = "WM/Text";
        public const string g_wszWMDVDID = "WM/DVDID";
        public const string g_wszWMWMContentID = "WM/WMContentID";
        public const string g_wszWMWMCollectionID = "WM/WMCollectionID";
        public const string g_wszWMWMCollectionGroupID = "WM/WMCollectionGroupID";
        public const string g_wszWMUniqueFileIdentifier = "WM/UniqueFileIdentifier";
        public const string g_wszWMModifiedBy = "WM/ModifiedBy";
        public const string g_wszWMRadioStationName = "WM/RadioStationName";
        public const string g_wszWMRadioStationOwner = "WM/RadioStationOwner";
        public const string g_wszWMPlaylistDelay = "WM/PlaylistDelay";
        public const string g_wszWMCodec = "WM/Codec";
        public const string g_wszWMDRM = "WM/DRM";
        public const string g_wszWMISRC = "WM/ISRC";
        public const string g_wszWMProvider = "WM/Provider";
        public const string g_wszWMProviderRating = "WM/ProviderRating";
        public const string g_wszWMProviderStyle = "WM/ProviderStyle";
        public const string g_wszWMContentDistributor = "WM/ContentDistributor";
        public const string g_wszWMSubscriptionContentID = "WM/SubscriptionContentID";
        public const string g_wszWMWMADRCPeakReference = "WM/WMADRCPeakReference";
        public const string g_wszWMWMADRCPeakTarget = "WM/WMADRCPeakTarget";
        public const string g_wszWMWMADRCAverageReference = "WM/WMADRCAverageReference";
        public const string g_wszWMWMADRCAverageTarget = "WM/WMADRCAverageTarget";
        ////////////////////////////////////////////////////////////////
        //
        // Attributes introduced in V10
        //
        public const string g_wszWMStreamTypeInfo = "WM/StreamTypeInfo";
        public const string g_wszWMPeakBitrate = "WM/PeakBitrate";
        public const string g_wszWMASFPacketCount = "WM/ASFPacketCount";
        public const string g_wszWMASFSecurityObjectsSize = "WM/ASFSecurityObjectsSize";
        public const string g_wszWMSharedUserRating = "WM/SharedUserRating";
        public const string g_wszWMSubTitleDescription = "WM/SubTitleDescription";
        public const string g_wszWMMediaCredits = "WM/MediaCredits";
        public const string g_wszWMParentalRatingReason = "WM/ParentalRatingReason";
        public const string g_wszWMOriginalReleaseTime = "WM/OriginalReleaseTime";
        public const string g_wszWMMediaStationCallSign = "WM/MediaStationCallSign";
        public const string g_wszWMMediaStationName = "WM/MediaStationName";
        public const string g_wszWMMediaNetworkAffiliation = "WM/MediaNetworkAffiliation";
        public const string g_wszWMMediaOriginalChannel = "WM/MediaOriginalChannel";
        public const string g_wszWMMediaOriginalBroadcastDateTime = "WM/MediaOriginalBroadcastDateTime";
        public const string g_wszWMMediaIsStereo = "WM/MediaIsStereo";
        public const string g_wszWMVideoClosedCaptioning = "WM/VideoClosedCaptioning";
        public const string g_wszWMMediaIsRepeat = "WM/MediaIsRepeat";
        public const string g_wszWMMediaIsLive = "WM/MediaIsLive";
        public const string g_wszWMMediaIsTape = "WM/MediaIsTape";
        public const string g_wszWMMediaIsDelay = "WM/MediaIsDelay";
        public const string g_wszWMMediaIsSubtitled = "WM/MediaIsSubtitled";
        public const string g_wszWMMediaIsPremiere = "WM/MediaIsPremiere";
        public const string g_wszWMMediaIsFinale = "WM/MediaIsFinale";
        public const string g_wszWMMediaIsSAP = "WM/MediaIsSAP";
        public const string g_wszWMProviderCopyright = "WM/ProviderCopyright";
        ////////////////////////////////////////////////////////////////
        //
        // Attributes introduced in V11
        //
        public const string g_wszWMISAN = "WM/ISAN";
        public const string g_wszWMADID = "WM/ADID";
        public const string g_wszWMWMShadowFileSourceFileType = "WM/WMShadowFileSourceFileType";
        public const string g_wszWMWMShadowFileSourceDRMType = "WM/WMShadowFileSourceDRMType";
        public const string g_wszWMWMCPDistributor = "WM/WMCPDistributor";
        public const string g_wszWMWMCPDistributorID = "WM/WMCPDistributorID";
        ////////////////////////////////////////////////////////////////
        //
        // These are setting names for use in Get/SetOutputSetting
        //
        public const string g_wszEarlyDataDelivery = "EarlyDataDelivery";
        public const string g_wszJustInTimeDecode = "JustInTimeDecode";
        public const string g_wszSingleOutputBuffer = "SingleOutputBuffer";
        public const string g_wszSoftwareScaling = "SoftwareScaling";
        public const string g_wszDeliverOnReceive = "DeliverOnReceive";
        public const string g_wszScrambledAudio = "ScrambledAudio";
        public const string g_wszDedicatedDeliveryThread = "DedicatedDeliveryThread";
        public const string g_wszEnableDiscreteOutput = "EnableDiscreteOutput";
        public const string g_wszSpeakerConfig = "SpeakerConfig";
        public const string g_wszDynamicRangeControl = "DynamicRangeControl";
        public const string g_wszAllowInterlacedOutput = "AllowInterlacedOutput";
        public const string g_wszVideoSampleDurations = "VideoSampleDurations";
        public const string g_wszStreamLanguage = "StreamLanguage";
        public const string g_wszEnableWMAProSPDIFOutput = "EnableWMAProSPDIFOutput";

        ////////////////////////////////////////////////////////////////
        //
        // These are setting names for use in Get/SetInputSetting
        //
        public const string g_wszDeinterlaceMode = "DeinterlaceMode";
        public const string g_wszInitialPatternForInverseTelecine = "InitialPatternForInverseTelecine";
        public const string g_wszJPEGCompressionQuality = "JPEGCompressionQuality";
        public const string g_wszWatermarkCLSID = "WatermarkCLSID";
        public const string g_wszWatermarkConfig = "WatermarkConfig";
        public const string g_wszInterlacedCoding = "InterlacedCoding";
        public const string g_wszFixedFrameRate = "FixedFrameRate";

        ////////////////////////////////////////////////////////////////
        //
        // All known IWMPropertyVault property names
        //
        // g_wszOriginalSourceFormatTag is obsolete and has been superceded by g_wszOriginalWaveFormat
        public const string g_wszOriginalSourceFormatTag = "_SOURCEFORMATTAG";
        public const string g_wszOriginalWaveFormat = "_ORIGINALWAVEFORMAT";
        public const string g_wszEDL = "_EDL";
        public const string g_wszComplexity = "_COMPLEXITYEX";
        public const string g_wszDecoderComplexityRequested = "_DECODERCOMPLEXITYPROFILE";

        ////////////////////////////////////////////////////////////////
        //
        // All known IWMIStreamProps property names
        //
        public const string g_wszReloadIndexOnSeek = "ReloadIndexOnSeek";
        public const string g_wszStreamNumIndexObjects = "StreamNumIndexObjects";
        public const string g_wszFailSeekOnError = "FailSeekOnError";
        public const string g_wszPermitSeeksBeyondEndOfStream = "PermitSeeksBeyondEndOfStream";
        public const string g_wszUsePacketAtSeekPoint = "UsePacketAtSeekPoint";
        public const string g_wszSourceBufferTime = "SourceBufferTime";
        public const string g_wszSourceMaxBytesAtOnce = "SourceMaxBytesAtOnce";

        ////////////////////////////////////////////////////////////////
        //
        // VBR encoding settings
        //
        public const string g_wszVBREnabled = "_VBRENABLED";
        public const string g_wszVBRQuality = "_VBRQUALITY";
        public const string g_wszVBRBitrateMax = "_RMAX";
        public const string g_wszVBRBufferWindowMax = "_BMAX";

        ////////////////////////////////////////////////////////////////
        //
        // VBR Video settings
        //
        public const string g_wszVBRPeak = "VBR Peak";
        public const string g_wszBufferAverage = "Buffer Average";

        ////////////////////////////////////////////////////////////////
        //
        // Codec encoding complexity settings
        //
        // g_wszComplexity should be used to set desired encoding complexity on the
        // stream's IWMPropertyVault (see above for definition)
        // The below settings can be queried from IWMCodecInfo3::GetCodecProp()
        //
        public const string g_wszComplexityMax = "_COMPLEXITYEXMAX";
        public const string g_wszComplexityOffline = "_COMPLEXITYEXOFFLINE";
        public const string g_wszComplexityLive = "_COMPLEXITYEXLIVE";
        public const string g_wszIsVBRSupported = "_ISVBRSUPPORTED";
        ////////////////////////////////////////////////////////////////
        //
        // Codec enumeration settings
        //
        // g_wszVBREnabled can be used as a codec enumeration setting (see above for definition)
        public const string g_wszNumPasses = "_PASSESUSED";

        ////////////////////////////////////////////////////////////////
        //
        // These are WMA Voice V9 attribute names and values
        //
        public const string g_wszMusicSpeechClassMode = "MusicSpeechClassMode";
        public const string g_wszMusicClassMode = "MusicClassMode";
        public const string g_wszSpeechClassMode = "SpeechClassMode";
        public const string g_wszMixedClassMode = "MixedClassMode";

        ////////////////////////////////////////////////////////////////
        //
        // The WMA Voice V9 supports the following format property.
        //
        public const string g_wszSpeechCaps = "SpeechFormatCap";

        ////////////////////////////////////////////////////////////////
        //
        // Multi-channel WMA properties
        //
        public const string g_wszPeakValue = "PeakValue";
        public const string g_wszAverageLevel = "AverageLevel";
        public const string g_wszFold6To2Channels3 = "Fold6To2Channels3";
        public const string g_wszFoldToChannelsTemplate = "Fold%luTo%luChannels%lu";

        ////////////////////////////////////////////////////////////////
        //
        // Complexity profile description strings
        //
        public const string g_wszDeviceConformanceTemplate = "DeviceConformanceTemplate";

        ////////////////////////////////////////////////////////////////
        //
        // Frame interpolation on video decode
        //
        public const string g_wszEnableFrameInterpolation = "EnableFrameInterpolation";

        ////////////////////////////////////////////////////////////////
        //
        // Needs previous sample for Delta frame on video decode
        //
        public const string g_wszNeedsPreviousSample = "NeedsPreviousSample";

        public const string g_wszWMDRM_ACTIONLIST_TAG = "ACTIONLIST";
        public const string g_wszWMDRM_ACTION_TAG = "ACTION";
        public const string g_wszWMDRM_RIGHT_PLAYBACK = "Play";
        public const string g_wszWMDRM_RIGHT_COPY = "Copy";
        public const string g_wszWMDRM_RIGHT_PLAYLIST_BURN = "PlaylistBurn";
        public const string g_wszWMDRM_RIGHT_CREATE_THUMBNAIL_IMAGE = "CreateThumbnailImage";
        public const string g_wszWMDRM_RIGHT_COPY_TO_CD = "Print.redbook";
        public const string g_wszWMDRM_RIGHT_COPY_TO_SDMI_DEVICE = "Transfer.SDMI";
        public const string g_wszWMDRM_RIGHT_COPY_TO_NON_SDMI_DEVICE = "Transfer.NONSDMI";
        public const string g_wszWMDRM_RIGHT_BACKUP = "Backup";
        public const string g_wszWMDRM_RIGHT_COLLABORATIVE_PLAY = "CollaborativePlay";
        public const string g_wszWMDRM_ActionAllowed = "ActionAllowed.";
        public const string g_wszWMDRM_ActionAllowed_Playback = "ActionAllowed.Play";
        public const string g_wszWMDRM_ActionAllowed_Copy = "ActionAllowed.Copy";
        public const string g_wszWMDRM_ActionAllowed_PlaylistBurn = "ActionAllowed.PlaylistBurn";
        public const string g_wszWMDRM_ActionAllowed_CreateThumbnailImage = "ActionAllowed.CreateThumbnailImage";
        public const string g_wszWMDRM_ActionAllowed_CopyToCD = "ActionAllowed.Print.redbook";
        public const string g_wszWMDRM_ActionAllowed_CopyToSDMIDevice = "ActionAllowed.Transfer.SDMI";
        public const string g_wszWMDRM_ActionAllowed_CopyToNonSDMIDevice = "ActionAllowed.Transfer.NONSDMI";
        public const string g_wszWMDRM_ActionAllowed_Backup = "ActionAllowed.Backup";
        public const string g_wszWMDRM_ActionAllowed_CollaborativePlay = "ActionAllowed.CollaborativePlay";
        public const string g_wszWMDRM_LicenseState = "LicenseStateData.";
        public const string g_wszWMDRM_LicenseState_Playback = "LicenseStateData.Play";
        public const string g_wszWMDRM_LicenseState_Copy = "LicenseStateData.Copy";
        public const string g_wszWMDRM_LicenseState_PlaylistBurn = "LicenseStateData.PlaylistBurn";
        public const string g_wszWMDRM_LicenseState_CreateThumbnailImage = "LicenseStateData.CreateThumbnailImage";
        public const string g_wszWMDRM_LicenseState_CopyToCD = "LicenseStateData.Print.redbook";
        public const string g_wszWMDRM_LicenseState_CopyToSDMIDevice = "LicenseStateData.Transfer.SDMI";
        public const string g_wszWMDRM_LicenseState_CopyToNonSDMIDevice = "LicenseStateData.Transfer.NONSDMI";
        public const string g_wszWMDRM_LicenseState_Backup = "LicenseStateData.Backup";
        public const string g_wszWMDRM_LicenseState_CollaborativePlay = "LicenseStateData.CollaborativePlay";
        public const string g_wszWMDRMNET_Revocation = "WMDRMNET_REVOCATION";
        public const string g_wszWMDRM_SAPLEVEL = "SAPLEVEL";
        public const string g_wszWMDRM_SAPRequired = "SAPRequired";
        public const string g_wszWMDRM_PRIORITY = "PRIORITY";
        public const string g_wszWMDRM_ISSUEDATE = "ISSUEDATE";
        public const string g_wszWMDRM_UplinkID = "UplinkID";
        public const string g_wszWMDRM_REVINFOVERSION = "REVINFOVERSION";

        public const string g_wszWMDRM_IsDRM = "IsDRM";
        public const string g_wszWMDRM_IsDRMCached = "IsDRMCached";
        public const string g_wszWMDRM_BaseLicenseAcqURL = "BaseLAURL";
        public const string g_wszWMDRM_Rights = "Rights";
        public const string g_wszWMDRM_LicenseID = "LID";
        public const string g_wszWMDRM_DRMHeader = "DRMHeader.";
        public const string g_wszWMDRM_DRMHeader_KeyID = "DRMHeader.KID";
        public const string g_wszWMDRM_DRMHeader_LicenseAcqURL = "DRMHeader.LAINFO";
        public const string g_wszWMDRM_DRMHeader_ContentID = "DRMHeader.CID";
        public const string g_wszWMDRM_DRMHeader_IndividualizedVersion = "DRMHeader.SECURITYVERSION";
        public const string g_wszWMDRM_DRMHeader_ContentDistributor = "DRMHeader.ContentDistributor";
        public const string g_wszWMDRM_DRMHeader_SubscriptionContentID = "DRMHeader.SubscriptionContentID";
    }

    [StructLayout(LayoutKind.Sequential)]
    public class WmShort
    {
        private short m_value;

        public WmShort()
        {
            m_value = 0;
        }

        public WmShort(short v)
        {
            m_value = v;
        }

        public override string ToString()
        {
            return m_value.ToString();
        }

        public override int GetHashCode()
        {
            return m_value.GetHashCode();
        }

        public static implicit operator short(WmShort l)
        {
            return l.m_value;
        }

        public static implicit operator WmShort(short l)
        {
            return new WmShort(l);
        }

        public short ToInt16()
        {
            return m_value;
        }

        public void Assign(short f)
        {
            m_value = f;
        }

        public static WmShort FromInt16(short l)
        {
            return new WmShort(l);
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public class WmInt
    {
        private int m_value;

        public WmInt()
        {
            m_value = 0;
        }

        public WmInt(int v)
        {
            m_value = v;
        }

        public override string ToString()
        {
            return m_value.ToString();
        }

        public override int GetHashCode()
        {
            return m_value.GetHashCode();
        }

        public static implicit operator int(WmInt l)
        {
            return l.m_value;
        }

        public static implicit operator WmInt(int l)
        {
            return new WmInt(l);
        }

        public int ToInt32()
        {
            return m_value;
        }

        public void Assign(int f)
        {
            m_value = f;
        }

        public static WmInt FromInt32(int l)
        {
            return new WmInt(l);
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    public class FourCC
    {
        protected const string m_SubTypeExtension = "-0000-0010-8000-00aa00389b71";
        private int m_fourCC;

        public FourCC(string fcc)
        {
            if (fcc.Length != 4)
            {
                throw new ArgumentException(fcc + " is not a valid FourCC");
            }

            byte[] asc = Encoding.ASCII.GetBytes(fcc);

            LoadFromBytes(asc[0], asc[1], asc[2], asc[3]);
        }

        public FourCC(char a, char b, char c, char d)
            : this(new string(new char[] { a, b, c, d }))
        { }

        public FourCC(int fcc)
        {
            m_fourCC = fcc;
        }

        public FourCC(byte[] b)
        {
            if (b.Length != 4)
            {
                throw new Exception("Invalid byte array passed to FourCC");
            }

            LoadFromBytes(b[0], b[1], b[2], b[3]);
        }

        public FourCC(byte a, byte b, byte c, byte d)
        {
            LoadFromBytes(a, b, c, d);
        }

        public FourCC(Guid g)
        {
            if (!IsA4ccSubtype(g))
            {
                throw new Exception("Not a FourCC Guid");
            }

            byte[] asc;
            asc = g.ToByteArray();

            LoadFromBytes(asc[0], asc[1], asc[2], asc[3]);
        }

        public void LoadFromBytes(byte a, byte b, byte c, byte d)
        {
            m_fourCC = a;
            m_fourCC |= b << 8;
            m_fourCC |= c << 16;
            m_fourCC |= d << 24;
        }

        public int ToInt32()
        {
            return m_fourCC;
        }

        public byte[] GetBytes()
        {
            byte[] b = new byte[4];

            b[0] = (byte)(m_fourCC & 0xff);
            b[1] = (byte)((m_fourCC >> 8) & 0xff);
            b[2] = (byte)((m_fourCC >> 16) & 0xff);
            b[3] = (byte)((m_fourCC >> 24) & 0xff);

            return b;
        }

        public static explicit operator int(FourCC f)
        {
            return f.ToInt32();
        }

        public Guid ToMediaSubtype()
        {
            return new Guid(m_fourCC.ToString("X") + m_SubTypeExtension);
        }

        public static bool operator ==(FourCC fcc1, FourCC fcc2)
        {
            // If both are null, or both are same instance, return true.
            if (Object.ReferenceEquals(fcc1, fcc2))
            {
                return true;
            }

            // If one is null, but not both, return false.
            if (((object)fcc1 == null) || ((object)fcc2 == null))
            {
                return false;
            }

            return fcc1.m_fourCC == fcc2.m_fourCC;
        }

        public static bool operator !=(FourCC fcc1, FourCC fcc2)
        {
            return !(fcc1 == fcc2);
        }

        public override bool Equals(object obj)
        {
            if (!(obj is FourCC))
                return false;

            return (obj as FourCC).m_fourCC == m_fourCC;
        }

        public override int GetHashCode()
        {
            return m_fourCC.GetHashCode();
        }

        public override string ToString()
        {
            char c;
            char[] ca = new char[4];

            c = Convert.ToChar(m_fourCC & 255);
            if (!Char.IsLetterOrDigit(c))
            {
                c = ' ';
            }
            ca[0] = c;

            c = Convert.ToChar((m_fourCC >> 8) & 255);
            if (!Char.IsLetterOrDigit(c))
            {
                c = ' ';
            }
            ca[1] = c;

            c = Convert.ToChar((m_fourCC >> 16) & 255);
            if (!Char.IsLetterOrDigit(c))
            {
                c = ' ';
            }
            ca[2] = c;

            c = Convert.ToChar((m_fourCC >> 24) & 255);
            if (!Char.IsLetterOrDigit(c))
            {
                c = ' ';
            }
            ca[3] = c;

            string s = new string(ca);

            return s;
        }

        public static bool IsA4ccSubtype(Guid g)
        {
            return (g.ToString().Contains(m_SubTypeExtension));
        }
    }

    #region Internal

    // These classes are used internally and there is probably no reason you will ever
    // need to use them directly.

    internal class MTMarshaler : ICustomMarshaler
    {
        [DllImport("Kernel32.dll", EntryPoint = "RtlMoveMemory"), SuppressUnmanagedCodeSecurity]
        private static extern void CopyMemory(IntPtr Destination, IntPtr Source, int Length);

        protected AMMediaType m_mt;

        public IntPtr MarshalManagedToNative(object managedObj)
        {
            m_mt = managedObj as AMMediaType;

            IntPtr ip = Marshal.AllocCoTaskMem(Marshal.SizeOf(m_mt) + m_mt.formatSize);

            // This class is only used for output.  No need to burn the cpu cycles to copy
            // over data that just gets overwritten.

            //Marshal.StructureToPtr(m_mt, ip, false);

            //if ((m_mt.formatSize > 0) && (m_mt.formatPtr != IntPtr.Zero))
            //{
            //    CopyMemory(new IntPtr(ip.ToInt64() + Marshal.SizeOf(m_mt)), m_mt.formatPtr, m_mt.formatSize);
            //}

            return ip;
        }

        // Called just after invoking the COM method.  The IntPtr is the same one that just got returned
        // from MarshalManagedToNative.  The return value is unused.
        public object MarshalNativeToManaged(IntPtr pNativeData)
        {
            if (m_mt == null)
            {
                m_mt = new AMMediaType();
            }
            Marshal.PtrToStructure(pNativeData, m_mt);

            if (m_mt.formatSize > 0)
            {
                IntPtr ip = m_mt.formatPtr;

                m_mt.formatPtr = Marshal.AllocCoTaskMem(m_mt.formatSize);
                CopyMemory(m_mt.formatPtr, ip, m_mt.formatSize);
            }

            return m_mt;
        }

        // It appears this routine is never called
        public void CleanUpManagedData(object ManagedObj)
        {
            m_mt = null;
        }

        public void CleanUpNativeData(IntPtr pNativeData)
        {
            Marshal.FreeCoTaskMem(pNativeData);
        }

        // The number of bytes to marshal out - never called
        public int GetNativeDataSize()
        {
            return -1;
        }

        // This method is called by interop to create the custom marshaler.  The (optional)
        // cookie is the value specified in MarshalCookie="asdf", or "" is none is specified.
        public static ICustomMarshaler GetInstance(string cookie)
        {
            return new MTMarshaler();
        }
    }


    #endregion
}
