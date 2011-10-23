//
// BwgBurn - CD-R/CD-RW/DVD-R/DVD-RW burning program for DotNet
// 
// Copyright (C) 2006 by Jack W. Griffin (butchg@comcast.net)
//
// This program is free software; you can redistribute it and/or modify 
// it under the terms of the GNU General Public License as published by 
// the Free Software Foundation; either version 2 of the License, or 
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but 
// WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY 
// or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License 
// for more details.
//
// You should have received a copy of the GNU General Public License along 
// with this program; if not, write to the 
//
// Free Software Foundation, Inc., 
// 59 Temple Place, Suite 330, 
// Boston, MA 02111-1307 USA
//

using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime;
using System.Runtime.InteropServices;
using System.IO;
using System.Diagnostics;
using Bwg.Logging;

namespace Bwg.Scsi
{
    //
    // Functions not yet supported:
    //     MechanismStatus()
    //     ReadBuffer()
    //     ReportKey()
    //     WriteVerify()
    //     WriteBuffer()

    /// <summary>
    /// 
    /// </summary>
    public unsafe class Device : WinDev
    {
        #region Structures for DeviceIoControl
        [StructLayout(LayoutKind.Explicit)]
        struct SCSI_PASS_THROUGH_DIRECT32
        {
            [FieldOffset(0)]public ushort Length;
            [FieldOffset(2)]public byte ScsiStatus;
            [FieldOffset(3)]public byte PathId;
            [FieldOffset(4)]public byte TargetId;
            [FieldOffset(5)]public byte Lun;
            [FieldOffset(6)]public byte CdbLength;
            [FieldOffset(7)]public byte SenseInfoLength;
            [FieldOffset(8)]public byte DataIn;
            [FieldOffset(12)]public uint DataTransferLength;
            [FieldOffset(16)]public uint TimeOutValue;
            [FieldOffset(20)]public IntPtr DataBuffer;
            [FieldOffset(24)]public uint SenseInfoOffset;
            [FieldOffset(28)]public fixed byte CdbData[16];
            [FieldOffset(48)]public fixed byte SenseInfo[32];
        }  ;

        [StructLayout(LayoutKind.Explicit)]
        struct SCSI_PASS_THROUGH_DIRECT64
        {
            [FieldOffset(0)]public ushort Length;
            [FieldOffset(2)]public byte ScsiStatus;
            [FieldOffset(3)]public byte PathId;
            [FieldOffset(4)]public byte TargetId;
            [FieldOffset(5)]public byte Lun;
            [FieldOffset(6)]public byte CdbLength;
            [FieldOffset(7)]public byte SenseInfoLength;
            [FieldOffset(8)]public byte DataIn;
            [FieldOffset(12)]public uint DataTransferLength;
            [FieldOffset(16)]public uint TimeOutValue;
            [FieldOffset(24)]public IntPtr DataBuffer;
            [FieldOffset(32)]public uint SenseInfoOffset;
            [FieldOffset(36)]public fixed byte CdbData[16];
            [FieldOffset(56)]public fixed byte SenseInfo[32];
        }  ;

        [StructLayout(LayoutKind.Explicit)]
        struct IO_SCSI_CAPABILITIES
        {
            [FieldOffset(0)]public uint Length;
            [FieldOffset(4)]public uint MaximumTransferLength;
            [FieldOffset(8)]public uint MaximumPhysicalPages;
            [FieldOffset(12)]public uint SupportedAsynchronousEvents;
            [FieldOffset(16)]public uint AlignmentMask;
            [FieldOffset(20)]public byte TaggedQueuing;
            [FieldOffset(21)]public byte AdapterScansDown ;
            [FieldOffset(22)]public byte AdapterUsesPio;
        } ;

        #endregion

        #region Private data members
        /// <summary>
        /// The bitsize of the OS
        /// </summary>
        private byte m_ossize;
        private bool m_ignore_long_write_error;
        private byte[] m_sense_info;
        private byte m_scsi_status;
        private Logger m_logger;
        private int m_MaximumTransferLength;
        #endregion

        #region private static data structures
        static ushort m_scsi_request_size_32 = 44;
        static ushort m_scsi_request_size_64 = 56;
        #endregion

        #region public constants
        /// <summary>
        /// This constant is the device code for an MMC device, which is a CDROM/DVD/BD/HD DVD drive of
        /// some type.
        /// </summary>
        public const int MMCDeviceType = 5;
        #endregion

        #region Private constants
        private const uint IOCTL_SCSI_PASS_THROUGH_DIRECT = 0x4d014;
        private const uint IOCTL_SCSI_GET_CAPABILITIES = 0x41010;

        private const uint ERROR_NOT_SUPPORTED = 50 ;
        #endregion

        #region Public Types

        /// <summary>
        /// This value can be used in the call to SetCDSpeed() to set the read or
        /// write speed to an optimal value.
        /// </summary>
        public const ushort OptimumSpeed = 0xffff;

        /// <summary>
        /// The track number for the "invisible" track which is the next track to be
        /// written on the device.
        /// </summary>
        public const byte InvisibleTrack = 0xff;

        /// <summary>
        /// The notification class indicating the type of events of interest
        /// </summary>
        [Flags]
        public enum NotificationClass : byte
        {
            /// <summary>
            /// 
            /// </summary>
            Reserved1 = (1 << 0),

            /// <summary>
            /// See SCSI Spec
            /// </summary>
            OperationalChange = (1 << 1),

            /// <summary>
            /// See SCSI Spec
            /// </summary>
            PowerManagement = (1 << 2),

            /// <summary>
            /// See SCSI Spec
            /// </summary>
            ExternalRequest = (1 << 3),

            /// <summary>
            /// See SCSI Spec
            /// </summary>
            Media = (1 << 4),

            /// <summary>
            /// See SCSI Spec
            /// </summary>
            MultiHost = (1 << 5),

            /// <summary>
            /// See SCSI Spec
            /// </summary>
            DeviceBusy = (1 << 6),

            /// <summary>
            /// See SCSI Spec
            /// </summary>
            Reserved2 = (1 << 7)
        } ;

        /// <summary>
        /// The close sessions/track type
        /// </summary>
        public enum CloseTrackSessionType : byte
        {
            /// <summary>
            /// Stop background format on DVD+RW
            /// </summary>
            StopBackGroundFormat = 0,

            /// <summary>
            /// Close a logical track
            /// </summary>
            CloseTrack = 1,

            /// <summary>
            /// Close a sessions (finalize a CD)
            /// </summary>
            CloseSession = 2,

            /// <summary>
            /// Finalize a disk
            /// </summary>
            FinalizeDisk = 3,

            /// <summary>
            /// Close a DVD+RW session with minimal radius
            /// </summary>
            CloseSessionMinimalRadius = 4,

            /// <summary>
            /// Finalize a disk with minimul radius
            /// </summary>
            FinalizeDiskWithMinimalRadius = 5,

            /// <summary>
            /// Finalize a disk in the most compatible way
            /// </summary>
            FinalizeDiskCompatible = 6
        } ;

        /// <summary>
        /// The track/session number type
        /// </summary>
        public enum ReadTrackType
        {
            /// <summary>
            /// Select the track containing the given logical block
            /// </summary>
            LBA = 0,

            /// <summary>
            /// Select the track given by a track number
            /// </summary>
            TrackNumber = 1,

            /// <summary>
            /// Select the first track of the given session number
            /// </summary>
            SessionNumber = 2,

            /// <summary>
            /// Reserved, do not use
            /// </summary>
            Reserved = 3
        } ;

        /// <summary>
        /// The sense key for the sense information
        /// </summary>
        public enum SenseKeyType
        {
            /// <summary>
            /// No sense information returned
            /// </summary>
            NoSense = 0,

            /// <summary>
            /// The last command completed sucessfully, with some recovery operation involved
            /// </summary>
            RecoveredError = 1 ,

            /// <summary>
            /// The device is not ready
            /// </summary>
            NotReady = 2,

            /// <summary>
            /// An error in the media or an error in the data
            /// </summary>
            MediumError = 3,

            /// <summary>
            /// A non-recoverable hardware error occurred
            /// </summary>
            HardwareError = 4,

            /// <summary>
            /// The request sent was not valid
            /// </summary>
            IllegalRequest = 5,

            /// <summary>
            /// Removable media was removed, or some other unit attention error
            /// </summary>
            UnitAttention = 6,

            /// <summary>
            /// A command that reads or writes the media was performed on a protected block
            /// </summary>
            DataProtect = 7,

            /// <summary>
            /// A write once media encountered a non-blank media
            /// </summary>
            BlankCheck = 8,

            /// <summary>
            /// Vendor specific sense information
            /// </summary>
            VendorSpecific = 9
        } ;

        /// <summary>
        /// Return value from ScsiDevice methods indicating the status of the command
        /// </summary>
        public enum CommandStatus
        {
            /// <summary>
            /// The method is not yet supported by this class
            /// </summary>
            NotSupported,

            /// <summary>
            /// The window IOCTL failed.  The property LastError can be queried to get the Win32 error
            /// associated with the failure.
            /// </summary>
            IoctlFailed,

            /// <summary>
            /// The SCSI device failed the result.  The sense information can be queried to determine more
            /// about why the method failed.
            /// </summary>
            DeviceFailed,

            /// <summary>
            /// The method was sucessful
            /// </summary>
            Success
        } ;

        /// <summary>
        /// Type of blanking operations that can be performed
        /// </summary>
        public enum BlankType
        {
            /// <summary>
            /// Blank the entire disk, perfoming a full blanking operation
            /// </summary>
            FullDisk = 0x00,

            /// <summary>
            /// Blank the disk, blanking only those blocks required to indicate the disk is blank
            /// </summary>
            MinimalDisk = 0x01,

            /// <summary>
            /// Blank only the track given.  Not valid for DVD-RW media
            /// </summary>
            Track = 0x02,

            /// <summary>
            /// Unreserve a reserved track track
            /// </summary>
            UnreserveTrack = 0x03,

            /// <summary>
            /// Blank the track tail
            /// </summary>
            TrackTail = 0x04,

            /// <summary>
            /// Unclose the session allowing data to be appended to the session
            /// </summary>
            UncloseSession = 0x05,

            /// <summary>
            /// Blank the last session
            /// </summary>
            Session = 0x06
        } ;

        /// <summary>
        /// Indicates the type of configuration to retreive from the device
        /// </summary>
        public enum GetConfigType
        {
            /// <summary>
            /// Return information about all features associated with the device
            /// </summary>
            AllFeatures = 0,

            /// <summary>
            /// Return information about only those features marked as current
            /// </summary>
            CurrentFeatures = 1,

            /// <summary>
            /// Return information about one specific feature
            /// </summary>
            OneFeature = 2,

            /// <summary>
            /// This value is reserved.  Using this value will cause an exception to be thrown.
            /// </summary>
            Reserved = 3
        } ;

        /// <summary>
        /// The type of load/unload action to take
        /// </summary>
        public enum LoadUnload
        {
            /// <summary>
            /// Abort any ongoing load or unload operation
            /// </summary>
            Abort = 0,

            /// <summary>
            /// Reserved, do not use.  Using this as an argument will result in a exception
            /// </summary>
            Reserved = 1,

            /// <summary>
            /// Load the media from a specific slot, or from the tray
            /// </summary>
            Load = 2,

            /// <summary>
            /// Unload the media from the device
            /// </summary>
            Unload = 3              // Unload the media
        } ;

        /// <summary>
        /// 
        /// </summary>
        public enum PageControl
        {
            /// <summary>
            /// Return the mode page parameters currently active in the device
            /// </summary>
            Current = 0,

            /// <summary>
            /// Returns a mask indicating which parameters are changable
            /// </summary>
            Changeable = 1,

            /// <summary>
            /// Return the default mode page parameters for the requested page
            /// </summary>
            Default = 2,

            /// <summary>
            /// Returns the saved mode page parameters for the requested page
            /// </summary>
            Saved = 3
        } ;

        /// <summary>
        /// 
        /// </summary>
        public enum PauseResumeAction
        {
            /// <summary>
            /// Pause the CD/DVD drive if it is playing
            /// </summary>
            Pause,

            /// <summary>
            /// Resume the CD/DVD drive if it is paused
            /// </summary>
            Resume
        } ;

        /// <summary>
        /// This type indicates the state of the drive with respect to removing the media.
        /// </summary>
        public enum PreventAllow
        {
            /// <summary>
            /// Allow the media to be removed from the device
            /// </summary>
            Allow = 0,

            /// <summary>
            /// Prevent the media from being removed from the device
            /// </summary>
            Prevent = 1,

            /// <summary>
            /// Allow the media to be removed from the device (not sure what presistent means)
            /// </summary>
            PresistentAllow = 2,

            /// <summary>
            /// Prevent the media from being removed from the device (not sure what presistent means)
            /// </summary>
            PresistentPrevent = 3
        } ;

        /// <summary>
        /// Direction to perform the scan operation
        /// </summary>
        public enum ScanDirection
        {
            /// <summary>
            /// Scan in the forward direction
            /// </summary>
            Forward,

            /// <summary>
            /// Scan in the reverse direction
            /// </summary>
            Reverse
        } ;

        /// <summary>
        /// The type of scan operation
        /// </summary>
        public enum ScanType
        {
            /// <summary>
            /// Logical block address can
            /// </summary>
            LogicalBlockAddress = 0,

            /// <summary>
            /// Time based scan
            /// </summary>
            Time = 1,

            /// <summary>
            /// Track number scan
            /// </summary>
            TrackNumber = 2,

            /// <summary>
            /// Reserved, will cause an excpetion if used
            /// </summary>
            Reserved = 3
        } ;

        /// <summary>
        /// This type gives the rotational contorl value when setting the speed of the
        /// CD player
        /// </summary>
        public enum RotationalControl
        {
            /// <summary>
            /// CLV and non-pure CAV
            /// </summary>
            CLVandNonPureCav = 0,

            /// <summary>
            /// Pure Cav
            /// </summary>
            PureCav = 1,
        } ;

        /// <summary>
        /// The power control value used when starting, stopping the drive
        /// </summary>
        public enum PowerControl
        {
            /// <summary>
            /// Do not change the current power setting
            /// </summary>
            NoChange = 0,

            /// <summary>
            /// Put player into idle state, reset the standby timer
            /// </summary>
            IdleState = 2,

            /// <summary>
            /// Place the drive in the standby state
            /// </summary>
            StandbyState = 3,

            /// <summary>
            /// Place the drive in the sleep state
            /// </summary>
            SleepState = 5
        } ;

        /// <summary>
        /// The start/stop state of the drive
        /// </summary>
        public enum StartState
        {
            /// <summary>
            /// Stop the disk
            /// </summary>
            StopDisk = 0,

            /// <summary>
            /// Start the disk
            /// </summary>
            StartDisk = 1,

            /// <summary>
            /// Eject the disk
            /// </summary>
            EjectDisk = 2,

            /// <summary>
            /// Load the disk
            /// </summary>
            LoadDisk = 3
        } ;

        #endregion

        #region Public Properties

        /// <summary>
        /// This property, if true, supresses any sense error messages about long writes
        /// in progress.  This is done because this error is a normal part of the burn
        /// process
        /// </summary>
        public bool DontDisplayIgnoreLongWriteInProgress
        {
            get
            {
                return m_ignore_long_write_error;
            }
            set
            {
                m_ignore_long_write_error = value;
            }
        }

        /// <summary>
        /// Returns TRUE if the sense information contains information about the
        /// progress of the current operation
        /// </summary>
        public bool HasSenseProgressInformation
        {
            get
            {
                if (m_sense_info == null)
                    return false;

                return (m_sense_info[15] & 0x80) != 0;
            }
        }

        /// <summary>
        /// The raw progress information from the sense information
        /// </summary>
        public ushort ProgressInformation
        {
            get
            {
                return (ushort)(m_sense_info[16] << 8 | m_sense_info[17]);
            }
        }

        /// <summary>
        /// The maximum transfer length for this SCSI device on this SCSI channel
        /// </summary>
        public int MaximumTransferLength 
        { 
            get 
            { 
                return m_MaximumTransferLength; 
            } 
        }

        /// <summary>
        /// This property returns true if the current sense error is a write protect error
        /// </summary>
        public bool IsWriteProtectError
        {
            get { return this.GetSenseAsc() == 0x27 && this.GetSenseAscq() == 0; }
        }

        /// <summary>
        /// If TRUE, log sense state
        /// </summary>
        public bool LogSenseState;

        #endregion

        #region Private Functions

        /// <summary>
        /// Format error message using sense code.
        /// </summary>
        /// <returns></returns>
		public string GetErrorString()
		{
			string res = messages.GetString(string.Format("SCSISenseKey_{0:X2}", (int)this.GetSenseKey())) ?? this.GetSenseKey().ToString();
			if (this.GetSenseKey() == SenseKeyType.NoSense)
				return res;
			return res + ": " + Device.LookupSenseError(this.GetSenseAsc(), this.GetSenseAscq());
		}

        /// <summary>
        /// Return a string associated with the ASC/ASCQ bytes
        /// </summary>
        /// <param name="asc">the sense ASC value, see the SCSI spec</param>
        /// <param name="ascq">the sense ASCQ value, see the SCSI spec</param>
        /// <returns>a string representing the error codes given by ASC and ASCQ</returns>
        public static string LookupSenseError(byte asc, byte ascq)
        {
			string res = messages.GetString(string.Format("SCSIErrorMessage_{0:X2}{1:X2}", asc, ascq));
			if (res != null)
				return res;
			string msg = messages.GetString("UnknownSCSIError") ?? "Unknown SCSI Error";
            return msg + " ASC=" + asc.ToString("X2") + ", ASCQ=" + ascq.ToString("X2");
        }

        /// <summary>
        /// Send the sense information to the logger
        /// </summary>
        private void LogSenseInformation(Command cmd)
        {
            if (m_ignore_long_write_error && IsLongWriteInProgress(GetSenseAsc(), GetSenseAscq()))
                return;

            if (m_logger != null && LogSenseState)
            {
                int len = GetSenseLength();
                int offset = 0;
                int line = 0;
                UserMessage m;
                string str;
                Logger logger = GetLogger();

                str = "SCSI Operation Failed: " ;
                str += LookupSenseError(GetSenseAsc(), GetSenseAscq()) ;
                m = new UserMessage(UserMessage.Category.Error, 0, str);
                m_logger.LogMessage(m);

                str = "    SenseKey = " + GetSenseKey().ToString("X");
                m = new UserMessage(UserMessage.Category.Error, 0, str);
                m_logger.LogMessage(m);

                str = "    SenseAsc = " + GetSenseAsc().ToString("X");
                m = new UserMessage(UserMessage.Category.Error, 0, str);
                m_logger.LogMessage(m);

                str = "    SenseAscq = " + GetSenseAscq().ToString("X");
                m = new UserMessage(UserMessage.Category.Error, 0, str);
                m_logger.LogMessage(m);

                while (offset < len)
                {
                    line = 0;
                    str = "    SenseData:";
                    while (offset < len && line < 8)
                    {
                        str += " " + GetSenseByte(offset++).ToString("X2");
                        line++;
                    }

                    m = new UserMessage(UserMessage.Category.Error, 0, str);
                    logger.LogMessage(m);
                }

                if (GetSenseAsc() == 0x24 && GetSenseAscq() == 0x00)
                {
                    // This is invalid field in CDB, so dump the CDB to the log
                    str = "INVALID CDB:";
                    for (byte i = 0; i < cmd.GetCDBLength(); i++)
                    {
                        byte b = cmd.GetCDB(i);
                        str += " " + b.ToString("X");
                    }
                    m = new UserMessage(UserMessage.Category.Error, 0, str);
                    m_logger.LogMessage(m);
                }
            }
        }

        private CommandStatus SendCommand(Command cmd)
        {
            return (m_ossize == 32) ? SendCommand32(cmd) : SendCommand64(cmd);
        }

        private CommandStatus SendCommand64(Command cmd)
        {
            SCSI_PASS_THROUGH_DIRECT64 f = new SCSI_PASS_THROUGH_DIRECT64();
            f.Length = m_scsi_request_size_64;
            f.CdbLength = (byte)cmd.GetCDBLength();
            f.DataIn = 0;
            if (cmd.Direction == Command.CmdDirection.In)
                f.DataIn = 1;
            f.DataTransferLength = (uint)cmd.BufferSize;
            f.TimeOutValue = (uint)cmd.TimeOut;
            for (byte i = 0; i < f.CdbLength; i++)
                f.CdbData[i] = cmd.GetCDB(i);
            f.SenseInfoOffset = 56;
            f.SenseInfoLength = 24;

            uint total = (uint)Marshal.SizeOf(f);

            uint ret = 0;

            // Set the buffer field
            f.DataBuffer = cmd.GetBuffer();

            // Send through ioctl field
            IntPtr pt = new IntPtr(&f);
            if (!Control(IOCTL_SCSI_PASS_THROUGH_DIRECT, pt, total, pt, total, ref ret, IntPtr.Zero))
            {
                string str ;
                str = "IOCTL_SCSI_PASS_THROUGH_DIRECT failed - " + Win32ErrorToString(LastError);
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Error, 0, str));
                return CommandStatus.IoctlFailed;
            }

            m_scsi_status = f.ScsiStatus;
            if (f.SenseInfoLength != 0)
            {
                m_sense_info = new byte[f.SenseInfoLength];
                for (int i = 0; i < f.SenseInfoLength; i++)
                    m_sense_info[i] = f.SenseInfo[i];
            }
            else
                m_sense_info = null;

            if (m_scsi_status != 0)
            {
                LogSenseInformation(cmd);
                return CommandStatus.DeviceFailed;
            }

            return CommandStatus.Success;
        }

        private CommandStatus SendCommand32(Command cmd)
        {
            SCSI_PASS_THROUGH_DIRECT32 f = new SCSI_PASS_THROUGH_DIRECT32();
            f.Length = m_scsi_request_size_32;
            f.CdbLength = (byte)cmd.GetCDBLength();
            f.DataIn = 0;
            if (cmd.Direction == Command.CmdDirection.In)
                f.DataIn = 1;
            f.DataTransferLength = (uint)cmd.BufferSize;
            f.TimeOutValue = (uint)cmd.TimeOut;
            for (byte i = 0; i < f.CdbLength; i++)
                f.CdbData[i] = cmd.GetCDB(i);
            f.SenseInfoOffset = 48;
            f.SenseInfoLength = 24;

            uint total = (uint)Marshal.SizeOf(f);

            uint ret = 0;

            // Set the buffer field
            f.DataBuffer = cmd.GetBuffer();

            // Send through ioctl field
            IntPtr pt = new IntPtr(&f);
            if (!Control(IOCTL_SCSI_PASS_THROUGH_DIRECT, pt, total, pt, total, ref ret, IntPtr.Zero))
            {
                string str ;
                str = "IOCTL_SCSI_PASS_THROUGH_DIRECT failed - " + Win32ErrorToString(LastError);
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Error, 0, str));
                return CommandStatus.IoctlFailed;
            }

            m_scsi_status = f.ScsiStatus;
            if (f.SenseInfoLength != 0)
            {
                m_sense_info = new byte[f.SenseInfoLength];
                for (int i = 0; i < f.SenseInfoLength; i++)
                    m_sense_info[i] = f.SenseInfo[i];
            }
            else
                m_sense_info = null;

            if (m_scsi_status != 0)
            {
                LogSenseInformation(cmd);
                return CommandStatus.DeviceFailed;
            }

            return CommandStatus.Success;
        }

        void QueryBufferSize()
        {
            IO_SCSI_CAPABILITIES f = new IO_SCSI_CAPABILITIES();
            f.Length = 23;

            uint ret = 0;
            IntPtr pt = new IntPtr(&f);
            uint total = (uint)Marshal.SizeOf(f);
            if (Control(IOCTL_SCSI_GET_CAPABILITIES, pt, total, pt, total, ref ret, IntPtr.Zero))
            {
                if (f.MaximumTransferLength > Int32.MaxValue)
                    m_MaximumTransferLength = Int32.MaxValue;
                else
                    m_MaximumTransferLength = (int)f.MaximumTransferLength;

                if (m_MaximumTransferLength > 56 * 2048)
                   m_MaximumTransferLength = 56 * 2048;
            }
            else
            {
                //
                // This is hardcoded for now because it works, until I can figure out
                // how to query devices for the maximum amount of data that can be transferred
                // at one time.
                //
                m_MaximumTransferLength = 26 * 2048;
            }
        }

        #endregion

		static global::System.Resources.ResourceManager messages;

        #region constructor
        /// <summary>
        /// 
        /// </summary>
        public Device(Logger l)
        {
            m_logger = l;
            LogSenseState = true;

            IntPtr p = new IntPtr();
            if (Marshal.SizeOf(p) == 4)
                m_ossize = 32;
            else
                m_ossize = 64;

            // m_logger.LogMessage(new UserMessage(UserMessage.Category.Info, 0, "Operating System Size: " + m_ossize.ToString()));
        }

        static Device()
        {
			messages = new global::System.Resources.ResourceManager("Bwg.Scsi.Messages", typeof(Device).Assembly);
        }

        #endregion

        #region Public Functions

        /// <summary>
        /// This static method returns true if the error given by the ASC and ASCQ values
        /// indicate a long write in progress
        /// </summary>
        /// <param name="asc"></param>
        /// <param name="ascq"></param>
        /// <returns></returns>
        public static bool IsLongWriteInProgress(byte asc, byte ascq)
        {
            return asc == 0x04 && ascq == 0x08 ;
        }

        /// <summary>
        /// Open the SCSI device
        /// </summary>
        /// <param name="name">name of the SCSI device</param>
        /// <returns>true if the open succeeded</returns>
        public override bool Open(string name)
        {
            if (!base.Open(name))
                return false;

            QueryBufferSize();

            return true;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="letter"></param>
        /// <returns></returns>
        public override bool Open(char letter)
        {
            if (!base.Open(letter))
                return false ;

            QueryBufferSize();

            return true;
        }

        /// <summary>
        /// This method return TRUE if the current SENSE error is the one given by the
        /// two parameters to the function
        /// </summary>
        /// <param name="asc"></param>
        /// <param name="ascq"></param>
        /// <returns></returns>
        public bool IsScsiError(byte asc, byte ascq)
        {
            return GetSenseAsc() == asc && GetSenseAscq() == ascq;
        }

        /// <summary>
        /// This function resets the device.
        /// </summary>
        /// <returns>true, if the reset was sucessful, false otherwise</returns>
        public bool Reset()
        {
            // 
            // How do I reset a driver that is speaking pass through SCSI?  It might be
            // an IDE drive, SCSI drive, USB drive, or Fireware drive.
            //
            return true;
        }

        /// <summary>
        /// Get the logger from the device
        /// </summary>
        /// <returns>The logger object</returns>
        public Logger GetLogger()
        {
            return m_logger;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public int GetSenseLength()
        {
            if (m_sense_info == null)
                return 0;

            return m_sense_info.GetLength(0);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="addr"></param>
        /// <returns></returns>
        public byte GetSenseByte(int addr)
        {
            return m_sense_info[addr];
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public byte GetScsiStatus() 
        { 
            return m_scsi_status;
        }

        /// <summary>
        /// Get the ASC byte from the sense information
        /// </summary>
        /// <returns>asc byte</returns>
        public byte GetSenseAsc()
        {
            return m_sense_info[12];
        }

        /// <summary>
        /// Get the ASCQ byte from the sense information
        /// </summary>
        /// <returns>ascq byte</returns>
        public byte GetSenseAscq()
        {
            return m_sense_info[13];
        }

        /// <summary>
        /// Get the 32 bits of sense information (offset 3)
        /// </summary>
        /// <returns>sense information</returns>
        public uint GetSenseInformation()
        {
            return (uint)((m_sense_info[3] << 24) | (m_sense_info[4] << 16) | (m_sense_info[5] << 8) | m_sense_info[6]);
        }

        /// <summary>
        /// Get the 32 bits of command specific information (offset 8)
        /// </summary>
        /// <returns>command specific sense information</returns>
        public uint GetSenseCommandSpecificInformation()
        {
            return (uint)((m_sense_info[8] << 24) | (m_sense_info[9] << 16) | (m_sense_info[10] << 8) | m_sense_info[11]);
        }

        /// <summary>
        /// This method returns the sense key for the sense information
        /// </summary>
        /// <returns>The SenseKeyType value that indicates the sense key for the sense information</returns>
        public SenseKeyType GetSenseKey()
        {
            return (SenseKeyType)(m_sense_info[2] & 0x0f);
        }


        #endregion

        #region SCSI Commands

        /// <summary>
        /// This function blanks a portion of the CD or DVD that is in the drive.
        /// </summary>
        /// <param name="immd"></param>If true, this function returns immediately and the blanking
        /// happens in the backgroun.  If false, this funtion does not return until the blanking operation
        /// is complete
        /// <param name="t"></param>The type of blanking operation
        /// <param name="addr"></param>The address for the blanking operation if an address is required
        /// <returns>
        /// Success - the command complete sucessfully
        /// IoctlFailed - the windows DeviceIoControl failed, LastError give the Win32 error code
        /// DeviceFailed - the device failed the command, the sense information has more data
        /// </returns>
        public CommandStatus Blank(bool immd, BlankType t, int addr)
        {
            if (m_logger != null)
            {
                string args = immd.ToString() + ", " + t.ToString() + ", " + addr.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.Blank(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.Blank, 12, 0, Command.CmdDirection.None, 60*30))
            {
                byte b = (byte)t;
                if (immd)
                    b |= (1 << 4);
                cmd.SetCDB8(1, b);

                cmd.SetCDB32(2, addr);
                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// This command allows closure of either a track or a session
        /// </summary>
        /// <param name="immd">If true, return immediately and perform the function in the background</param>
        /// <param name="closetype">This parameter indicates the close type, see the SCSI-3 MMC spec for more information</param>
        /// <param name="track">For those close types requiring a track number, this is the number, see the SCSI-3 MMC spec for more information</param>
        /// <returns></returns>
        public CommandStatus CloseTrackSession(bool immd, CloseTrackSessionType closetype, ushort track)
        {
            if (m_logger != null)
            {
                string args = immd.ToString() + ", 0x" + closetype.ToString() + ", " + track.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.CloseTrackSession(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.CloseTrackSession, 10, 0, Command.CmdDirection.None, 60 * 5))
            {
                if (immd)
                    cmd.SetCDB8(1, 1);

                cmd.SetCDB8(2, (byte)closetype);
                cmd.SetCDB16(4, track) ;

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }
            return CommandStatus.Success;
        }

        /// <summary>
        /// This function instructs the device to erase blocks on the disk.
        /// </summary>
        /// <param name="immd">If true, return immediately and perform the function in the background</param>
        /// <param name="lba">The starting block address for the erase operation</param>
        /// <param name="count">The number of blocks to erase.  If this count is zero, the ERA bit is set in the SCSI erase request</param>
        /// <returns></returns>
        public CommandStatus Erase(bool immd, uint lba, ushort count)
        {
            if (m_logger != null)
            {
                string args = immd.ToString() + ", " + lba.ToString() + ", " + count.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.Erase(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.Erase, 10, 0, Command.CmdDirection.None, 60 * 30))
            {
                byte b = 0;

                if (immd)
                    b |= 0x02;
                if (count == 0)
                    b |= 0x04;
                cmd.SetCDB8(1, b);
                cmd.SetCDB32(2, lba);
                cmd.SetCDB16(7, count);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// This method sends the format unit command down to the SCSI device.
        /// </summary>
        /// <returns></returns>
        public CommandStatus FormatUnit(FormatParameterList plist)
        {
            if (m_logger != null)
            {
                string args = "plist" ;
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.FormatUnit(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.FormatUnit, 6, (ushort)plist.Size, Command.CmdDirection.Out, 60*60))
            {
                cmd.SetCDB8(1, 0x11);

                plist.FormatToMemory(cmd.GetBuffer());

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="type"></param>
        /// <param name="start"></param>
        /// <param name="result"></param>
        /// <returns></returns>
        public CommandStatus GetConfiguration(GetConfigType type, ushort start, out FeatureList result)
        {
            if (m_logger != null)
            {
                string args = type.ToString() + ", " + start.ToString() + ", out result";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.GetConfiguration(" + args + ")"));
            }

            uint len = 0;
            result = null;

            if (type == GetConfigType.Reserved)
                throw new Exception("cannot use reserved value") ;

            using (Command cmd = new Command(ScsiCommandCode.GetConfiguration, 10, 8, Command.CmdDirection.In, 10))
            {
                cmd.SetCDB8(1, (byte)type);
                cmd.SetCDB16(2, start);
                cmd.SetCDB16(7, 8);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                len = cmd.GetBuffer32(0);
                len += 4;       // Add four for the length field
            }

            using (Command cmd = new Command(ScsiCommandCode.GetConfiguration, 10, (ushort)len, Command.CmdDirection.In, 10))
            {
                cmd.SetCDB8(1, (byte)type);
                cmd.SetCDB16(2, start);
                cmd.SetCDB16(7, (ushort)len);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                result = new FeatureList(cmd.GetBuffer(), cmd.BufferSize);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="polled"></param>
        /// <param name="request"></param>
        /// <param name="result"></param>
        /// <returns></returns>
        public CommandStatus GetEventStatusNotification(bool polled, NotificationClass request, out EventStatusNotification result)
        {
            if (m_logger != null)
            {
                string args = polled.ToString() + ", " + request.ToString() + ", out result";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.GetEventStatusNotification(" + args + ")"));
            }

            ushort len = 0;
            result = null;

            using (Command cmd = new Command(ScsiCommandCode.GetEventStatusNotification, 10, 4, Command.CmdDirection.In, 10))
            {
                if (polled)
                    cmd.SetCDB8(1, 1);
                cmd.SetCDB8(4, (byte)request);
                cmd.SetCDB16(7, 4);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                byte n = cmd.GetBuffer8(2);
                if ((n & 0x80) != 0)
                {
                    // There are no events, just capture the header
                    result = new EventStatusNotification(cmd.GetBuffer(), cmd.BufferSize);
                    return CommandStatus.Success;
                }

                //
                // There are event notifications to be grabbed, allocate space for these
                //
                len = cmd.GetBuffer16(0);
                len += 4;               // For the length field
            }

            using (Command cmd = new Command(ScsiCommandCode.GetEventStatusNotification, 10, len, Command.CmdDirection.In, 10))
            {
                if (polled)
                    cmd.SetCDB8(1, 1);
                cmd.SetCDB8(4, (byte)request);
                cmd.SetCDB16(7, len);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                result = new EventStatusNotification(cmd.GetBuffer(), cmd.BufferSize);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus GetSpeed(out SpeedDescriptorList list)
        {
            ushort initial_size = 8 + 4 * 16;
            list = null;

            if (m_logger != null)
            {
                string args = string.Empty;
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.GetWriteSpeed(" + args + ")"));
            }

            uint len = 0;
            using (Command cmd = new Command(ScsiCommandCode.GetPerformance, 12, initial_size, Command.CmdDirection.In, 10 * 60))
            {
                cmd.SetCDB16(8, (ushort)((initial_size - 8) / 16));
                cmd.SetCDB8(10, 3);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                len = cmd.GetBuffer32(0);
                len += 4;                       // For the length field
            }

            using (Command cmd = new Command(ScsiCommandCode.GetPerformance, 12, (ushort)len, Command.CmdDirection.In, 10 * 60))
            {
                cmd.SetCDB16(8, (ushort)((len - 8) / 16));
                cmd.SetCDB8(10, 3);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;


                list = new SpeedDescriptorList(cmd.GetBuffer(), cmd.BufferSize);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus GetPerformance(uint lba, PerformanceList.DataType rwtype, PerformanceList.ExceptType extype, out PerformanceList list)
        {
            list = null;

            if (m_logger != null)
            {
                string args = rwtype.ToString() + ", " + extype.ToString() + ", list";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.GetPerformance(" + args + ")"));
            }

            uint len = 0;
            using (Command cmd = new Command(ScsiCommandCode.GetPerformance, 12, 24, Command.CmdDirection.In, 10))
            {
                byte b = 0x10;
                if (rwtype == PerformanceList.DataType.WriteData)
                    b |= 0x04 ;

                b |= (byte)extype;

                cmd.SetCDB8(1, b);
                cmd.SetCDB16(8, 1);

                if (extype == PerformanceList.ExceptType.Entire)
                    cmd.SetCDB32(2, lba);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                len = cmd.GetBuffer32(0);
                len += 4;       // For the length field
            }

            using (Command cmd = new Command(ScsiCommandCode.GetPerformance, 12, (ushort)len, Command.CmdDirection.In, 10))
            {
                byte b = 0x10;
                if (rwtype == PerformanceList.DataType.WriteData)
                    b |= 0x04;

                b |= (byte)extype;

                cmd.SetCDB8(1, b);

                if (extype == PerformanceList.ExceptType.Entire)
                    cmd.SetCDB32(2, lba);

                cmd.SetCDB16(8, (ushort)((len - 8) / 16));

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                list = new PerformanceList(cmd.GetBuffer(), cmd.BufferSize);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// Perform a SCSI inquiry on the device to get information about the device
        /// </summary>
        /// <param name="result">The return value describing the inquiry results</param>
        /// <returns></returns>
        public CommandStatus Inquiry(out InquiryResult result)
        {
            if (m_logger != null)
            {
                string args = "out result";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.Inquiry(" + args + ")"));
            }

            result = null;

            byte len = 0;
            using (Command cmd = new Command(ScsiCommandCode.Inquiry, 6, 36, Command.CmdDirection.In, 10))
            {
                cmd.SetCDB16(3, 36);
                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                len = cmd.GetBuffer8(4);
                len += 5;

                if (len <= cmd.BufferSize)
                {
                    result = new InquiryResult(cmd.GetBuffer(), len);

                    if (m_logger != null)
                        m_logger.DumpBuffer(9, "Raw Inquiry Result", cmd.GetBuffer(), len);
                    return CommandStatus.Success;
                }

                //
                // As an oddity, the Sony DW-G120A only supports requests that are an even number
                // of bytes.
                //
                if ((len % 2) == 1)
                    len = (byte)((len / 2 * 2) + (((len % 2) == 1) ? 2 : 0));
            }

            using (Command cmd = new Command(ScsiCommandCode.Inquiry, 6, len, Command.CmdDirection.In, 100))
            {
                cmd.SetCDB8(4, len);
                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                result = new InquiryResult(cmd.GetBuffer(), cmd.BufferSize);

                if (m_logger != null)
                    m_logger.DumpBuffer(9, "Raw Inquiry Result", cmd.GetBuffer(), cmd.BufferSize);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="immd"></param>
        /// <param name="action"></param>
        /// <param name="slot"></param>
        /// <returns></returns>
        public CommandStatus LoadUnloadMedium(bool immd, LoadUnload action, byte slot)
        {
            if (m_logger != null)
            {
                string args = immd.ToString() + ", " + action.ToString() + ", " + slot.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.LoadUnloadMedium(" + args + ")"));
            }
            
            if (action == LoadUnload.Reserved)
                throw new Exception("invalid action - cannot use reserved value");

            using (Command cmd = new Command(ScsiCommandCode.LoadUnloadMedium, 12, 0, Command.CmdDirection.None, 60 * 5))
            {
                if (immd)
                    cmd.SetCDB8(1, 1);

                cmd.SetCDB8(4, (byte)action);
                cmd.SetCDB8(8, slot);
                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus MechanismStatus()
        {
            if (m_logger != null)
            {
                string args = "NOT IMPLEMENTED";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.MechanismStatus(" + args + ")"));
            }
            return CommandStatus.NotSupported;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="save"></param>
        /// <param name="table"></param>
        /// <returns></returns>
        public CommandStatus ModeSelect(bool save, ModeTable table)
        {
            if (m_logger != null)
            {
                string t = "";
                for (int i = 0; i < table.Pages.Count; i++)
                {
                    if (t.Length > 0)
                        t += ", ";
                    t = t + table.Pages[i].PageCode;
                }
                string args = save.ToString() + ", ModeTable(" + t + ")";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ModeSelect(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.ModeSelect, 10, table.Size, Command.CmdDirection.Out, 60 * 5))
            {
                byte b = 0x10;
                if (save)
                    b |= 0x01;

                cmd.SetCDB8(1, b);
                cmd.SetCDB16(7, table.Size);
                table.Format(cmd.GetBuffer());

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="pc"></param>
        /// <param name="page"></param>
        /// <param name="table"></param>
        /// <returns></returns>
        public CommandStatus ModeSense(PageControl pc, byte page, out ModeTable table)
        {
            if (m_logger != null)
            {
                string args = pc.ToString() + ", " + page.ToString() + ", out result";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ModeSense(" + args + ")"));
            }

            ushort len = 0;

            table = null;

            byte b = 0;
            b |= (byte)((byte)(pc) << 7);
            b |= (byte)(page & 0x3f);

            using (Command cmd = new Command(ScsiCommandCode.ModeSense, 10, 8, Command.CmdDirection.In, 60 * 5))
            {
                cmd.SetCDB8(1, 8);              // Disable block descriptors
                cmd.SetCDB8(2, b);
                cmd.SetCDB16(7, 8);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                len = cmd.GetBuffer16(0);
                len += 2;
            }

            using (Command cmd = new Command(ScsiCommandCode.ModeSense, 10, len, Command.CmdDirection.In, 60 * 5))
            {
                cmd.SetCDB8(1, 8);              // Disable block descriptors
                cmd.SetCDB8(2, b);
                cmd.SetCDB16(7, len);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                table = new ModeTable(cmd.GetBuffer(), cmd.BufferSize);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="action"></param>
        /// <returns></returns>
        public CommandStatus PauseResume(PauseResumeAction action)
        {
            if (m_logger != null)
            {
                string args = action.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.PauseResumeAction(" + args + ")"));
            }
            using (Command cmd = new Command(ScsiCommandCode.PauseResume, 10, 0, Command.CmdDirection.None, 60 * 5))
            {
                if (action == PauseResumeAction.Resume)
                    cmd.SetCDB8(8, 1);
                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// This command requests that the SCSI device begins an audio playback operation.
        /// </summary>
        /// <param name="lba">The starting logical block address for the playback</param>
        /// <param name="length">The length of the disk to play</param>
        /// <returns></returns>
        public CommandStatus PlayAudio(uint lba, uint length)
        {
            if (m_logger != null)
            {
                string args = lba.ToString() + ", " + length.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.PlayAudio(" + args + ")"));
            }
            if (length > 65535)
            {
                using (Command cmd = new Command(ScsiCommandCode.PlayAudio12, 12, 0, Command.CmdDirection.None, 5))
                {
                    cmd.SetCDB32(2, lba);
                    cmd.SetCDB32(6, length);
                    CommandStatus st = SendCommand(cmd);
                    if (st != CommandStatus.Success)
                        return st;
                }
            }
            else
            {
                using (Command cmd = new Command(ScsiCommandCode.PlayAudio10, 10, 0, Command.CmdDirection.None, 5))
                {
                    cmd.SetCDB32(2, lba);
                    cmd.SetCDB16(7, (ushort)length);
                    CommandStatus st = SendCommand(cmd);
                    if (st != CommandStatus.Success)
                        return st;
                }
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="start"></param>
        /// <param name="end"></param>
        /// <returns></returns>
        public CommandStatus PlayAudio(MinuteSecondFrame start, MinuteSecondFrame end)
        {
            if (m_logger != null)
            {
                string args = "NOT IMPLEMENTED";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.PlayAudio(" + args + ")"));
            }
            return CommandStatus.NotSupported;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="action"></param>
        /// <returns></returns>
        public CommandStatus PreventAllowMediumRemoval(PreventAllow action)
        {
            if (m_logger != null)
            {
                string args = action.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.PreventAllowMediumRemoval(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.PreventAllowMediumRemoval, 6, 0, Command.CmdDirection.None, 60 * 5))
            {
                cmd.SetCDB8(4, (byte)action);
                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// This method requests that the SCSI device transfer data to the given buffer
        /// </summary>
        /// <param name="force">If true, the data is forced from the media and cannot be read from the cache</param>
        /// <param name="streaming">If true, this is a streaming read</param>
        /// <param name="lba">The starting logical address for the data</param>
        /// <param name="length">The length of the data to read</param>
        /// <param name="data">The data buffer to received the data</param>
        /// <returns></returns>
        public CommandStatus Read(bool force, bool streaming, uint lba, uint length, ref byte [] data)
        {
            if (m_logger != null)
            {
                string args = force.ToString() + ", " + streaming.ToString() + ", " + lba.ToString() + ", " + length.ToString() + ", buffer";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.Read(" + args + ")"));
            }

            if (streaming || length > 65535)
            {
                using (Command cmd = new Command(ScsiCommandCode.Read12, 12, (ushort)data.GetLength(0), Command.CmdDirection.In, 5 * 60))
                {
                    if (force)
                        cmd.SetCDB8(1, 4);              // Set the FUA bit

                    cmd.SetCDB32(2, lba);
                    cmd.SetCDB32(6, length);

                    if (streaming)
                        cmd.SetCDB8(10, 0x80);          // Set the streaming bit

                    CommandStatus st = SendCommand(cmd);
                    if (st != CommandStatus.Success)
                        return st;

                    Marshal.Copy(cmd.GetBuffer(), data, 0, data.GetLength(0));
                }
            }
            else
            {
                using (Command cmd = new Command(ScsiCommandCode.Read, 10, (ushort)data.GetLength(0), Command.CmdDirection.In, 5 * 60))
                {
                    if (force)
                        cmd.SetCDB8(1, 4);              // Set the FUA bit

                    cmd.SetCDB32(2, lba);
                    cmd.SetCDB16(7, (ushort)length);

                    CommandStatus st = SendCommand(cmd);
                    if (st != CommandStatus.Success)
                        return st;

                    Marshal.Copy(cmd.GetBuffer(), data, 0, data.GetLength(0));
                }
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// Read data from the device into a memory buffer
        /// </summary>
        /// <param name="force">If true, the data is forced from the media and cannot be read from the cache</param>
        /// <param name="streaming">If true, this is a streaming read</param>
        /// <param name="lba">The starting logical address for the data</param>
        /// <param name="length">The length of the data to read</param>
        /// <param name="data">The buffer to receive the data</param>
        /// <param name="size">The size of the buffer given by the data parameter</param>
        /// <returns></returns>
        public CommandStatus Read(bool force, bool streaming, uint lba, uint length, IntPtr data, int size)
        {
            if (m_logger != null)
            {
                string args = force.ToString() + ", " + streaming.ToString() + ", " + lba.ToString() + ", " + length.ToString() + ", IntPtr, " + size.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.Read(" + args + ")"));
            }

            if (streaming || length > 65535)
            {
                using (Command cmd = new Command(ScsiCommandCode.Read12, 12, data, size, Command.CmdDirection.In, 5 * 60))
                {
                    if (force)
                        cmd.SetCDB8(1, 4);

                    cmd.SetCDB32(2, lba);
                    cmd.SetCDB32(6, length);

                    if (streaming)
                        cmd.SetCDB8(10, 0x80);

                    CommandStatus st = SendCommand(cmd);
                    if (st != CommandStatus.Success)
                        return st;
                }
            }
            else
            {
                using (Command cmd = new Command(ScsiCommandCode.Read, 10, data, size, Command.CmdDirection.In, 5 * 60))
                {
                    if (force)
                        cmd.SetCDB8(1, 4);              // Set the FUA bit

                    cmd.SetCDB32(2, lba);
                    cmd.SetCDB16(7, (ushort)length);

                    CommandStatus st = SendCommand(cmd);
                    if (st != CommandStatus.Success)
                        return st;
                }
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus ReadBuffer()
        {
            if (m_logger != null)
            {
                string args = "NOT IMPLEMENTED";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadBuffer(" + args + ")"));
            }
            return CommandStatus.NotSupported;
        }

        /// <summary>
        /// This method returns the buffer capacity for the pass through SCSI device. 
        /// </summary>
        /// <param name="blocks">if true, the length and available is given in blocks, otherwise bytes</param>
        /// <param name="length">return vaue, the length of the buffer</param>
        /// <param name="avail">return value, the available space in the buffer</param>
        /// <returns>status of the command</returns>
        public CommandStatus ReadBufferCapacity(bool blocks, out int length, out int avail)
        {
            if (m_logger != null)
            {
                string args = blocks.ToString() + ", out length, out available";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 9, "Bwg.Scsi.Device.ReadBufferCapacity(" + args + ")"));
            }
            length = 0;
            avail = 0;

            using (Command cmd = new Command(ScsiCommandCode.ReadBufferCapacity, 10, 12, Command.CmdDirection.In, 60 * 5))
            {
                cmd.SetCDB16(7, 12);
                if (blocks)
                    cmd.SetCDB8(1, 1);
                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                length = (int)cmd.GetBuffer32(4);
                avail = (int)cmd.GetBuffer32(8);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="lba"></param>
        /// <param name="blocklen"></param>
        /// <returns></returns>
        public CommandStatus ReadCapacity(out uint lba, out uint blocklen)
        {
            if (m_logger != null)
            {
                string args = "out lba, out blocklen";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadCapacity(" + args + ")"));
            }
            blocklen = 0;
            lba = 0;

            using (Command cmd = new Command(ScsiCommandCode.ReadCapacity, 10, 12, Command.CmdDirection.In, 60 * 5))
            {
                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                lba = cmd.GetBuffer32(0);
                blocklen = cmd.GetBuffer32(4);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="exp"></param>
        /// <param name="dap"></param>
        /// <param name="start"></param>
        /// <param name="length"></param>
        /// <param name="data">the memory area </param>
        /// <param name="size">the size of the memory area given by the data parameter</param>
        /// <returns></returns>
        public CommandStatus ReadCD(byte exp, bool dap, uint start, uint length, IntPtr data, int size)
        {
            if (m_logger != null)
            {
                string args = exp.ToString() + ", " + dap.ToString() + ", " + start.ToString() + ", " + length.ToString() + ", data, " + size.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadCD(" + args + ")"));
            }

            if (exp != 1 && exp != 2 && exp != 3 && exp != 4 && exp != 5)
                return CommandStatus.NotSupported;

            using (Command cmd = new Command(ScsiCommandCode.ReadCd, 12, data, size, Command.CmdDirection.In, 5 * 60))
            {
                byte b = (byte)((exp & 0x07) << 2);
                if (dap)
                    b |= 0x02;
                cmd.SetCDB8(1, b);
                cmd.SetCDB32(2, start);
                cmd.SetCDB24(6, length);
                cmd.SetCDB8(9, 0x10);           // User data only

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

		/// <summary>
		/// 
		/// </summary>
		public enum SubChannelMode
		{
			/// <summary></summary>
			None,
			/// <summary>+ 16 bytes</summary>
			QOnly,
			/// <summary>+ 96 bytes</summary>
			RWMode 
		};

		/// <summary>
		/// 
		/// </summary>
		public enum C2ErrorMode
		{
			/// <summary></summary>
			None,
			/// <summary> +294 bytes</summary>
			Mode294,
			/// <summary> +296 bytes</summary>
			Mode296, 
		};

		/// <summary>
		/// 
		/// </summary>
		public enum MainChannelSelection
		{
			/// <summary>
			/// 
			/// </summary>
			UserData,
			/// <summary>
			/// 
			/// </summary>
			F8h
		};

		/// <summary>
		/// 
		/// </summary>
		/// <param name="mainmode">main channel mode</param>
		/// <param name="submode">subchannel mode</param>
		/// <param name="c2mode">C2 errors report mode</param>
		/// <param name="exp">expected sector type</param>
		/// <param name="dap"></param>
		/// <param name="start"></param>
		/// <param name="length"></param>
		/// <param name="data">the memory area </param>
		/// <param name="timeout">timeout (in seconds)</param>
		/// <returns></returns>
		public CommandStatus ReadCDAndSubChannel(MainChannelSelection mainmode, SubChannelMode submode, C2ErrorMode c2mode, byte exp, bool dap, uint start, uint length, IntPtr data, int timeout)
		{
			if (m_logger != null)
			{
				string args = exp.ToString() + ", " + dap.ToString() + ", " + start.ToString() + ", " + length.ToString() + ", data";
				m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadCD(" + args + ")"));
			}

			int size = (4 * 588 +
				(submode == SubChannelMode.QOnly ? 16 : submode == SubChannelMode.RWMode ? 96 : 0) +
				(c2mode == C2ErrorMode.Mode294 ? 294 : c2mode == C2ErrorMode.Mode296 ? 296 : 0)) * (int) length;

			byte mode = (byte) (submode == SubChannelMode.QOnly ? 2 : submode == SubChannelMode.RWMode ? 4 : 0);

			if (exp != 1 && exp != 2 && exp != 3 && exp != 4 && exp != 5)
				return CommandStatus.NotSupported;

			using (Command cmd = new Command(ScsiCommandCode.ReadCd, 12, data, size, Command.CmdDirection.In, timeout))
			{
				byte b = (byte)((exp & 0x07) << 2);
				if (dap)
					b |= 0x02;
				byte byte9 = (byte) (mainmode == MainChannelSelection.UserData ? 0x10 : 0xF8);
				if (c2mode == C2ErrorMode.Mode294)
					byte9 |= 0x02;
				else if (c2mode == C2ErrorMode.Mode296)
					byte9 |= 0x04;
				cmd.SetCDB8(1, b);
				cmd.SetCDB32(2, start);
				cmd.SetCDB24(6, length);
				cmd.SetCDB8(9, byte9); // User data + possibly c2 errors
				cmd.SetCDB8(10, mode);          // Subchannel

				CommandStatus st = SendCommand(cmd);
				if (st != CommandStatus.Success)
					return st;
			}

			return CommandStatus.Success;
		}

		/// <summary>
		/// 
		/// </summary>
		/// <param name="submode">subchannel mode</param>
		/// <param name="start"></param>
		/// <param name="length"></param>
		/// <param name="data">the memory area </param>
		/// <param name="timeout">timeout (in seconds)</param>
		/// <returns></returns>
		public CommandStatus ReadCDDA(SubChannelMode submode, uint start, uint length, IntPtr data, int timeout)
		{
			if (m_logger != null)
			{
				string args = start.ToString() + ", " + length.ToString() + ", data";
				m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadCDDA(" + args + ")"));
			}

			byte mode = (byte)(submode == SubChannelMode.QOnly ? 1 : submode == SubChannelMode.RWMode ? 2 : 0);
			int size = 4 * 588 + (submode == SubChannelMode.QOnly ? 16 : submode == SubChannelMode.RWMode ? 96 : 0);

			using (Command cmd = new Command(ScsiCommandCode.ReadCDDA, 12, data, size, Command.CmdDirection.In, timeout))
			{
				cmd.SetCDB8(1, 0 << 5); // lun
				cmd.SetCDB32(2, start);
				cmd.SetCDB24(7, length);
				cmd.SetCDB8(10, mode); // Subchannel

				CommandStatus st = SendCommand(cmd);
				if (st != CommandStatus.Success)
					return st;
			}

			return CommandStatus.Success;
		}

        /// <summary>
        /// Read CD header data from the disk
        /// </summary>
        /// <param name="sector">the sector number containing the header data to read</param>
        /// <param name="hdr">the return data read</param>
        /// <returns></returns>
        public CommandStatus ReadCD(uint sector, out HeaderData hdr)
        {
            hdr = null;

            if (m_logger != null)
            {
                string args = sector.ToString() + ", out HeaderData hdr" ;
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadCD(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.ReadCd, 12, 4, Command.CmdDirection.In, 10))
            {
                cmd.SetCDB32(2, sector);
                cmd.SetCDB24(6, 1);
                cmd.SetCDB8(9, 0x20);          // Header data only

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                hdr = new HeaderData(cmd.GetBuffer(), cmd.BufferSize);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// This method reads subheader CD data from a data mode 2 track
        /// </summary>
        /// <param name="sector">the sector # of the data to read</param>
        /// <param name="hdr">return subheader data</param>
        /// <returns></returns>
        public CommandStatus ReadCD(uint sector, out SubheaderData hdr)
        {
            hdr = null ;
            if (m_logger != null)
            {
                string args = sector.ToString() + ", out SubheaderData hdr";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadCD(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.ReadCd, 12, 4, Command.CmdDirection.In, 10))
            {
                cmd.SetCDB32(2, sector);
                cmd.SetCDB24(6, 1);
                cmd.SetCDB8(9, 0x40);          // Header data only

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                hdr = new SubheaderData(cmd.GetBuffer(), cmd.BufferSize);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// Read the subchannel data from a series of sectors
        /// </summary>
        /// <param name="sector"></param>
        /// <param name="length"></param>
        /// <param name="data"></param>
        /// <param name="mode">the subchannel mode</param>
		/// <param name="timeout">timeout (in seconds)</param>
        /// <returns></returns>
        public CommandStatus ReadSubChannel(byte mode, uint sector, uint length, ref byte[] data, int timeout)
        {
            byte bytes_per_sector;

            if (m_logger != null)
            {
                string args = sector.ToString() + ", " + length.ToString() + ", data";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadSubChannel(" + args + ")"));
            }

            if (mode != 1 && mode != 2 && mode != 4)
                throw new Exception("invalid read mode for ReadSubchannel() call");

            bytes_per_sector = 96;
            if (mode == 2)
                bytes_per_sector = 16;

            if (data.GetLength(0) < length * bytes_per_sector)
                throw new Exception("data buffer is not large enough to hold the data requested");


            using (Command cmd = new Command(ScsiCommandCode.ReadCd, 12, (int)(length * bytes_per_sector), Command.CmdDirection.In, timeout))
            {
                cmd.SetCDB32(2, sector);            // The sector number to start with
                cmd.SetCDB24(6, length);            // The length in sectors
                cmd.SetCDB8(10, mode);                 // Corrected, de-interleaved P - W data

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                Marshal.Copy(cmd.GetBuffer(), data, 0, (int)(length * bytes_per_sector));
            }
            return CommandStatus.Success ;
        }

		/// <summary>
		/// Read the subchannel data from a series of sectors
		/// </summary>
		/// <param name="mode">subchannel mode</param>
		/// <param name="track">track number</param>
		/// <param name="data">output buffer</param>
		/// <param name="offs">output buffer offset</param>
		/// <param name="timeout">timeout (in seconds)</param>
		/// <returns></returns>
		public CommandStatus ReadSubChannel42(byte mode, int track, ref byte[] data, int offs, int timeout)
		{
			if (m_logger != null)
			{
				m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadSubChannel42()"));
			}

			if (mode != 1 && mode != 2 && mode != 3)
				throw new Exception("invalid read mode for ReadSubchannel42() call");

			int size = mode == 1 ? 16 : 24;
			if (offs + data.GetLength(0) < size)
				throw new Exception("data buffer is not large enough to hold the data requested");

			using (Command cmd = new Command(ScsiCommandCode.ReadSubChannel, 10, size, Command.CmdDirection.In, timeout))
			{
				// 42 00 40 01  00 00 00 00 10 00
				//cmd.SetCDB8(1, 2); // MSF
				cmd.SetCDB8(2, 64); // SUBQ
				cmd.SetCDB8(3, mode);
				cmd.SetCDB8(6, (byte)track);
				cmd.SetCDB16(7, (ushort)size);

				CommandStatus st = SendCommand(cmd);
				if (st != CommandStatus.Success)
					return st;

				Marshal.Copy(cmd.GetBuffer(), data, offs, size);
			}
			return CommandStatus.Success;
		}

        /// <summary>
        /// Read the CD text information from the leadin using the ReadTocPmaAtip command form
        /// </summary>
        /// <param name="data"></param>
		/// <param name="_timeout"></param>
        /// <returns></returns>
        public CommandStatus ReadCDText(out byte [] data, int _timeout)
        {
            ushort len;

            if (m_logger != null)
            {
                string args = "info";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadCDText(" + args + ")"));
            }

            data = null;
			using (Command cmd = new Command(ScsiCommandCode.ReadTocPmaAtip, 10, 4, Command.CmdDirection.In, _timeout))
            {
                cmd.SetCDB8(2, 5);                  // CDText info in leadin
                cmd.SetCDB16(7, 4);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                len = cmd.GetBuffer16(0);
                len += 2;

                if (len <= 4)
                {
                    data = new byte[len];
                    Marshal.Copy(cmd.GetBuffer(), data, 0, len);
                    return CommandStatus.Success;
                }
            }

			using (Command cmd = new Command(ScsiCommandCode.ReadTocPmaAtip, 10, len, Command.CmdDirection.In, _timeout))
            {
                cmd.SetCDB8(2, 5);                 // CDText info in leadin
                cmd.SetCDB16(7, len);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                data = new byte[len];
                Marshal.Copy(cmd.GetBuffer(), data, 0, len);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus ReadDiskInformation(out DiskInformation result)
        {
            if (m_logger != null)
            {
                string args = "out result";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadDiskInformation(" + args + ")"));
            }

            ushort len = 0;
            result = null;

            using (Command cmd = new Command(ScsiCommandCode.ReadDiskInformation, 10, 34, Command.CmdDirection.In, 60))
            {
                cmd.SetCDB16(7, 34);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;


                len = cmd.GetBuffer16(0);
                if (len <= 34)
                {
                    result = new DiskInformation(cmd.GetBuffer(), cmd.BufferSize);
                    return CommandStatus.Success;
                }
                len += 2;
            }

            using (Command cmd = new Command(ScsiCommandCode.ReadDiskInformation, 10, len, Command.CmdDirection.In, 60))
            {
                cmd.SetCDB16(7, len);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                result = new DiskInformation(cmd.GetBuffer(), cmd.BufferSize);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// Return the AACS Volume Identifier as a series of bytes
        /// </summary>
        /// <param name="data">the AACS volume identifier</param>
        /// <returns>status of command execution</returns>
        public CommandStatus ReadDiskStructureAACSVolumeIdentifier(out byte [] data)
        {
            return ReadDiskStructureReturnBytes(0x80, out data);
        }

        /// <summary>
        /// Return the AACS media serial number
        /// </summary>
        /// <param name="data">the media serial number</param>
        /// <returns>status of command execution</returns>
        public CommandStatus ReadDiskStructureAACSMediaSerialNumber(out byte[] data)
        {
            return ReadDiskStructureReturnBytes(0x81, out data);
        }

        /// <summary>
        /// This method reads the PAC data for a blu-ray disk
        /// </summary>
        /// <param name="pacid">the pac id of the pac to read</param>
        /// <param name="pacfmt">the format of the pac to read</param>
        /// <param name="data">return storage for the pac data</param>
        /// <returns></returns>
        public CommandStatus ReadDiskStructurePac(uint pacid, byte pacfmt, out byte [] data)
        {
            data = null;

            if (m_logger != null)
            {
                string args = "out byte[] data";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadDvdStructurePac(" + args + ")"));
            }

            ushort len = 4;
            using (Command cmd = new Command(ScsiCommandCode.ReadDvdStructure, 12, len, Command.CmdDirection.In, 60))
            {
                cmd.SetCDB24(2, pacid);
                cmd.SetCDB8(6, pacfmt);
                cmd.SetCDB8(7, 0x30);            // Read PAC data
                cmd.SetCDB16(8, len);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                len = (ushort)(cmd.GetBuffer16(0) + 2);
                if (len == 2)
                {
                    data = new byte[0];
                    return CommandStatus.Success;
                }
                Debug.Assert(len < 8192);
            }

            using (Command cmd = new Command(ScsiCommandCode.ReadDvdStructure, 12, len, Command.CmdDirection.In, 60))
            {
                cmd.SetCDB24(2, pacid);
                cmd.SetCDB8(6, pacfmt);
                cmd.SetCDB8(7, 0x30);         // Read manufacturing information
                cmd.SetCDB16(8, len);          // Up to 2k of data

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                len = (ushort)(cmd.GetBuffer16(0) + 2);
                data = new byte[len];
                Marshal.Copy(cmd.GetBuffer(), data, 0, len);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// Return the data from a ReadDvdStructure request as a series of bytes.
        /// </summary>
        /// <param name="format">the format of the read dvd structure request</param>
        /// <param name="data">the AACS volume identifier</param>
        /// <returns>status of command execution</returns>
        private CommandStatus ReadDiskStructureReturnBytes(byte format, out byte[] data)
        {
            data = null;

            if (m_logger != null)
            {
                string args = "out byte[] data";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadDvdStructureAACSVolumeIdentifier(" + args + ")"));
            }

            ushort len = 4;
            using (Command cmd = new Command(ScsiCommandCode.ReadDvdStructure, 12, len, Command.CmdDirection.In, 60))
            {
                cmd.SetCDB8(7, format);         // Read manufacturing information
                cmd.SetCDB16(8, len);          // Up to 2k of data

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                len = cmd.GetBuffer16(0);
                if (len == 0)
                {
                    data = new byte[0];
                    return CommandStatus.Success;
                }
                Debug.Assert(len < 8192);
            }

            using (Command cmd = new Command(ScsiCommandCode.ReadDvdStructure, 12, len, Command.CmdDirection.In, 60))
            {
                cmd.SetCDB8(7, format);         // Read manufacturing information
                cmd.SetCDB16(8, len);          // Up to 2k of data

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                len = cmd.GetBuffer16(0);
                data = new byte[len];
                Marshal.Copy(cmd.GetBuffer(), data, 0, len);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus ReadDiskStructure(uint addr, byte layer, byte format, ref byte [] data)
        {
            if (m_logger != null)
            {
                string args = string.Empty;
                args += "addr=" + addr.ToString();
                args += ", layer=" + layer.ToString();
                args += ", format=" + format.ToString();
                args += "ref byte[] data";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadDvdStructure(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.ReadDvdStructure, 12, 2048, Command.CmdDirection.In, 60))
            {
                cmd.SetCDB32(2, addr) ;         // Address = 0 
                cmd.SetCDB8(6, layer);          // Layer number = 0
                cmd.SetCDB8(7, format);         // Read manufacturing information
                cmd.SetCDB16(8, 2048);          // Up to 2k of data

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
                
                int len = data.GetLength(0);
                if (len > 2048)
                    len = 2048;
                Marshal.Copy(cmd.GetBuffer(), data, 0, len);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// Query the media for the spare sectors available
        /// </summary>
        /// <param name="primary">the number of primary spare sectors available</param>
        /// <param name="sec_avail">the number of secondary spare sectors available</param>
        /// <param name="sec_total">the number of secondary spare sectors total</param>
        /// <returns></returns>
        public CommandStatus ReadDVDRamSpareAreaInfo(out uint primary, out uint sec_avail, out uint sec_total)
        {
            primary = 0;
            sec_avail = 0;
            sec_total = 0;

            if (m_logger != null)
            {
                string args = "out primary, out sec_avail, out sec_total";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadDVDRamSpareAreaInfo(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.ReadDvdStructure, 12, 2048, Command.CmdDirection.In, 60))
            {
                cmd.SetCDB8(7, 0x0a);         // Read manufacturing information
                cmd.SetCDB16(8, 16);          // Up to 2k of data

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                primary = cmd.GetBuffer32(4);
                sec_avail = cmd.GetBuffer32(8);
                sec_total = cmd.GetBuffer32(12);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// Returns the size of layer 0 for DL media (DVD+R DL and DVD-R DL)
        /// </summary>
        /// <param name="isfixed">return value, if true size of L0 is changable</param>
        /// <param name="size">return value, the current size of L0</param>
        /// <returns>status of the command</returns>
        public CommandStatus ReadDvdLayer0Size(out bool isfixed, out uint size)
        {
            if (m_logger != null)
            {
                string args = "ReadDvdLayer0Size, out bool isfixed, out uint size";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadDvdStructure(" + args + ")"));
            }

            size = 0;
            isfixed = false;

            using (Command cmd = new Command(ScsiCommandCode.ReadDvdStructure, 12, 12, Command.CmdDirection.In, 5 * 60))
            {
                cmd.SetCDB8(7, 0x20);         // Read manufacturing information
                cmd.SetCDB8(8, 12);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                if ((cmd.GetBuffer8(4) & 0x80) != 0)
                    isfixed = true;

                size = cmd.GetBuffer32(8);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// Returns the start address of the Layer 0 middle zone (DVD+R DL, DVD-R DL)
        /// </summary>
        /// <param name="isfixed">return value, if true, the middle zone start is changeable</param>
        /// <param name="location">return value, the location of the L0 middle zone</param>
        /// <returns>status of the command</returns>
        public CommandStatus ReadDvdMiddleZoneStartAddr(out bool isfixed, out uint location)
        {
            if (m_logger != null)
            {
                string args = "ReadDvdMiddleZoneStartAddr, out bool isfixed, out uint size";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadDvdStructure(" + args + ")"));
            }

            location = 0;
            isfixed = false;

            using (Command cmd = new Command(ScsiCommandCode.ReadDvdStructure, 12, 12, Command.CmdDirection.In, 5 * 60))
            {
                cmd.SetCDB8(7, 0x21);
                cmd.SetCDB8(8, 12);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                if ((cmd.GetBuffer8(4) & 0x80) != 0)
                    isfixed = true;

                location = cmd.GetBuffer32(8);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// Reads the jump interval size (DVD-R DL)
        /// </summary>
        /// <param name="size">return value, the size of the jump interval</param>
        /// <returns>status of the command</returns>
        public CommandStatus ReadDvdJumpIntervalSize(out uint size)
        {
            if (m_logger != null)
            {
                string args = "ReadDvdJumpIntervalSize, out uint size";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadDvdStructure(" + args + ")"));
            }
            size = 0;

            using (Command cmd = new Command(ScsiCommandCode.ReadDvdStructure, 12, 12, Command.CmdDirection.In, 5 * 60))
            {
                cmd.SetCDB8(7, 0x22);
                cmd.SetCDB8(8, 12);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                size = cmd.GetBuffer32(8);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// Returs the manual jump address for the media (DVD-R DL)
        /// </summary>
        /// <param name="addr">return value, the jump address</param>
        /// <returns>the command status</returns>
        public CommandStatus ReadDvdManualLayerJumpAddress(out uint addr)
        {
            if (m_logger != null)
            {
                string args = "ReadDvdManualLayerJumpAddress, out uint size";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadDvdStructure(" + args + ")"));
            }
            addr = 0;

            using (Command cmd = new Command(ScsiCommandCode.ReadDvdStructure, 12, 12, Command.CmdDirection.In, 5 * 60))
            {
                cmd.SetCDB8(7, 0x23);
                cmd.SetCDB8(8, 12);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                addr = cmd.GetBuffer32(8);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// This method returns the remapping address (DVD-R DL(
        /// </summary>
        /// <param name="location">the remapping address</param>
        /// <returns>the command status</returns>
        public CommandStatus ReadDvdRemappingAddress(out uint location)
        {
            if (m_logger != null)
            {
                string args = "ReadDvdRemappingAddress, out uint size";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadDvdStructure(" + args + ")"));
            }

            location = 0;

            using (Command cmd = new Command(ScsiCommandCode.ReadDvdStructure, 12, 12, Command.CmdDirection.In, 5 * 60))
            {
                cmd.SetCDB8(7, 0x24);
                cmd.SetCDB8(8, 12);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                location = cmd.GetBuffer32(8);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// Send the layer boundary information to the drive.
        /// </summary>
        /// <param name="boundary">the location of the boundary between layers in blocks</param>
        /// <returns>the status of the command</returns>
        public CommandStatus SendDvdLayerBoundaryInformation(uint boundary)
        {
            if (m_logger != null)
            {
                string args = "SendDvdLayerBoundaryInformation, boundary=" + boundary.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.SendDvdStructure(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.SendDvdStructure, 12, 12, Command.CmdDirection.Out, 5 * 60))
            {
                cmd.SetCDB8(7, 0x20);
                cmd.SetCDB16(8, 12);

                cmd.SetBuffer16(0, 10);
                cmd.SetBuffer32(8, boundary);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus ReadFormatCapacities(bool all, out IList<CapacityDescriptor> caplist)
        {
            caplist = new List<CapacityDescriptor>();

            if (m_logger != null)
            {
                string args = all.ToString() + ", out caplist";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadFormatCapacities(" + args + ")"));
            }
            int len;

            using (Command cmd = new Command(ScsiCommandCode.ReadFormatCapacities, 10, 8, Command.CmdDirection.In, 60))
            {
                cmd.SetCDB16(7, 8);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                len = cmd.GetBuffer8(3) + 4;
            }
            using (Command cmd = new Command(ScsiCommandCode.ReadFormatCapacities, 10, len, Command.CmdDirection.In, 60))
            {
                cmd.SetCDB16(7, (ushort)len);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                int offset = 4;
                len = cmd.GetBuffer8(3) + 4;
                while (offset < len)
                {
                    caplist.Add(new CapacityDescriptor(cmd.GetBuffer(), offset, cmd.BufferSize));
                    offset += 8;
                }
            }
            return CommandStatus.Success;
        }

        /// <summary>
        /// Read the table of contents from the disk
        /// </summary>
        /// <param name="track">the track or session to find the TOC for</param>
        /// <param name="toc">a list return value containins a list of table of content entryies</param>
        /// <param name="mode">time versus lba mode</param>
        /// <returns></returns>
        public CommandStatus ReadToc(byte track, bool mode, out IList<TocEntry> toc)
        {
            ushort len;
            toc = null;

            if (m_logger != null)
            {
                string args = "info";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadToc(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.ReadTocPmaAtip, 10, 16, Command.CmdDirection.In, 5 * 60))
            {
                if (mode)
                    cmd.SetCDB8(1, 2);
                cmd.SetCDB8(6, track);
                cmd.SetCDB16(7, 16);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                len = cmd.GetBuffer16(0);
                len += 2;
            }

            using (Command cmd = new Command(ScsiCommandCode.ReadTocPmaAtip, 10, len, Command.CmdDirection.In, 5 * 60))
            {
                if (mode)
                    cmd.SetCDB8(1, 2);
                cmd.SetCDB8(6, track);
                cmd.SetCDB16(7, len);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                int offset = 4;
                toc = new List<TocEntry>();

                while (offset + 8 <= len)
                {
                    TocEntry entry = new TocEntry(cmd.GetBuffer(), offset, cmd.BufferSize, mode);
                    toc.Add(entry);

                    offset += 8;
                }
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// This method reads the ATIP information from a CD media
        /// </summary>
        /// <returns></returns>
        public CommandStatus ReadAtip(out AtipInfo info)
        {
            ushort len;
            info = null;

            if (m_logger != null)
            {
                string args = "info";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadAtip(" + args + ")"));
            }
            using (Command cmd = new Command(ScsiCommandCode.ReadTocPmaAtip, 10, 32, Command.CmdDirection.In, 5 * 60))
            {
                cmd.SetCDB8(2, 4);
                cmd.SetCDB16(7, 32);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                len = cmd.GetBuffer16(0);
                len += 2;
            }

            if (len <= 4)
                return CommandStatus.Success;

            using (Command cmd = new Command(ScsiCommandCode.ReadTocPmaAtip, 10, len, Command.CmdDirection.In, 5 * 60))
            {
                cmd.SetCDB8(2, 4);
                cmd.SetCDB16(7, len);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                info = new AtipInfo(cmd.GetBuffer(), len);
            }

            return CommandStatus.Success;
        }

		/// <summary>
		/// This method reads the ATIP information from a CD media
		/// </summary>
		/// <returns></returns>
		public CommandStatus ReadFullToc(out byte[] data)
		{
			ushort len;
			data = null;

			if (m_logger != null)
			{
				string args = "info";
				m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadFullToc(" + args + ")"));
			}
			using (Command cmd = new Command(ScsiCommandCode.ReadTocPmaAtip, 10, 32, Command.CmdDirection.In, 5 * 60))
			{
				cmd.SetCDB8(2, 2);
				cmd.SetCDB16(7, 32);

				CommandStatus st = SendCommand(cmd);
				if (st != CommandStatus.Success)
					return st;

				len = cmd.GetBuffer16(0);
				len += 2;

				if (len <= 32)
				{
					data = new byte[len];
					Marshal.Copy(cmd.GetBuffer(), data, 0, len);
					return CommandStatus.Success;
				}
			}

			using (Command cmd = new Command(ScsiCommandCode.ReadTocPmaAtip, 10, len, Command.CmdDirection.In, 5 * 60))
			{
				cmd.SetCDB8(2, 2);
				cmd.SetCDB16(7, len);

				CommandStatus st = SendCommand(cmd);
				if (st != CommandStatus.Success)
					return st;

				//info = new AtipInfo(cmd.GetBuffer(), len);
				data = new byte[len];
				Marshal.Copy(cmd.GetBuffer(), data, 0, len);
			}

			return CommandStatus.Success;
		}

		/// <summary>
		/// This method reads the ATIP information from a CD media
		/// </summary>
		/// <returns></returns>
		public CommandStatus ReadPMA(out byte[] data)
		{
			ushort len;
			data = null;

			if (m_logger != null)
			{
				string args = "info";
				m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadFullToc(" + args + ")"));
			}
			using (Command cmd = new Command(ScsiCommandCode.ReadTocPmaAtip, 10, 320, Command.CmdDirection.In, 5 * 60))
			{
				cmd.SetCDB8(1, 2);
				cmd.SetCDB8(2, 3);
				cmd.SetCDB16(7, 320);

				CommandStatus st = SendCommand(cmd);
				if (st != CommandStatus.Success)
					return st;

				len = cmd.GetBuffer16(0);
				len += 2;

				if (len <= 320)
				{
					data = new byte[len];
					Marshal.Copy(cmd.GetBuffer(), data, 0, len);
					return CommandStatus.Success;
				}
			}

			using (Command cmd = new Command(ScsiCommandCode.ReadTocPmaAtip, 10, len, Command.CmdDirection.In, 5 * 60))
			{
				cmd.SetCDB8(2, 3);
				cmd.SetCDB16(7, len);

				CommandStatus st = SendCommand(cmd);
				if (st != CommandStatus.Success)
					return st;

				//info = new AtipInfo(cmd.GetBuffer(), len);
				data = new byte[len];
				Marshal.Copy(cmd.GetBuffer(), data, 0, len);
			}

			return CommandStatus.Success;
		}

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus ReadTrackInformation(ReadTrackType type, uint addr, out TrackInformation info)
        {
            info = null;

            if (m_logger != null)
            {
                string args = "info";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReadTrackInformation(" + args + ")"));
            }
            using (Command cmd = new Command(ScsiCommandCode.ReadTrackInformation, 10, 48, Command.CmdDirection.In, 5 * 60))
            {
                cmd.SetCDB8(1, (byte)type);
                cmd.SetCDB32(2, addr);
                cmd.SetCDB16(7, 40);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                info = new TrackInformation(cmd.GetBuffer(), cmd.BufferSize);
            }
            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus RepairTrack(ushort track)
        {
            if (m_logger != null)
            {
                string args = track.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.RepairTrack(" + args + ")"));
            }
            using (Command cmd = new Command(ScsiCommandCode.RepairTrack, 10, 0, Command.CmdDirection.None, 5 * 60))
            {
                cmd.SetCDB16(4, track);
                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus ReportKey()
        {
            if (m_logger != null)
            {
                string args = "NOT IMPLEMENTED";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReportKey(" + args + ")"));
            }
            return CommandStatus.NotSupported;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus RequestSense()
        {
            if (m_logger != null)
            {
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.RequestSense()"));
            }

            using (Command cmd = new Command(ScsiCommandCode.RequestSense, 6, 18, Command.CmdDirection.None, 5*60))
            {
                cmd.SetCDB8(4, 18);
                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;

                m_sense_info = new byte[18];
                Marshal.Copy(cmd.GetBuffer(), m_sense_info, 0, 18);
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus ReserveTrack(uint size)
        {
            if (m_logger != null)
            {
                string args = size.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.ReserveTrack(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.ReserveTrack, 10, 0, Command.CmdDirection.None, 5 * 60))
            {
                cmd.SetCDB32(5, size);
                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus Scan(ScanDirection dir, uint lba, ScanType type)
        {
            if (m_logger != null)
            {
                string args = dir.ToString() + ", " + lba.ToString() + ", " + type.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.Scan(" + args + ")"));
            }

            if (type == ScanType.Reserved)
                throw new Exception("type parameter is of type Reserved");

            using (Command cmd = new Command(ScsiCommandCode.Scan, 12, 0, Command.CmdDirection.None, 10))
            {
                if (dir == ScanDirection.Reverse)
                    cmd.SetCDB8(1, 0x10);
                cmd.SetCDB32(2, lba);
                cmd.SetCDB8(9, (byte)((byte)type << 6));

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus Seek(uint lba)
        {
            if (m_logger != null)
            {
                string args = lba.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.Seek(" + args + ")"));
            }
            using (Command cmd = new Command(ScsiCommandCode.Seek, 10, 0, Command.CmdDirection.None, 10))
            {
                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus SendCueSheet(byte [] sheet)
        {
            if (m_logger != null)
            {
                string args = sheet.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.SendCueSheet(" + args + ")"));

                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 1, "Cue Sheet"));
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 1, "----------------------------------------------"));
                for (int i = 0; i < sheet.GetLength(0) / 8; i++)
                {
                    string s = "" ;

                    for (int j = 0; j < 8; j++)
                    {
                        if (j != 0)
                            s += " ";
                        s += sheet[i * 8 + j].ToString("x2");
                    }
                    m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 1, s));
                }
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 1, "----------------------------------------------"));
            }

            ushort len = (ushort)sheet.GetLength(0);
            using (Command cmd = new Command(ScsiCommandCode.SendCueSheet, 10, len, Command.CmdDirection.Out, 10))
            {
                cmd.SetCDB32(5, len);
                cmd.SetCDB8(5, 0);
                Marshal.Copy(sheet, 0, cmd.GetBuffer(), len);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus SendDVDStructure()
        {
            if (m_logger != null)
            {
                string args = "NOT IMPLEMENTED";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.SendDVDStructure(" + args + ")"));
            }
            return CommandStatus.NotSupported;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus SendKey()
        {
            if (m_logger != null)
            {
                string args = "NOT IMPLEMENTED";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.SendKey(" + args + ")"));
            }
            return CommandStatus.NotSupported;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus SendOpcInformation(bool doopc, object o)
        {
            if (m_logger != null)
            {
                string args = doopc.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.SendOpcInformation(" + args + ")"));
            }

            ushort len = 0;

            if (o != null)
                throw new Exception("SendOpcInformation() - non-null OPC information not supported");

            using (Command cmd = new Command(ScsiCommandCode.SendOpcInformation, 10, len, Command.CmdDirection.Out, 60))
            {
                if (doopc)
                    cmd.SetCDB8(1, 1);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }
            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus SetCdSpeed(RotationalControl ctrl, ushort read, ushort write)
        {
            if (m_logger != null)
            {
                string args = ctrl.ToString() + ", " + read.ToString() + ", " + write.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.SetCdSpeed(" + args + ")"));
            }
            using (Command cmd = new Command(ScsiCommandCode.SetCdSpeed, 12, 0, Command.CmdDirection.Out, 30))
            {
                cmd.SetCDB8(1, (byte)ctrl);
                cmd.SetCDB16(2, read);
                cmd.SetCDB16(4, write);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }
            return CommandStatus.Success;
        }

		/// <summary>
		/// 
		/// </summary>
		/// <returns></returns>
		public CommandStatus SetCdSpeedDA(RotationalControl ctrl, ushort read, ushort write)
		{
			if (m_logger != null)
			{
				string args = ctrl.ToString() + ", " + read.ToString() + ", " + write.ToString();
				m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.SetCdSpeed(" + args + ")"));
			}
			using (Command cmd = new Command((ScsiCommandCode)0xDA, 12, 0, Command.CmdDirection.None, 2))
			{
				cmd.SetCDB8(1, (byte)ctrl);
				cmd.SetCDB16(2, read);
				cmd.SetCDB16(4, write);

				CommandStatus st = SendCommand(cmd);
				if (st != CommandStatus.Success)
					return st;
			}
			return CommandStatus.Success;
		}

        /// <summary>
        ///
        /// </summary>
        /// <returns></returns>
        public CommandStatus SetReadAhead(uint trigger, uint lba)
        {
            if (m_logger != null)
            {
                string args = trigger.ToString() + ", " + lba.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.SetReadAhead(" + args + ")"));
            }
            using (Command cmd = new Command(ScsiCommandCode.SetReadAhead, 12, 0, Command.CmdDirection.None, 2))
            {
                cmd.SetCDB32(2, trigger);
                cmd.SetCDB32(6, lba);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }
            return CommandStatus.Success; 
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus SetStreaming(SpeedDescriptor desc)
        {
            if (m_logger != null)
            {
                string args = "desc";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.SetStreaming(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.SetStreaming, 12, 28, Command.CmdDirection.Out, 2))
            {
                cmd.SetCDB8(8, 0);          // Performance Descriptor
                cmd.SetCDB16(9, 28);        // Length of the performance descriptor

                byte b = 0;
                if (desc.Exact)
                    b |= 0x02;
                int wrc = (int)(desc.WRC) ;
                b |= (byte)(wrc << 3);
                cmd.SetBuffer8(0, b);               // Control info, byte 0

                cmd.SetBuffer32(4, 0);                          // Start LBA
                cmd.SetBuffer32(8, (uint)desc.EndLBA);                // End LBA
                cmd.SetBuffer32(12, (uint)desc.ReadSpeed);            // Read size
                cmd.SetBuffer32(16, 1000);                      // Read time
                cmd.SetBuffer32(20, (uint)desc.WriteSpeed);           // Write size
                cmd.SetBuffer32(24, 1000);                      // Write time

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus SetStreaming(RotationalControl rot, int startlba, int endlba, int readsize, int readtime, int writesize, int writetime)
        {
            if (m_logger != null)
            {
                string args = "desc";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.SetStreaming(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.SetStreaming, 12, 28, Command.CmdDirection.Out, 60*5))
            {
                cmd.SetCDB8(8, 0);          // Performance Descriptor
                cmd.SetCDB16(9, 28);        // Length of the performance descriptor

                byte b = 0;
                int wrc = (int)rot;
                b |= (byte)(wrc << 3);
                cmd.SetBuffer8(0, b);               // Control info, byte 0

                cmd.SetBuffer32(4, (uint)startlba);             // Start LBA
                cmd.SetBuffer32(8, (uint)endlba);               // End LBA
                cmd.SetBuffer32(12, (uint)readsize);            // Read size
                cmd.SetBuffer32(16, (uint)readtime);            // Read time
                cmd.SetBuffer32(20, (uint)writesize);           // Write size
                cmd.SetBuffer32(24, (uint)writetime);           // Write time

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }
        
        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus StartStopUnit(bool immd, PowerControl pc, StartState state)
        {
            if (m_logger != null)
            {
                string args = immd.ToString() + ", " + pc.ToString() + ", " + state.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.StartStopUnit(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.StartStopUnit, 6, 0, Command.CmdDirection.None, 30))
            {
                if (immd)
                    cmd.SetCDB8(1, 1) ;

                byte b = (byte)(((byte)pc << 4) | ((byte)state)) ;
                cmd.SetCDB8(4, b) ;

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }
            return CommandStatus.Success; 
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus StopPlayScan()
        {
            using (Command cmd = new Command(ScsiCommandCode.StopPlayScan, 10, 0, Command.CmdDirection.None, 30))
            {
                if (m_logger != null)
                {
                    string args = "";
                    m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.StopPlayScan(" + args + ")"));
                }
                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }
            return CommandStatus.Success; 
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus SynchronizeCache(bool immd)
        {
            if (m_logger != null)
            {
                string args = "";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.SynchronizeCache(" + args + ")"));
            }
            using (Command cmd = new Command(ScsiCommandCode.SyncronizeCache, 10, 0, Command.CmdDirection.None, 30 * 60))
            {
                if (immd)
                    cmd.SetCDB8(1, 2);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }
            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus TestUnitReady(out bool ready)
        {
            if (m_logger != null)
            {
                string args = "out ready";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 9, "Bwg.Scsi.Device.TestUnitReady(" + args + ")"));
            }

            ready = true;
            using (Command cmd = new Command(ScsiCommandCode.TestUnitReady, 6, 0, Command.CmdDirection.None, 60))
            {
                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                {
                    ready = false ;

                    if (st == CommandStatus.DeviceFailed && GetSenseKey() == SenseKeyType.NotReady)
                        st = CommandStatus.Success;

                    return st;
                }
            }
                
            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus Verify(int start, int size)
        {
            if (m_logger != null)
            {
                string args = start.ToString() + ", " + size.ToString();
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.Verify(" + args + ")"));
            }

            using (Command cmd = new Command(ScsiCommandCode.Verify, 10, 2048, Command.CmdDirection.None, 60 * 60))
            {
                cmd.SetCDB32(2, (uint)start);
                cmd.SetCDB16(7, (ushort)size);

                CommandStatus st = SendCommand(cmd);
                if (st != CommandStatus.Success)
                    return st;
            }

            return CommandStatus.Success;
        }


        /// <summary>
        /// This method requests that the SCSI device transfer data from the given buffer to the device
        /// </summary>
        /// <param name="force">If true, the data is forced from the media and cannot be read from the cache</param>
        /// <param name="streaming">If true, this is a streaming read</param>
        /// <param name="lba">The starting logical address for the data</param>
        /// <param name="sector_size">the size of the sector data in bytes</param>
        /// <param name="length">The length of the data to write in sectors</param>
        /// <param name="data">The data buffer to received the data</param>
        /// <returns></returns>
        public CommandStatus Write(bool force, bool streaming, int lba, int sector_size, int length, ref byte[] data)
        {
            if (m_logger != null)
            {
                string args = force.ToString() + ", " + streaming.ToString() + ", " + lba.ToString() + ", " + length.ToString() + ", buffer";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.Write(" + args + ")"));
            }

            Debug.Assert(length == data.GetLength(0) * sector_size);

            fixed (byte *bptr = &data[0])
            {
                IntPtr bufptr = new IntPtr(bptr);
                if (streaming || length > 65535)
                {
                    using (Command cmd = new Command(ScsiCommandCode.Write12, 12, bufptr, length * sector_size, Command.CmdDirection.Out, 5 * 60))
                    {
                        if (force)
                            cmd.SetCDB8(1, 4);              // Set the FUA bit

                        cmd.SetCDB32(2, lba);
                        cmd.SetCDB32(6, length);

                        if (streaming)
                            cmd.SetCDB8(10, 0x80);          // Set the streaming bit

                        CommandStatus st = SendCommand(cmd);
                        if (st != CommandStatus.Success)
                            return st;
                    }
                }
                else
                {
                    using (Command cmd = new Command(ScsiCommandCode.Write, 10, bufptr, length * sector_size, Command.CmdDirection.Out, 5 * 60))
                    {
                        if (force)
                            cmd.SetCDB8(1, 4);              // Set the FUA bit

                        cmd.SetCDB32(2, lba);
                        cmd.SetCDB16(7, (ushort)length);

                        CommandStatus st = SendCommand(cmd);
                        if (st != CommandStatus.Success)
                            return st;
                    }
                }
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// Write data to the device into a memory buffer
        /// </summary>
        /// <param name="force">If true, the data is forced from the media and cannot be read from the cache</param>
        /// <param name="streaming">If true, this is a streaming read</param>
        /// <param name="lba">The starting logical address for the data</param>
        /// <param name="sector_size">the size of a sector in bytes</param>
        /// <param name="length">The length of the data to write in sectors</param>
        /// <param name="data">The buffer to receive the data</param>
        /// <returns></returns>
        public CommandStatus Write(bool force, bool streaming, int lba, int sector_size, int length, IntPtr data)
        {
            if (m_logger != null)
            {
                string args = force.ToString() + ", " + streaming.ToString() + ", " + lba.ToString() + ", " + length.ToString() + ", IntPtr, ";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.Write(" + args + ")"));
            }

            if (streaming || length > 65535)
            {
                using (Command cmd = new Command(ScsiCommandCode.Write12, 12, data, length * sector_size, Command.CmdDirection.Out, 5 * 60))
                {
                    if (force)
                        cmd.SetCDB8(1, 4);          // Set the FUA bit

                    cmd.SetCDB32(2, lba);
                    cmd.SetCDB32(6, length);

                    if (streaming)
                        cmd.SetCDB8(10, 0x80);

                    CommandStatus st = SendCommand(cmd);
                    if (st != CommandStatus.Success)
                    {
                        m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 4, "Write failed at lba " + lba.ToString()));
                        return st;
                    }
                }
            }
            else
            {
                using (Command cmd = new Command(ScsiCommandCode.Write, 10, data, length * sector_size, Command.CmdDirection.Out, 5 * 60))
                {
                    if (force)
                        cmd.SetCDB8(1, 4);              // Set the FUA bit

                    cmd.SetCDB32(2, lba);
                    cmd.SetCDB16(7, (ushort)length);

                    CommandStatus st = SendCommand(cmd);
                    if (st != CommandStatus.Success)
                    {
                        m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 4, "Write failed at lba " + lba.ToString()));
                        return st;
                    }
                }
            }

            return CommandStatus.Success;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus WriteVerify()
        {
            if (m_logger != null)
            {
                string args = "NOT IMPLEMENTED";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.WriteVerify(" + args + ")"));
            }
            return CommandStatus.NotSupported;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public CommandStatus WriteBuffer()
        {
            if (m_logger != null)
            {
                string args = "NOT IMPLEMENTED";
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Debug, 8, "Bwg.Scsi.Device.WriteBuffer(" + args + ")"));
            }
            return CommandStatus.NotSupported;
        }

        #endregion
    }
}
