// SPDX-License-Identifier: GPL-2.0-or-later
using System;
using System.Runtime.InteropServices;
using Bwg.Logging;

namespace Bwg.Scsi
{
    public unsafe partial class Device
    {
        // Linux sg_io_hdr_t: natural native alignment, including pointer-sized fields.
        [StructLayout(LayoutKind.Sequential)]
        internal struct SgIoHeader
        {
            public int InterfaceId, Direction;
            public byte CommandLength, SenseLength;
            public ushort VectorCount;
            public uint TransferLength;
            public IntPtr Data, Command, Sense;
            public uint Timeout, Flags;
            public int PackId;
            public IntPtr User;
            public byte Status, MaskedStatus, MessageStatus, SenseWritten;
            public ushort HostStatus, DriverStatus;
            public int Residual;
            public uint Duration, Info;
        }

        private CommandStatus SendCommandLinux(Command cmd)
        {
            byte* cdb = stackalloc byte[16];
            byte* sense = stackalloc byte[252];
            new Span<byte>(sense, 252).Clear();
            if (cmd.GetCDBLength() > 16) throw new ArgumentException("SCSI command exceeds 16 bytes.");
            for (int i = 0; i < cmd.GetCDBLength(); i++) cdb[i] = cmd.GetCDB((byte)i);
            var request = new SgIoHeader
            {
                InterfaceId = 'S',
                Direction = cmd.Direction == Command.CmdDirection.In ? -3 : cmd.Direction == Command.CmdDirection.Out ? -2 : -1,
                CommandLength = (byte)cmd.GetCDBLength(), SenseLength = 252,
                TransferLength = (uint)cmd.BufferSize, Data = cmd.GetBuffer(),
                Command = (IntPtr)cdb, Sense = (IntPtr)sense,
                Timeout = checked((uint)Math.Max(1, cmd.TimeOut) * 1000)
            };
            m_scsi_status = 0;
            m_sense_info = null;
            if (!ScsiControl((IntPtr)(&request)))
            {
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Error, 0, "SG_IO failed: " + Win32ErrorToString(LastError)));
                return CommandStatus.IoctlFailed;
            }
            m_scsi_status = request.Status;
            if (request.SenseWritten > 0)
                m_sense_info = NormalizeSense(new ReadOnlySpan<byte>(sense, request.SenseWritten));
            // DRIVER_SENSE (8) accompanies a valid SCSI CHECK CONDITION; other
            // driver/host failures must never be accepted as successfully read audio.
            if (request.HostStatus != 0 || (request.DriverStatus & 0x0f) != 0 && (request.DriverStatus & 0x0f) != 8)
            {
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Error, 0,
                    string.Format("SG_IO transport error: host={0:X}, driver={1:X}", request.HostStatus, request.DriverStatus)));
                return CommandStatus.IoctlFailed;
            }
            if (request.Status != 0 || (request.Info & 1) != 0)
            {
                LogSenseInformation(cmd);
                return CommandStatus.DeviceFailed;
            }
            // INQUIRY/TOC may legitimately return less than their allocation size.
            // Audio sector reads may not: stale/zero-filled tail bytes corrupt a rip.
            if (request.Residual != 0 && (cdb[0] == 0xbe || cdb[0] == 0xb9 || cdb[0] == 0xd8))
            {
                m_logger.LogMessage(new UserMessage(UserMessage.Category.Error, 0, "Incomplete audio sector transfer."));
                return CommandStatus.IoctlFailed;
            }
            return CommandStatus.Success;
        }

        internal static byte[] NormalizeSense(ReadOnlySpan<byte> sense)
        {
            var result = new byte[Math.Max(24, sense.Length)];
            if (sense.Length >= 4 && ((sense[0] & 0x7f) == 0x72 || (sense[0] & 0x7f) == 0x73))
            {
                // Bwg's MMC error handling consumes fixed-format sense fields.
                result[0] = 0x70; result[2] = (byte)(sense[1] & 0x0f);
                result[7] = 10; result[12] = sense[2]; result[13] = sense[3];
            }
            else sense.CopyTo(result);
            return result;
        }
    }
}
