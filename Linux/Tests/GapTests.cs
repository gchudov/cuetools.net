// SPDX-License-Identifier: GPL-2.0-or-later
using System;
using System.Reflection;
using System.Runtime.InteropServices;
using Bwg.Logging;
using Bwg.Scsi;
using CUETools.CDImage;
using CUETools.Codecs;
using CUETools.Ripper.SCSI;

internal static unsafe class GapTests
{
    internal static void Run(Action<bool, string> check)
    {
        CheckRecovery(check, Fault.None, 0);
        CheckRecovery(check, Fault.Crc, 1);
        CheckRecovery(check, Fault.StalePosition, 2);
        CheckRecovery(check, Fault.Transport, 1);
        using (var drive = new GapDrive { Pioneer = true, Failure = Fault.PioneerPacket, FailuresLeft = 2 })
        using (var reader = CreateReader(drive))
        {
            // Five identical, CRC-valid Q packets captured from the BDR-XS06.
            // They say sector 153822 although sectors 153845-153849 were read.
            int boundary = Locate(reader, 153822, 153871, 12);
            check(boundary == 153855 && drive.FailuresLeft == 0,
                "Captured Pioneer stale packets recover to the exact track 12 boundary");
        }
        using (var drive = new GapDrive { Failure = Fault.Crc, FailuresLeft = int.MaxValue })
        using (var reader = CreateReader(drive))
        {
            bool rejected = false;
            try { reader.DetectGaps(); }
            catch (Exception error) { rejected = error.Message.Contains("gap detection failed"); }
            check(rejected && !reader.GapsDetected, "Persistent invalid subchannel data fails without publishing guessed gaps");
            check(drive.Requests <= 3, "Gap failure retries are bounded at the failing search position");
        }
        using (var drive = new GapDrive { Failure = Fault.Crc, FailuresLeft = int.MaxValue })
        using (var reader = CreateReader(drive))
        {
            bool rejected = false;
            try { Locate(reader, 100, 110, 1); }
            catch (TargetInvocationException error) { rejected = error.InnerException.Message.Contains("gap detection failed"); }
            check(rejected && drive.Requests <= 3, "Small search ranges cannot accept persistent bad checksums as a gap");
        }
    }

    private static int Locate(CDDriveReader reader, int first, int last, int track)
    {
        var method = typeof(CDDriveReader).GetMethod("LocateLastSector", BindingFlags.Instance | BindingFlags.NonPublic,
            null, new[] { typeof(int), typeof(int), typeof(int), typeof(int), typeof(int).MakeByRefType(), typeof(int).MakeByRefType() }, null);
        var arguments = new object[] { first, last, track, -1, 1, 0 };
        method.Invoke(reader, arguments);
        return (int)arguments[5];
    }

    private static void CheckRecovery(Action<bool, string> check, Fault fault, int failures)
    {
        using (var drive = new GapDrive { Failure = fault, FailuresLeft = failures })
        using (var reader = CreateReader(drive))
        {
            check(reader.DetectGaps(), "Complete gap detection after " + fault);
            check(reader.GapsDetected && reader.TOC[2].Pregap == 150 && reader.TOC[2][0].Start == 4000,
                "Preserve the actual 150-sector pregap after " + fault);
            check(reader.TOC[1].Start == 0 && reader.TOC[2].Start == 4150 && reader.TOC.AudioLength == 8000,
                "Preserve track starts and audio length after " + fault);
        }
    }

    private static CDDriveReader CreateReader(GapDrive drive)
    {
        var reader = new CDDriveReader();
        var toc = new CDImageLayout();
        if (drive.Pioneer)
            for (int i = 0; i < PioneerStarts.Length - 1; i++)
                toc.AddTrack(new CDTrack((uint)i + 1, (uint)PioneerStarts[i], (uint)(PioneerStarts[i + 1] - PioneerStarts[i]), true, false));
        else
        {
            toc.AddTrack(new CDTrack(1, 0, 4150, true, false));
            toc.AddTrack(new CDTrack(2, 4150, 3850, true, false));
        }
        Set(reader, "m_device", drive);
        Set(reader, "_toc", toc);
        Set(reader, "_toc2", toc.Clone());
        Set(reader, "readCommandFound", true);
        var field = typeof(CDDriveReader).GetField("_gapDetection", BindingFlags.Instance | BindingFlags.NonPublic);
        field.SetValue(reader, Enum.Parse(field.FieldType, "ReadCD"));
        return reader;
    }

    private static void Set(CDDriveReader reader, string name, object value)
    {
        typeof(CDDriveReader).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic).SetValue(reader, value);
    }

    private static readonly int[] PioneerStarts = { 0, 11712, 22399, 38095, 48014, 58738, 72879, 81958,
        89966, 103812, 123754, 138355, 153856, 170700, 179533, 190427, 204292 };
    private enum Fault { None, Crc, StalePosition, Transport, PioneerPacket }

    private sealed class GapDrive : Device
    {
        internal Fault Failure;
        internal bool Pioneer;
        internal int FailuresLeft, Requests;
        private readonly Crc16Ccitt crc = new Crc16Ccitt(InitialCrcValue.Zeros);
        internal GapDrive() : base(new Logger()) { }

        protected override bool ScsiControl(IntPtr pointer)
        {
            var request = (SgIoHeader*)pointer;
            byte* command = (byte*)request->Command;
            if (command[0] != 0xbe || command[9] != 0 || command[10] != 2)
                throw new Exception("Unexpected command in subchannel test.");
            int start = command[2] << 24 | command[3] << 16 | command[4] << 8 | command[5];
            int sectors = command[6] << 16 | command[7] << 8 | command[8];
            Requests++;
            bool fail = FailuresLeft > 0;
            if (fail) FailuresLeft--;
            if (fail && Failure == Fault.Transport) { request->HostStatus = 3; return true; }
            var data = new byte[sectors * 16];
            for (int i = 0; i < sectors; i++)
            {
                if (fail && Failure == Fault.PioneerPacket)
                {
                    Convert.FromHexString("0112010326170034127263b400000000").CopyTo(data, i * 16);
                    continue;
                }
                int position = fail && Failure == Fault.StalePosition ? 0 : start + i;
                int offset = i * 16, absolute = position + 150;
                data[offset] = 1;
                data[offset + 1] = Pioneer ? Bcd(Array.FindLastIndex(PioneerStarts, value => value <= position) + 1) : (byte)(position < 4000 ? 1 : 2);
                data[offset + 2] = (byte)(!Pioneer && position >= 4000 && position < 4150 ? 0 : 1);
                data[offset + 7] = Bcd(absolute / 4500);
                data[offset + 8] = Bcd(absolute / 75 % 60);
                data[offset + 9] = Bcd(absolute % 75);
                ushort checksum = (ushort)(crc.ComputeChecksum(data, offset, 10) ^ 0xffff);
                data[offset + 10] = (byte)(checksum >> 8);
                data[offset + 11] = (byte)checksum;
                if (fail && Failure == Fault.Crc) data[offset + 3] ^= 1;
            }
            Marshal.Copy(data, 0, request->Data, data.Length);
            return true;
        }
        private static byte Bcd(int value) { return (byte)(value / 10 * 16 + value % 10); }
    }
}
