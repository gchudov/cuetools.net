// SPDX-License-Identifier: GPL-2.0-or-later
using System;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using Bwg.Logging;
using Bwg.Scsi;
using CUETools.CDImage;
using CUETools.Ripper.SCSI;

internal static unsafe class CalibrationReadTests
{
    internal static void Run(Action<bool, string> check)
    {
        using (var drive = new SampleDrive())
        using (var reader = new CDDriveReader())
        {
            Set(reader, "m_device", drive);
            Set(reader, "_toc", new CDImageLayout("0 600 1200 1800"));
            Set(reader, "readCommandFound", true);
            Set(reader, "m_max_sectors", 16);
            var command = typeof(CDDriveReader).GetField("_readCDCommand", BindingFlags.Instance | BindingFlags.NonPublic);
            command.SetValue(reader, Enum.Parse(command.FieldType, "ReadCdBEh"));
            Set(reader, "_c2ErrorMode", Device.C2ErrorMode.None);
            reader.UserData = new long[11, 2, 2352];
            reader.C2Count = new byte[11, 294];
            reader.DriveOffset = 667;
            reader.Position = 123;
            reader.CorrectionQuality = 2;
            byte[] data = reader.ReadOffsetCalibrationSample(445, 11);
            check(drive.Requests == 2 && data.Length == 11 * 2352 && data.All(b => b == 23),
                "Calibration reads exactly eleven sectors twice and returns secure PCM");
            check(reader.Position == 123 && reader.DriveOffset == 667 && reader.CorrectionQuality == 2 && reader.Timeout == 10,
                "Calibration restores position, offset, correction quality and timeout");
            check(reader.RetryCount.All(b => b == 3), "Calibration does not contaminate rip retry history");
            drive.Mismatch = true;
            bool rejected = false;
            try { reader.ReadOffsetCalibrationSample(445, 11); }
            catch (ReadCDException) { rejected = true; }
            check(rejected && drive.Requests == 4, "Conflicting reads fail after two passes instead of prolonged retries");
            check(reader.Position == 123 && reader.CorrectionQuality == 2 && reader.RetryCount.All(b => b == 3),
                "Failed calibration also restores the rip state");
            drive.Fail = true;
            rejected = false;
            try { reader.ReadOffsetCalibrationSample(445, 11); }
            catch (SCSIException) { rejected = true; }
            check(rejected && drive.Requests == 5 && reader.Timeout == 10 && reader.Position == 123,
                "Transport error stops promptly and restores reader state");
        }
    }

    private static void Set(CDDriveReader reader, string field, object value)
    {
        typeof(CDDriveReader).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic).SetValue(reader, value);
    }

    private sealed class SampleDrive : Device
    {
        internal int Requests;
        internal bool Mismatch, Fail;
        internal SampleDrive() : base(new Logger()) { }
        protected override bool ScsiControl(IntPtr pointer)
        {
            var request = (SgIoHeader*)pointer;
            byte* command = (byte*)request->Command;
            int start = command[2] << 24 | command[3] << 16 | command[4] << 8 | command[5];
            int sectors = command[6] << 16 | command[7] << 8 | command[8];
            if (command[0] != 0xbe || start != 445 || sectors != 11 || request->Timeout != 2000)
                throw new Exception("Calibration command exceeded its sector or time bounds.");
            Requests++;
            if (Fail) { request->HostStatus = 3; return true; }
            var data = Enumerable.Repeat((byte)(Mismatch && Requests % 2 == 0 ? 24 : 23), 11 * 2352).ToArray();
            Marshal.Copy(data, 0, request->Data, data.Length);
            return true;
        }
    }
}
