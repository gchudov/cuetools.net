// SPDX-License-Identifier: GPL-2.0-or-later
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using Bwg.Logging;
using Bwg.Scsi;
using CUETools.Codecs;
using CUETools.Codecs.Flake;

internal static unsafe class Program
{
    private static int count;
    private static void Check(bool condition, string message)
    {
        if (!condition) throw new Exception(message);
        count++;
    }

    private sealed class Drive : Device
    {
        internal Action<IntPtr> Reply;
        internal bool Fail;
        internal Drive() : base(new Logger()) { }
        protected override bool ScsiControl(IntPtr request)
        {
            var header = (SgIoHeader*)request;
            Check(header->InterfaceId == 'S' && header->VectorCount == 0, "SG request header");
            Check(header->SenseLength >= 24 && header->Timeout > 0, "Sense buffer and timeout");
            Reply?.Invoke(request);
            return !Fail;
        }
    }

    public static int Main()
    {
        try
        {
            // Values independently checked against Linux's <scsi/sg.h> in abi.c.
            Check(Marshal.SizeOf<Device.SgIoHeader>() == (IntPtr.Size == 8 ? 88 : 64), "sg_io_hdr ABI size");
            Check(Marshal.OffsetOf<Device.SgIoHeader>("Data").ToInt32() == 16, "SG data pointer offset");
            Check(Marshal.OffsetOf<Device.SgIoHeader>("Residual").ToInt32() == (IntPtr.Size == 8 ? 72 : 52), "SG residual offset");
            using (var drive = new Drive())
            {
                drive.Reply = pointer =>
                {
                    var request = (Device.SgIoHeader*)pointer;
                    Check(request->Direction == -3 && request->CommandLength == 6 && request->Timeout == 10000, "INQUIRY request");
                    Check(Marshal.ReadByte(request->Command) == 0x12, "INQUIRY opcode");
                    var data = new byte[36]; data[0] = 5; data[1] = 0x80; data[3] = 2; data[4] = 31;
                    Encoding.ASCII.GetBytes("TEST    OPTICAL DRIVE   1.00").CopyTo(data, 8);
                    Marshal.Copy(data, 0, request->Data, data.Length);
                };
                Check(drive.Inquiry(out var inquiry) == Device.CommandStatus.Success && inquiry.Valid && inquiry.PeripheralDeviceType == 5, "Optical inquiry decoding");
                byte* audio = stackalloc byte[2352];
                drive.Reply = pointer =>
                {
                    var request = (Device.SgIoHeader*)pointer;
                    Check(request->Direction == -3 && request->CommandLength == 12 && request->TransferLength == 2352, "READ CD transfer");
                    Check(Marshal.ReadByte(request->Command) == 0xbe && Marshal.ReadByte(request->Command, 5) == 42 && Marshal.ReadByte(request->Command, 8) == 1, "READ CD sector addressing");
                    request->Residual = 4;
                };
                Check(drive.ReadCD(1, false, 42, 1, (IntPtr)audio, 2352) == Device.CommandStatus.IoctlFailed, "Reject incomplete audio transfers");
                drive.Reply = pointer =>
                {
                    var request = (Device.SgIoHeader*)pointer;
                    Check(request->TransferLength == 16 * 16, "Subchannel-only transfer excludes audio bytes");
                    Check(Marshal.ReadByte(request->Command, 9) == 0 && Marshal.ReadByte(request->Command, 10) == 2, "Q-only READ CD command");
                };
                Check(drive.ReadCDAndSubChannel(Device.MainChannelSelection.None, Device.SubChannelMode.QOnly,
                    Device.C2ErrorMode.None, 1, false, 3, 16, (IntPtr)audio, 10) == Device.CommandStatus.Success,
                    "Subchannel-only reads do not trigger incomplete-audio rejection");
                drive.Reply = pointer => { ((Device.SgIoHeader*)pointer)->HostStatus = 3; };
                Check(drive.Inquiry(out _) == Device.CommandStatus.IoctlFailed, "Transport failure is not audio");
                drive.Reply = pointer =>
                {
                    var request = (Device.SgIoHeader*)pointer;
                    request->Status = 2; request->DriverStatus = 8; request->Info = 1; request->SenseWritten = 8;
                    Marshal.Copy(new byte[] { 0x72, 2, 0x3a, 0, 0, 0, 0, 0 }, 0, request->Sense, 8);
                };
                Check(drive.Inquiry(out _) == Device.CommandStatus.DeviceFailed && drive.GetSenseAsc() == 0x3a, "Descriptor-format no-disc sense");
                drive.Reply = null; drive.Fail = true;
                Check(drive.Inquiry(out _) == Device.CommandStatus.IoctlFailed && drive.GetSenseLength() == 0, "Clear stale sense after ioctl failure");
            }
            bool rejected = false;
            try { CUETools.Ripper.OpticalDrives.Resolve("/dev/null"); } catch (IOException) { rejected = true; }
            Check(rejected, "Reject non-optical devices");
            RoundTrip();
            Console.WriteLine("Passed " + count + " Linux CUE Ripper checks.");
            return 0;
        }
        catch (Exception error) { Console.Error.WriteLine(error); return 1; }
    }

    private static void RoundTrip()
    {
        string file = Path.Combine(Path.GetTempPath(), "cueripper-" + Guid.NewGuid().ToString("N") + ".flac");
        const int length = 44100;
        var samples = new int[length, 2];
        for (int i = 0; i < length; i++)
        {
            samples[i, 0] = (short)(i * 71);
            samples[i, 1] = (short)(Math.Sin(i * 0.17) * 32000);
        }
        try
        {
            var settings = new EncoderSettings { PCM = AudioPCMConfig.RedBook };
            settings.SetEncoderModeIndex(8);
            var encoder = new AudioEncoder(settings, file) { FinalSampleCount = length };
            try { encoder.Write(new AudioBuffer(AudioPCMConfig.RedBook, samples, length)); }
            finally { encoder.Close(); }
            using (var tags = TagLib.File.Create(file))
            {
                tags.Tag.Album = "Linux ripping test — 音楽";
                var xiph = (TagLib.Ogg.XiphComment)tags.GetTag(TagLib.TagTypes.Xiph);
                xiph.SetField("CUESHEET", "FILE \"test.flac\" WAVE\n  TRACK 01 AUDIO\n    INDEX 01 00:00:00\n");
                xiph.SetField("LOG", "AccurateRip summary\nTest fixture (no submitted disc).\n");
                tags.Save();
            }
            using (var tags = TagLib.File.Create(file))
                Check(tags.Tag.Album == "Linux ripping test — 音楽", "Unicode FLAC tags");
            var decoder = new AudioDecoder(new DecoderSettings(), file);
            try
            {
                Check(decoder.Length == length, "FLAC sample count");
                var buffer = new AudioBuffer(decoder, 4096);
                int position = 0;
                while (decoder.Read(buffer, -1) != 0)
                {
                    for (int i = 0; i < buffer.Length; i++, position++)
                        if (buffer.Samples[i, 0] != samples[position, 0] || buffer.Samples[i, 1] != samples[position, 1])
                            throw new Exception("FLAC PCM mismatch at " + position);
                }
                Check(position == length, "FLAC round trip is sample-exact after tagging");
            }
            finally { decoder.Close(); }
            using (var exclusive = new FileStream(file, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
                Check(exclusive.Length > 0, "All rip/tagging streams closed");
        }
        finally { File.Delete(file); }
    }
}
