// SPDX-License-Identifier: GPL-2.0-or-later
using System;
using System.Collections.Generic;
using CUETools.AccurateRip;
using CUETools.CDImage;
using CUETools.Codecs;
using CUETools.ConsoleRipper;

internal static class OffsetCalibrationTests
{
    internal static void Run(Action<bool, string> check)
    {
        var toc = new CDImageLayout("0 600 1200 1800 2400 3000 3600 4200");
        byte[] audio = new byte[4200 * 2352];
        uint state = 1729;
        for (int i = 0; i < audio.Length; i++)
        {
            state ^= state << 13; state ^= state >> 17; state ^= state << 5;
            audio[i] = (byte)state;
        }
        var verify = new AccurateRipVerify(toc, null);
        var buffer = new AudioBuffer(AudioPCMConfig.RedBook, 600 * 588);
        for (int i = 0; i < 7; i++)
        {
            var block = new byte[600 * 2352];
            Buffer.BlockCopy(audio, i * block.Length, block, 0, block.Length);
            buffer.Prepare(block, 600 * 588);
            verify.Write(buffer);
        }
        var disc = new AccDisk();
        for (int track = 0; track < 7; track++)
            disc.tracks.Add(new AccTrack { count = 2, Frame450CRC = verify.CRC450(track, 0) });
        var discs = new List<AccDisk> { disc };
        int reads = 0;
        Func<int, int, int, byte[]> sample = (sector, count, correction) =>
        {
            check(count == 11, "Read only eleven sectors for offset calibration");
            reads++;
            var data = new byte[count * 2352];
            Buffer.BlockCopy(audio, sector * 2352 - correction * 4, data, 0, data.Length);
            return data;
        };
        // The existing whole-disc verifier is the oracle, independent of the
        // new rolling checksum. Check signs, exact zero and both search edges.
        foreach (int correction in new[] { 0, 6, -667, 2940, -2940 })
        {
            reads = 0;
            int offset, tracks;
            bool found = OffsetCalibration.TryCalibrate(toc, discs,
                (s, c) => sample(s, c, correction), TimeSpan.FromSeconds(5), out offset, out tracks);
            check(found && offset == correction && tracks == 3 && reads == 3,
                "Three independent frame-450 checksums establish correction " + correction);
        }
        var allChecksums = OffsetCalibration.Checksums(sample(445, 11, 0));
        foreach (int correction in new[] { -2939, -667, 0, 6, 123, 2939 })
            check(allChecksums[correction] == verify.CRC450(0, correction), "Rolling checksum agrees with AccurateRip at " + correction);

        int result, matched;
        reads = 0;
        check(!OffsetCalibration.TryCalibrate(toc, new List<AccDisk>(),
            (s, c) => sample(s, c, 0), TimeSpan.FromSeconds(5), out result, out matched) && reads == 0,
            "Missing database performs no drive reads");
        check(!OffsetCalibration.TryCalibrate(toc, discs,
            (s, c) => sample(s, c, 0), TimeSpan.Zero, out result, out matched) && reads == 0,
            "Expired budget performs no drive reads");
        check(!OffsetCalibration.TryCalibrate(toc, discs,
            (s, c) => new byte[c * 2352], TimeSpan.FromSeconds(5), out result, out matched),
            "Silence cannot establish an offset");

        var pressing = new AccDisk();
        for (int track = 0; track < 7; track++)
            pressing.tracks.Add(new AccTrack { count = 20, Frame450CRC = verify.CRC450(track, 123) });
        discs.Add(pressing);
        check(!OffsetCalibration.TryCalibrate(toc, discs,
            (s, c) => sample(s, c, 0), TimeSpan.FromSeconds(5), out result, out matched),
            "Multiple pressings remain ambiguous even if one has higher confidence");
        discs.Remove(pressing);
        check(!OffsetCalibration.TryCalibrate(toc, discs,
            (s, c) => sample(s, c, s < 600 ? 6 : 7), TimeSpan.FromSeconds(5), out result, out matched),
            "Disagreement between tracks rejects calibration");
        check(OffsetCalibration.TryCalibrate(toc, discs,
            (s, c) => s < 600 ? new byte[c * 2352] : sample(s, c, 6), TimeSpan.FromSeconds(5), out result, out matched)
            && result == 6 && matched == 3, "Skip silent intro and require three other tracks");
        reads = 0;
        check(!OffsetCalibration.TryCalibrate(toc, discs,
            (s, c) => sample(s, c, 8000), TimeSpan.FromSeconds(5), out result, out matched) && reads == 6,
            "Unmatched disc stops after six short samples");
        check(!OffsetCalibration.TryCalibrate(toc, discs,
            (s, c) => s < 1200 ? sample(s, c, 6) : new byte[c * 2352], TimeSpan.FromSeconds(5), out result, out matched),
            "Two matching tracks are insufficient");

        var mixed = new CDImageLayout(4, 3, 2, "0 1000 1600 2200 2800");
        var mixedDisc = new AccDisk();
        mixedDisc.tracks.Add(new AccTrack());
        mixedDisc.tracks.AddRange(disc.tracks.GetRange(0, 3));
        check(OffsetCalibration.TryCalibrate(mixed, new List<AccDisk> { mixedDisc },
            (s, c) => sample(s, c, 6), TimeSpan.FromSeconds(5), out result, out matched) && result == 6,
            "Mixed-mode audio is relative to the first audio sector, database entries retain physical track numbers");
        bool incomplete = false;
        try { OffsetCalibration.Checksums(new byte[2352]); }
        catch (ArgumentException) { incomplete = true; }
        check(incomplete, "Truncated sample rejected");
    }
}
