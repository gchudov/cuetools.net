// SPDX-License-Identifier: GPL-2.0-or-later
using System;
using System.Collections.Generic;
using System.Diagnostics;
using CUETools.AccurateRip;
using CUETools.CDImage;

namespace CUETools.ConsoleRipper
{
    internal static class OffsetCalibration
    {
        internal const int Radius = 5 * 588;
        internal const int FirstFrame = 445;
        internal const int FrameCount = 11;
        private const int RequiredTracks = 3;
        private const int MaximumSamples = 6;

        // The offset-finding checksum covers one frame at frame 450, not a
        // complete track. Scan +/- five frames using a rolling weighted sum.
        internal static Dictionary<int, uint> Checksums(byte[] audio)
        {
            if (audio == null || audio.Length != FrameCount * 2352)
                throw new ArgumentException("Incomplete offset calibration sample.", "audio");
            var samples = new uint[audio.Length / 4];
            for (int i = 0; i < samples.Length; i++)
                samples[i] = (uint)(audio[i * 4] | audio[i * 4 + 1] << 8 |
                    audio[i * 4 + 2] << 16 | audio[i * 4 + 3] << 24);
            var result = new Dictionary<int, uint>();
            unchecked
            {
                uint sum = 0, crc = 0;
                for (int i = 0; i < 588; i++)
                {
                    sum += samples[i];
                    crc += samples[i] * (uint)(i + 1);
                }
                for (int start = 0; start <= 2 * Radius; start++)
                {
                    if (crc != 0) result.Add(start - Radius, crc);
                    if (start == 2 * Radius) break;
                    uint next = samples[start + 588];
                    crc = crc - sum + 588 * next;
                    sum = sum - samples[start] + next;
                }
            }
            return result;
        }

        internal static bool TryCalibrate(CDImageLayout toc, IList<AccDisk> discs,
            Func<int, int, byte[]> read, TimeSpan budget, out int offset, out int matchedTracks)
        {
            offset = 0;
            matchedTracks = 0;
            if (toc.AudioTracks < RequiredTracks || discs == null || discs.Count == 0)
                return false;
            var timer = Stopwatch.StartNew();
            HashSet<int> candidates = null;
            int reads = 0;
            for (int track = toc.FirstAudio; track < toc.FirstAudio + toc.AudioTracks; track++)
            {
                if (reads >= MaximumSamples || timer.Elapsed >= budget) break;
                if (toc[track].Length < FirstFrame + FrameCount) continue;
                var expected = new HashSet<uint>();
                foreach (var disc in discs)
                    if (track <= disc.tracks.Count && disc.tracks[track - 1].count > 0 &&
                        disc.tracks[track - 1].Frame450CRC != 0)
                        expected.Add(disc.tracks[track - 1].Frame450CRC);
                if (expected.Count == 0) continue;
                reads++;
                var checksums = Checksums(read(checked((int)(toc[track].Start - toc[toc.FirstAudio][0].Start) + FirstFrame), FrameCount));
                var matches = new HashSet<int>();
                foreach (var pair in checksums)
                    if (expected.Contains(pair.Value)) matches.Add(pair.Key);
                // Silence/absent matches provide no calibration evidence.
                if (matches.Count == 0) continue;
                matchedTracks++;
                if (candidates == null) candidates = matches;
                else candidates.IntersectWith(matches);
                if (candidates.Count == 0) return false;
                if (matchedTracks >= RequiredTracks && candidates.Count == 1)
                {
                    foreach (int candidate in candidates) offset = candidate;
                    return true;
                }
            }
            return false;
        }
    }
}
