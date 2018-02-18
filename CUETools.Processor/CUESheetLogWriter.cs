using System;
using System.IO;
using CUETools.AccurateRip;
using CUETools.CDImage;
using CUETools.Ripper;
using System.Globalization;

namespace CUETools.Processor
{
    public class CUESheetLogWriter
    {
        #region AccurateRip

        public static void WriteAccurateRipLog(CUESheet sheet, TextWriter writer)
        {
            writer.WriteLine("[CUETools log; Date: {0}; Version: {1}]", DateTime.Now, CUESheet.CUEToolsVersion);
            if (sheet.PreGapLength != 0)
                writer.WriteLine("Pregap length {0}.", sheet.PreGapLengthMSF);
            if (!sheet.TOC[1].IsAudio)
                writer.WriteLine("Playstation type data track length {0}.", sheet.TOC[sheet.TOC.FirstAudio].StartMSF);
            if (!sheet.TOC[sheet.TOC.TrackCount].IsAudio)
                writer.WriteLine("CD-Extra data track length {0}.",
                    sheet.TOC[sheet.TOC.TrackCount].Length == 0 && sheet.MinDataTrackLength.HasValue ?
                        CDImageLayout.TimeToString(sheet.MinDataTrackLength.Value) + " - " + CDImageLayout.TimeToString(sheet.MinDataTrackLength.Value + 74) :
                        sheet.TOC[sheet.TOC.TrackCount].LengthMSF);
            if (sheet.CDDBDiscIdTag != null && AccurateRipVerify.CalculateCDDBId(sheet.TOC).ToUpper() != sheet.CDDBDiscIdTag.ToUpper() && !sheet.MinDataTrackLength.HasValue)
                writer.WriteLine("CDDBId mismatch: {0} vs {1}", sheet.CDDBDiscIdTag.ToUpper(), AccurateRipVerify.CalculateCDDBId(sheet.TOC).ToUpper());
            if (sheet.AccurateRipId != null && AccurateRipVerify.CalculateAccurateRipId(sheet.TOC) != sheet.AccurateRipId)
                writer.WriteLine("Using preserved id, actual id is {0}.", AccurateRipVerify.CalculateAccurateRipId(sheet.TOC));
            if (sheet.Truncated4608)
                writer.WriteLine("Truncated 4608 extra samples in some input files.");
            if (sheet.PaddedToFrame)
                writer.WriteLine("Padded some input files to a frame boundary.");

            if (!sheet.Processed)
            {
                if (sheet.IsUsingCUEToolsDB) sheet.GenerateCTDBLog(writer);
                if (sheet.IsUsingAccurateRip)
                writer.WriteLine("[AccurateRip ID: {0}] {1}.", sheet.AccurateRipId ?? AccurateRipVerify.CalculateAccurateRipId(sheet.TOC), sheet.ArVerify.ARStatus ?? "found");
                return;
            }

            if (sheet.HDCDDecoder != null && string.Format("{0:s}", sheet.HDCDDecoder) != "")
                writer.WriteLine("HDCD: {0:f}", sheet.HDCDDecoder);
            if (0 != sheet.WriteOffset)
                writer.WriteLine("Offset applied: {0}", sheet.WriteOffset);
            if (sheet.IsUsingCUEToolsDBFix)// && _CUEToolsDB.SelectedEntry != null)
                writer.WriteLine("CUETools DB: corrected {0} errors.", sheet.CTDB.SelectedEntry.repair.CorrectableErrors);
            else if (sheet.IsUsingCUEToolsDB)
                sheet.GenerateCTDBLog(writer);
            sheet.ArVerify.GenerateFullLog(writer, sheet.Config.arLogVerbose, sheet.AccurateRipId ?? AccurateRipVerify.CalculateAccurateRipId(sheet.TOC));
        }

        public static string GetAccurateRipLog(CUESheet sheet)
        {
            using (StringWriter stringWriter = new StringWriter())
            {
                WriteAccurateRipLog(sheet, stringWriter);
                return stringWriter.ToString();
            }
        }

        #endregion

        #region TOC

        public static string GetTOCContents(CUESheet sheet)
        {
            StringWriter sw = new StringWriter();
            sw.WriteLine("     Track |   Start  |  Length  | Start sector | End sector ");
            sw.WriteLine("    ---------------------------------------------------------");
            for (int track = 1; track <= sheet.TOC.TrackCount; track++)
            {
                sw.WriteLine("{0,9}  | {1,8} | {2,8} |  {3,8}    | {4,8}   ",
                    track, // sheet.TOC[track].Number,
                    CDImageLayout.TimeToString(sheet.TOC[track].Start, "{0,2}:{1:00}.{2:00}"),
                    CDImageLayout.TimeToString(sheet.TOC[track].Length, "{0,2}:{1:00}.{2:00}"),
                    sheet.TOC[track].Start,
                    sheet.TOC[track].End);
            }
            return sw.ToString();
        }

        #endregion

        #region EAC

        private static double GetRangeQuality(CUESheet sheet, uint start, uint length)
        {
            int retrySectorsCount = 0;
            for (int i = 0; i < (int)length; i++)
                retrySectorsCount += sheet.CDRipper.RetryCount[(int)start - (int)sheet.TOC[sheet.TOC.FirstAudio][0].Start + i];
#if LOGQ
            int max_scans = (16 << sheet.CDRipper.CorrectionQuality) - 1;
            return 100 * (1.0 - Math.Log(retrySectorsCount / 100.0 + 1) / Math.Log(max_scans * (int)length / 100.0 + 1));
#else
            return 100.0 * (sheet.CDRipper.CorrectionQuality + 1) * length / retrySectorsCount;
#endif
        }

        public static string GetExactAudioCopyLog(CUESheet sheet)
        {
            StringWriter logWriter = new StringWriter(CultureInfo.InvariantCulture);
            string eacHeader = "{7}\r\n" +
                "\r\n" +
                "EAC extraction logfile from {0:d'.' MMMM yyyy', 'H':'mm}\r\n" +
                "\r\n" +
                "{1} / {2}\r\n" +
                "\r\n" +
                "Used drive  : {3}   Adapter: 1  ID: 0\r\n" +
                "\r\n" +
                "Read mode               : {4}\r\n" +
                "Utilize accurate stream : Yes\r\n" +
                "Defeat audio cache      : Yes\r\n" +
                "Make use of C2 pointers : No\r\n" +
                "\r\n" +
                "Read offset correction                      : {5}\r\n" +
                "Overread into Lead-In and Lead-Out          : No\r\n" +
                "Fill up missing offset samples with silence : Yes\r\n" +
                "Delete leading and trailing silent blocks   : No\r\n" +
                "Null samples used in CRC calculations       : Yes\r\n" +
                "Used interface                              : Native Win32 interface for Win NT & 2000\r\n" +
                "{6}" +
                "\r\n" +
                "Used output format : Internal WAV Routines\r\n" +
                "Sample format      : 44.100 Hz; 16 Bit; Stereo\r\n";

            logWriter.WriteLine(eacHeader,
                DateTime.Now,
                sheet.Metadata.Artist, sheet.Metadata.Title,
                sheet.CDRipper.EACName,
                sheet.CDRipper.CorrectionQuality > 0 ? "Secure" : "Burst",
                sheet.CDRipper.DriveOffset,
                (sheet.OutputStyle == CUEStyle.SingleFile || sheet.OutputStyle == CUEStyle.SingleFileWithCUE) ? "" :
                    "Gap handling                                : " +
                    (sheet.CDRipper.GapsDetected ? "Appended to previous track\r\n" : "Not detected, thus appended to previous track\r\n"),
                sheet.CDRipper.RipperVersion); // "Exact Audio Copy V0.99 prebeta 4 from 23. January 2008"

            logWriter.WriteLine();
            logWriter.WriteLine("TOC of the extracted CD");
            logWriter.WriteLine();
            logWriter.Write(GetTOCContents(sheet));
            logWriter.WriteLine();

            bool htoaToFile = ((sheet.OutputStyle == CUEStyle.GapsAppended) && sheet.Config.preserveHTOA &&
                (sheet.TOC.Pregap != 0));
            int accurateTracks = 0, knownTracks = 0;
            bool wereErrors = false;
            if (sheet.OutputStyle != CUEStyle.SingleFile && sheet.OutputStyle != CUEStyle.SingleFileWithCUE)
            {
                logWriter.WriteLine();
                for (int track = 0; track < sheet.TOC.AudioTracks; track++)
                {
                    logWriter.WriteLine("Track {0,2}", track + 1);
                    logWriter.WriteLine();
                    logWriter.WriteLine("     Filename {0}", Path.ChangeExtension(Path.GetFullPath(sheet.DestPaths[track + (htoaToFile ? 1 : 0)]), ".wav"));
                    if (sheet.TOC[track + sheet.TOC.FirstAudio].Pregap > 0 || track + sheet.TOC.FirstAudio == 1)
                    {
                        logWriter.WriteLine();
                        logWriter.WriteLine("     Pre-gap length  0:{0}.{1:00}", CDImageLayout.TimeToString(sheet.TOC[track + sheet.TOC.FirstAudio].Pregap + (track + sheet.TOC.FirstAudio == 1 ? 150U : 0U), "{0:00}:{1:00}"), (sheet.TOC[track + sheet.TOC.FirstAudio].Pregap % 75) * 100 / 75);
                    }

                    wereErrors |= sheet.PrintErrors(logWriter, sheet.TOC[track + sheet.TOC.FirstAudio].Start, sheet.TOC[track + sheet.TOC.FirstAudio].Length);

                    logWriter.WriteLine();
                    logWriter.WriteLine("     Peak level {0:F1} %", (sheet.ArVerify.PeakLevel(track + 1) * 1000 / 65534) * 0.1);
                    logWriter.WriteLine("     Track quality {0:F1} %", GetRangeQuality(sheet, sheet.TOC[track + sheet.TOC.FirstAudio].Start, sheet.TOC[track + sheet.TOC.FirstAudio].Length));
                    if (sheet.ArTestVerify != null)
                    logWriter.WriteLine("     Test CRC {0:X8}", sheet.ArTestVerify.CRC32(track + 1));
                    logWriter.WriteLine("     Copy CRC {0:X8}", sheet.ArVerify.CRC32(track + 1));
                    if (sheet.ArVerify.Total(track) == 0)
                        logWriter.WriteLine("     Track not present in AccurateRip database");
                    else
                    {
                        knownTracks++;
                        if (sheet.ArVerify.Confidence(track) == 0)
                            logWriter.WriteLine("     Cannot be verified as accurate (confidence {0})  [{1:X8}], AccurateRip returned [{2:X8}]", sheet.ArVerify.Total(track), sheet.ArVerify.CRC(track), sheet.ArVerify.DBCRC(track));
                        else
                        {
                            logWriter.WriteLine("     Accurately ripped (confidence {0})  [{1:X8}]", sheet.ArVerify.Confidence(track), sheet.ArVerify.CRC(track));
                            accurateTracks++;
                        }
                    }
                    logWriter.WriteLine("     Copy OK");
                    logWriter.WriteLine();
                }
            }
            else
            {
                logWriter.WriteLine();
                logWriter.WriteLine("Range status and errors");
                logWriter.WriteLine();
                logWriter.WriteLine("Selected range");
                logWriter.WriteLine();
                logWriter.WriteLine("     Filename {0}", Path.ChangeExtension(Path.GetFullPath(sheet.DestPaths[0]), ".wav"));
                wereErrors = sheet.PrintErrors(logWriter, sheet.TOC[sheet.TOC.FirstAudio][0].Start, sheet.TOC.AudioLength);
                logWriter.WriteLine();
                logWriter.WriteLine("     Peak level {0:F1} %", (sheet.ArVerify.PeakLevel() * 1000 / 65535) * 0.1);
                logWriter.WriteLine("     Range quality {0:F1} %", GetRangeQuality(sheet, sheet.TOC[sheet.TOC.FirstAudio][0].Start, sheet.TOC.AudioLength));
                if (sheet.ArTestVerify != null)
                logWriter.WriteLine("     Test CRC {0:X8}", sheet.ArTestVerify.CRC32(0));
                logWriter.WriteLine("     Copy CRC {0:X8}", sheet.ArVerify.CRC32(0));
                logWriter.WriteLine("     Copy OK");
                logWriter.WriteLine();
                if (wereErrors)
                    logWriter.WriteLine("There were errors");
                else
                    logWriter.WriteLine("No errors occurred");
                logWriter.WriteLine();
                logWriter.WriteLine();
                logWriter.WriteLine("AccurateRip summary");
                logWriter.WriteLine();
                for (int track = 0; track < sheet.TOC.AudioTracks; track++)
                {
                    if (sheet.ArVerify.Total(track) == 0)
                        logWriter.WriteLine("Track {0,2}  not present in database", track + 1);
                    else
                    {
                        knownTracks++;
                        if (sheet.ArVerify.Confidence(track) == 0)
                            logWriter.WriteLine("Track {3,2}  cannot be verified as accurate (confidence {0})  [{1:X8}], AccurateRip returned [{2:X8}]", sheet.ArVerify.Total(track), sheet.ArVerify.CRC(track), sheet.ArVerify.DBCRC(track), track + 1);
                        else
                        {
                            logWriter.WriteLine("Track {2,2}  accurately ripped (confidence {0})  [{1:X8}]", sheet.ArVerify.Confidence(track), sheet.ArVerify.CRC(track), track + 1);
                            accurateTracks++;
                        }
                    }
                }
            }
            logWriter.WriteLine();
            if (knownTracks == 0)
                logWriter.WriteLine("None of the tracks are present in the AccurateRip database");
            else if (accurateTracks == 0)
            {
                logWriter.WriteLine("No tracks could be verified as accurate");
                logWriter.WriteLine("You may have a different pressing from the one(s) in the database");
            }
            else if (accurateTracks == sheet.TrackCount)
                logWriter.WriteLine("All tracks accurately ripped");
            else
            {
                logWriter.WriteLine("{0,2} track(s) accurately ripped", accurateTracks);
                if (sheet.TrackCount - knownTracks > 0)
                    logWriter.WriteLine("{0,2} track(s) not present in the AccurateRip database", sheet.TrackCount - knownTracks);
                logWriter.WriteLine();
                logWriter.WriteLine("Some tracks could not be verified as accurate");
            }
            logWriter.WriteLine();
            if (sheet.OutputStyle != CUEStyle.SingleFile && sheet.OutputStyle != CUEStyle.SingleFileWithCUE)
            {
                if (wereErrors)
                    logWriter.WriteLine("There were errors");
                else
                    logWriter.WriteLine("No errors occurred");
                logWriter.WriteLine();
            }
            logWriter.WriteLine("End of status report");
            logWriter.Close();

            return logWriter.ToString();
        }

        #endregion

        #region Ripper

        public static string GetRipperLog(CUESheet sheet)
        {
            StringWriter logWriter = new StringWriter(CultureInfo.InvariantCulture);
            logWriter.WriteLine("{0}", sheet.CDRipper.RipperVersion);
            logWriter.WriteLine("Extraction logfile from : {0}", DateTime.Now);
            logWriter.WriteLine("Used drive              : {0}", sheet.CDRipper.ARName);
            logWriter.WriteLine("Read offset correction  : {0}", sheet.CDRipper.DriveOffset);
            logWriter.WriteLine("Read command            : {0}", sheet.CDRipper.CurrentReadCommand);
            logWriter.WriteLine("Secure mode             : {0}", sheet.CDRipper.CorrectionQuality);
            logWriter.WriteLine("Disk length             : {0}", CDImageLayout.TimeToString(sheet.TOC.AudioLength));
            logWriter.WriteLine("AccurateRip             : {0}", sheet.ArVerify.ARStatus == null ? "ok" : sheet.ArVerify.ARStatus);
            if (sheet.HDCDDecoder != null && string.Format("{0:s}", sheet.HDCDDecoder) != "")
                logWriter.WriteLine("HDCD                    : {0:f}", sheet.HDCDDecoder);
            logWriter.WriteLine();
            logWriter.WriteLine("TOC of the extracted CD");
            logWriter.WriteLine();
            logWriter.Write(GetTOCContents(sheet));
            logWriter.WriteLine();
            logWriter.WriteLine("     Track |   Pregap  | Indexes");
            logWriter.WriteLine("    ---------------------------------------------------------");
            for (int track = 1; track <= sheet.TOC.TrackCount; track++)
                logWriter.WriteLine("{0,9}  | {1,8} |    {2,2}",
                    sheet.TOC[track].Number,
                    CDImageLayout.TimeToString(sheet.TOC[track].Pregap + (track == 1 ? 150U : 0U)),
                    sheet.TOC[track].LastIndex);
            logWriter.WriteLine();
            logWriter.WriteLine("Destination files");
            foreach (string path in sheet.DestPaths)
                logWriter.WriteLine("    {0}", path);
            bool wereErrors = sheet.PrintErrors(logWriter, sheet.TOC[sheet.TOC.FirstAudio][0].Start, sheet.TOC.AudioLength);
            if (wereErrors)
            {
                logWriter.WriteLine();
                if (wereErrors)
                    logWriter.WriteLine("There were errors");
                else
                    logWriter.WriteLine("No errors occurred");
            }
            if (sheet.IsUsingCUEToolsDB)
            {
                logWriter.WriteLine();
                sheet.GenerateCTDBLog(logWriter);
            }
            if (sheet.IsUsingAccurateRip)
            {
                logWriter.WriteLine();
                logWriter.WriteLine("AccurateRip summary");
                logWriter.WriteLine();
                sheet.ArVerify.GenerateFullLog(logWriter, true, AccurateRipVerify.CalculateAccurateRipId(sheet.TOC));
            }
            logWriter.WriteLine();
            logWriter.WriteLine("End of status report");
            logWriter.Close();

            return logWriter.ToString();
        }

        #endregion
    }
}
