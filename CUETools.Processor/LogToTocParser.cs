using System;
using System.IO;
using CUETools.CDImage;

namespace CUETools.Processor
{
    class LogToTocParser
    {
        public static CDImageLayout LogToToc(CDImageLayout toc, string eacLog)
        {
            CDImageLayout tocFromLog = new CDImageLayout();
            using (StringReader sr = new StringReader(eacLog))
            {
                bool isEACLog = false;
                bool iscdda2wavlog = false;
                string lineStr;
                int prevTrNo = 1, prevTrStart = 0;
                uint firstPreGap = 0;
                while ((lineStr = sr.ReadLine()) != null)
                {
                    if (isEACLog)
                    {
                        string[] n = lineStr.Split('|');
                        uint trNo, trStart, trEnd;
                        if (n.Length == 5 && uint.TryParse(n[0], out trNo) && uint.TryParse(n[3], out trStart) && uint.TryParse(n[4], out trEnd) && trNo == tocFromLog.TrackCount + 1)
                        {
                            bool isAudio = true;
                            if (tocFromLog.TrackCount >= toc.TrackCount &&
                                trStart == tocFromLog[tocFromLog.TrackCount].End + 1U + 152U * 75U
                                )
                                isAudio = false;
                            if (tocFromLog.TrackCount < toc.TrackCount &&
                                !toc[tocFromLog.TrackCount + 1].IsAudio
                                )
                                isAudio = false;
                            tocFromLog.AddTrack(new CDTrack(trNo, trStart, trEnd + 1 - trStart, isAudio, false));
                        }
                        else
                        {
                            string[] sepTrack = { "Track" };
                            string[] sepGap = { "Pre-gap length" };

                            string[] partsTrack = lineStr.Split(sepTrack, StringSplitOptions.None);
                            if (partsTrack.Length == 2 && uint.TryParse(partsTrack[1], out trNo))
                            {
                                prevTrNo = (int)trNo;
                                continue;
                            }

                            string[] partsGap = lineStr.Split(sepGap, StringSplitOptions.None);
                            if (partsGap.Length == 2)
                            {
                                string[] n1 = partsGap[1].Split(':', '.');
                                int h, m, s, f;
                                if (n1.Length == 4 && int.TryParse(n1[0], out h) && int.TryParse(n1[1], out m) && int.TryParse(n1[2], out s) && int.TryParse(n1[3], out f))
                                {
                                    uint gap = (uint)((f * 3 + 2) / 4 + 75 * (s + 60 * (m + 60 * h)));
                                    if (prevTrNo == 1)
                                        gap -= 150;
                                    if (prevTrNo == 1)
                                        firstPreGap = gap - toc[1].Start;
                                    //else
                                    //firstPreGap += gap;
                                    while (prevTrNo > tocFromLog.TrackCount && toc.TrackCount > tocFromLog.TrackCount)
                                    {
                                        tocFromLog.AddTrack(new CDTrack((uint)tocFromLog.TrackCount + 1,
                                            toc[tocFromLog.TrackCount + 1].Start + firstPreGap,
                                            toc[tocFromLog.TrackCount + 1].Length,
                                            toc[tocFromLog.TrackCount + 1].IsAudio, false));
                                    }
                                    if (prevTrNo <= tocFromLog.TrackCount)
                                        tocFromLog[prevTrNo].Pregap = gap;
                                }
                            }
                        }
                    }
                    else if (iscdda2wavlog)
                    {
                        foreach (string entry in lineStr.Split(','))
                        {
                            string[] n = entry.Split('(');
                            if (n.Length < 2) continue;
                            // assert n.Length == 2;
                            string key = n[0].Trim(' ', '.');
                            int trStart = int.Parse(n[1].Trim(' ', ')'));
                            bool isAudio = true; // !!!
                            if (key != "1")
                                tocFromLog.AddTrack(new CDTrack((uint)prevTrNo, (uint)prevTrStart, (uint)(trStart - prevTrStart), isAudio, false));
                            if (key == "lead-out")
                            {
                                iscdda2wavlog = false;
                                break;
                            }
                            prevTrNo = int.Parse(key);
                            prevTrStart = trStart;
                        }
                    }
                    else if (lineStr.StartsWith("TOC of the extracted CD")
                        || lineStr.StartsWith("Exact Audio Copy")
                        || lineStr.StartsWith("EAC extraction logfile")
                        || lineStr.StartsWith("CUERipper")
                        || lineStr.StartsWith("     Track |   Start  |  Length  | Start sector | End sector")
                        )
                        isEACLog = true;
                    else if (lineStr.StartsWith("Table of Contents: starting sectors"))
                        iscdda2wavlog = true;
                }
            }
            if (tocFromLog.TrackCount == 0)
                return null;
            tocFromLog[1][0].Start = 0;
            return tocFromLog;
        }
    }
}
