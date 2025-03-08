using System;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using CUETools.CDImage;
using CUETools.Codecs;
#if NETSTANDARD2_0
using System.Runtime.InteropServices;
using CUETools.Interop;
#endif

namespace CUETools.Ripper
{
	public interface ICDRipper : IAudioSource, IDisposable
	{
		bool Open(char Drive);
        void EjectDisk();
        void DisableEjectDisc(bool bDisable);
		bool DetectGaps();
		bool GapsDetected { get; }
		CDImageLayout TOC { get; }
		string ARName { get; }
		string EACName { get; }
		int DriveOffset { get; set; }
		int DriveC2ErrorMode { get; set; }
		bool ForceBE { get; set; }
		bool ForceD8 { get; set; }
		string RipperVersion { get; }
		string CurrentReadCommand { get; }
		int CorrectionQuality { get; set; }
		BitArray FailedSectors { get; }
        byte[] RetryCount { get; }

		event EventHandler<ReadProgressArgs> ReadProgress;
	}

	public class CDDrivesList
	{
        public static char[] DrivesAvailable()
		{
#if NETSTANDARD2_0
			if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
			{
				// DriveInfo doesn't return CD drives on linux...
				// For now we'll use a quirky workaround!
				var driveList = new List<char>();
				for (int i = 0; i < 10; ++i)
				{
                    var path = $"{Linux.CDROM_DEVICE_PATH}{i}";

					if (!Linux.PathExists(path)) break;
					driveList.Add($"{i}"[0]);
				}

				return driveList.ToArray();
			}
#endif

            List<char> result = new List<char>();
            foreach (DriveInfo info in DriveInfo.GetDrives())
            {
                if (info.DriveType == DriveType.CDRom)
                    result.Add(info.Name[0]);
            }

            return result.ToArray();
		}
	}

	public sealed class ReadProgressArgs : EventArgs
	{
		public string Action;
		public int Position;
		public int Pass;
		public int PassStart, PassEnd;
		public int ErrorsCount;
		public DateTime PassTime;

		public ReadProgressArgs()
		{
		}

		public ReadProgressArgs(int position, int pass, int passStart, int passEnd, int errorsCount, DateTime passTime)
		{
			Position = position;
			Pass = pass;
			PassStart = passStart;
			PassEnd = passEnd;
			ErrorsCount = errorsCount;
			PassTime = passTime;
		}
	}

    public static class BitArrayUtils
    {
        public static int PopulationCount(this BitArray bits, int start, int len)
        {
            int cnt = 0;
            for (int i = start; i < start + len; i++)
                if (bits[i])
                    cnt++;
            return cnt;
        }

        public static int PopulationCount(this BitArray bits)
        {
            return bits.PopulationCount(0, bits.Count);
        }
    }
}

namespace System.Runtime.CompilerServices
{
    [AttributeUsageAttribute(AttributeTargets.Assembly | AttributeTargets.Class | AttributeTargets.Method)]
    internal sealed class ExtensionAttribute : Attribute
    {
        public ExtensionAttribute() { }
    }
}
