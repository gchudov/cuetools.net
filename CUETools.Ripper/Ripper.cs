using System;
using System.Collections;
using System.Collections.Generic;
using CUETools.CDImage;
using CUETools.Codecs;
using System.Text;

namespace CUETools.Ripper
{
	public interface ICDRipper : IAudioSource, IDisposable
	{
		bool Open(char Drive);
		CDImageLayout TOC { get; }
		string ARName { get; }
		string EACName { get; }
		int DriveOffset { get; set; }
		string RipperVersion { get; }
		string CurrentReadCommand { get; }
		int CorrectionQuality { get; set; }
		BitArray Errors { get; }

		event EventHandler<ReadProgressArgs> ReadProgress;
	}

	public sealed class ReadProgressArgs : EventArgs
	{
		public int Position;
		public int Pass;
		public int PassStart, PassEnd;
		public int ErrorsCount;
		public DateTime PassTime;

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
}
