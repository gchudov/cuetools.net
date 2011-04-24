using System;
using System.Collections.Generic;
using System.Text;
using CUETools.AccurateRip;
using CUETools.Codecs;
using CUETools.CDImage;

namespace CUETools.TestHelpers
{
	public class TestImageGenerator
	{
		public CDImageLayout toc;
		public int seed, offset, start, end, errors;

		public TestImageGenerator(CDImageLayout toc, int seed, int offset, int errors, int start, int end)
		{
			this.toc = toc;
			this.seed = seed;
			this.offset = offset;
			this.start = start;
			this.end = end;
			this.errors = errors;
		}

		public TestImageGenerator(string trackoffsets, int seed, int offset, int errors, int start, int end)
			: this (new CDImageLayout(trackoffsets), seed, offset, errors, start, end)
		{
		}

		public TestImageGenerator(CDImageLayout toc, int seed, int offset, int errors)
			: this(toc, seed, offset, errors, 0, (int)toc.AudioLength * 588)
		{
		}

		public TestImageGenerator(string trackoffsets, int seed, int offset, int errors)
			: this(new CDImageLayout(trackoffsets), seed, offset, errors)
		{
		}

		public TestImageGenerator(TestImageGenerator copy)
		{
			this.toc = copy.toc;
			this.seed = copy.seed;
			this.offset = copy.offset;
			this.start = copy.start;
			this.end = copy.end;
			this.errors = copy.errors;
		}

		public void Write(IAudioDest dest)
		{
			if (start < 0 || start > end || end > toc.AudioLength * 588)
				throw new ArgumentOutOfRangeException();
			var src = new NoiseAndErrorsGenerator(AudioPCMConfig.RedBook, end - start, seed, offset + start, errors);
			var buff = new AudioBuffer(src, 588 * 10);
			var rnd = new Random(seed);
			//dest.Position = start;
			while (src.Remaining > 0)
			{
				src.Read(buff, rnd.Next(1, buff.Size));
				dest.Write(buff);
			}
		}

		public AccurateRipVerify CreateAccurateRipVerify()
		{
			var ar = new AccurateRipVerify(toc, null);
			ar.Position = start;
			Write(ar);
			return ar;
		}

		public CDRepairEncode CreateCDRepairEncode(int stride, int npar)
		{
			var ar = new AccurateRipVerify(toc, null);
			var encode = new CDRepairEncode(ar, stride, npar);
			ar.Position = start;
			Write(ar);
			//ar.Close();
			return encode;
		}

		public static AccurateRipVerify CreateAccurateRipVerify(string trackoffsets, int seed, int offset, int start, int end)
		{
			var generator = new TestImageGenerator(trackoffsets, seed, offset, 0, start, end);
			return generator.CreateAccurateRipVerify();
		}

		public static AccurateRipVerify CreateAccurateRipVerify(string trackoffsets, int seed, int offset)
		{
			var generator = new TestImageGenerator(trackoffsets, seed, offset, 0);
			return generator.CreateAccurateRipVerify();
		}
	}
}
