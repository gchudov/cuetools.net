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
		public int seed, offset, start, end, errors, maxStrideErrors;

		public TestImageGenerator(CDImageLayout toc, int seed, int offset, int errors, int maxStrideErrors, int start, int end)
		{
			this.toc = toc;
			this.seed = seed;
			this.offset = offset;
			this.start = start;
			this.end = end;
			this.errors = errors;
            this.maxStrideErrors = maxStrideErrors;
		}

        public TestImageGenerator(string trackoffsets, int seed, int offset, int errors, int maxStrideErrors, int start, int end)
            : this(new CDImageLayout(trackoffsets), seed, offset, errors, maxStrideErrors, start, end)
		{
		}

		public TestImageGenerator(CDImageLayout toc, int seed, int offset, int errors = 0, int maxStrideErrors = 0)
            : this(toc, seed, offset, errors, maxStrideErrors, 0, (int)toc.AudioLength * 588)
		{
		}

        public TestImageGenerator(string trackoffsets, int seed, int offset = 0, int errors = 0, int maxStrideErrors = 0)
			: this(new CDImageLayout(trackoffsets), seed, offset, errors, maxStrideErrors)
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
            this.maxStrideErrors = copy.maxStrideErrors;
		}

		public void Write(IAudioDest dest)
		{
			if (start < 0 || start > end || end > toc.AudioLength * 588)
				throw new ArgumentOutOfRangeException();
			var src = new NoiseAndErrorsGenerator(AudioPCMConfig.RedBook, end - start, seed, offset + start, errors, maxStrideErrors);
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

		public CDRepairEncode CreateCDRepairEncode(int stride)
		{
			var ar = new AccurateRipVerify(toc, null);
			var encode = new CDRepairEncode(ar, stride);
			ar.Position = start;
			Write(ar);
			//ar.Close();
			return encode;
		}

		public static AccurateRipVerify CreateAccurateRipVerify(string trackoffsets, int seed, int offset, int start, int end)
		{
			var generator = new TestImageGenerator(trackoffsets, seed, offset, 0, 0, start, end);
			return generator.CreateAccurateRipVerify();
		}

		public static AccurateRipVerify CreateAccurateRipVerify(string trackoffsets, int seed, int offset)
		{
			var generator = new TestImageGenerator(trackoffsets, seed, offset, 0, 0);
			return generator.CreateAccurateRipVerify();
		}
	}
}
