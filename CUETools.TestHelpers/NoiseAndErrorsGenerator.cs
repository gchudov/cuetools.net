using System;
using System.Collections.Generic;
using System.Text;
using CUETools.Codecs;

namespace CUETools.TestHelpers
{
	public class NoiseAndErrorsGenerator : IAudioSource
	{
		private long _sampleOffset, _sampleCount;
		private AudioPCMConfig pcm;
		private Random rnd, rnd2;
		private byte[] temp;
		private int[] errors;
		private int tempOffs;
		private int nextError;

        public NoiseAndErrorsGenerator(AudioPCMConfig pcm, long sampleCount, int seed, int offset, int errors, int maxStrideErrors = 0)
		{
			if (offset < 0)
				throw new ArgumentOutOfRangeException("offset", "offset cannot be negative");
			if (errors < 0)
				throw new ArgumentOutOfRangeException("offset", "errors cannot be negative");

			this._sampleOffset = 0;
			this._sampleCount = sampleCount;
			this.pcm = pcm;
			this.rnd = new Random(seed);
			this.temp = new byte[8192 * pcm.BlockAlign];
			this.tempOffs = temp.Length;
			int byteOff = offset * pcm.BlockAlign;
			for (int k = 0; k < byteOff / temp.Length; k++)
				rnd.NextBytes(temp);
			if (byteOff % temp.Length > 0)
				rnd.NextBytes(new byte[byteOff % temp.Length]);
			this.errors = new int[errors];
			this.rnd2 = new Random(seed);
            var strideErrors = new int[10 * 588];
            for (int i = 0; i < errors; i++)
            {
                do
                {
                    this.errors[i] = this.rnd2.Next(0, (int)sampleCount);
                } while (maxStrideErrors > 0 && strideErrors[this.errors[i] % (10 * 588)] >= maxStrideErrors);
                strideErrors[this.errors[i] % (10 * 588)]++;
            }
			this.rnd2 = new Random(seed);
			Array.Sort(this.errors);
			this.nextError = 0;
		}

        public IAudioDecoderSettings Settings => null;

		public NoiseAndErrorsGenerator(long sampleCount)
			: this(AudioPCMConfig.RedBook, sampleCount, 0, 0, 0)
		{
		}

		public long Length
		{
			get
			{
				return _sampleCount;
			}
		}

		public long Remaining
		{
			get
			{
				return _sampleCount - _sampleOffset;
			}
		}

		public long Position
		{
			get
			{
				return _sampleOffset;
			}
			set
			{
				_sampleOffset = value;
			}
		}

		public AudioPCMConfig PCM { get { return pcm; } }

		public int Read(AudioBuffer buff, int maxLength)
		{
			buff.Prepare(this, maxLength);

			int buffOffs = 0;
			while (buffOffs < buff.ByteLength)
			{
				if (tempOffs == temp.Length)
				{
					rnd.NextBytes(temp);
					tempOffs = 0;
				}
				int chunk = Math.Min(buff.ByteLength - buffOffs, temp.Length - tempOffs);
				Array.Copy(temp, tempOffs, buff.Bytes, buffOffs, chunk);
				buffOffs += chunk;
				tempOffs += chunk;
			}

			while (this.nextError < this.errors.Length && this.errors[this.nextError] < _sampleOffset + buff.Length)
			{
				for (int i = 0; i < PCM.BlockAlign; i++)
					buff.Bytes[(this.errors[this.nextError] - _sampleOffset) * PCM.BlockAlign + i] ^= (byte)this.rnd2.Next(1, 255);
				this.nextError++;
			}

			_sampleOffset += buff.Length;
			return buff.Length;
		}

		public void Close()
		{
		}

		public string Path { get { return null; } }
	}
}
