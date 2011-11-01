using System;
using CUETools.Codecs;
using CUETools.DSP.Resampler.Internal;

namespace CUETools.DSP.Resampler
{
	public class SOXResampler
	{
		private AudioPCMConfig inputPCM;
        private AudioPCMConfig outputPCM;
        private rate_t[] rate;
        private rate_t[] rateUp2;
        private rate_shared_t shared;
        private rate_shared_t sharedUp2;

		public SOXResampler(AudioPCMConfig inputPCM, AudioPCMConfig outputPCM, SOXResamplerConfig config)
		{
			this.inputPCM = inputPCM;
			this.outputPCM = outputPCM;
			
			if (inputPCM.ChannelCount != outputPCM.ChannelCount)
				throw new NotSupportedException();

			if (outputPCM.SampleRate == inputPCM.SampleRate * 4 && config.Quality >= SOXResamplerQuality.Medium)
			{
				this.rate = new rate_t[inputPCM.ChannelCount];
				this.shared = new rate_shared_t();
				this.rateUp2 = new rate_t[inputPCM.ChannelCount];
				this.sharedUp2 = new rate_shared_t();

				for (int i = 0; i < inputPCM.ChannelCount; i++)
				{
					rateUp2[i] = new rate_t(inputPCM.SampleRate, inputPCM.SampleRate * 2, sharedUp2, 0.5,
						config.Quality, -1, config.Phase, config.Bandwidth, config.AllowAliasing);
					rate[i] = new rate_t(inputPCM.SampleRate * 2, inputPCM.SampleRate * 4, shared, 0.5,
						config.Quality, -1, 50, 90, true);
				}
			}
			else
			{
				this.rate = new rate_t[inputPCM.ChannelCount];
				this.shared = new rate_shared_t();

				for (int i = 0; i < inputPCM.ChannelCount; i++)
				{
					rate[i] = new rate_t(inputPCM.SampleRate, outputPCM.SampleRate, shared, (double)inputPCM.SampleRate / outputPCM.SampleRate,
						config.Quality, -1, config.Phase, config.Bandwidth, config.AllowAliasing);
				}
			}
		}

		public void Flow(AudioBuffer input, AudioBuffer output)
		{
			if (input.PCM.SampleRate != inputPCM.SampleRate || output.PCM.SampleRate != outputPCM.SampleRate ||
				input.PCM.ChannelCount != inputPCM.ChannelCount || output.PCM.ChannelCount != outputPCM.ChannelCount)
				throw new NotSupportedException();
			if (rateUp2 == null)
			{
				output.Prepare(-1);
				int odone = output.Size;
				for (int channel = 0; channel < inputPCM.ChannelCount; channel++)
				{
					rate[channel].input(input.Float, channel, input.Length);
					rate[channel].process();
					rate[channel].output(output.Float, channel, ref odone);
				}
				output.Length = odone;
			}
			else
				throw new NotSupportedException();
		}

		public AudioPCMConfig InputPCM
		{
			get
			{
				return inputPCM;
			}
		}

		public AudioPCMConfig OutputPCM
		{
			get
			{
				return outputPCM;
			}
		}
	}
}
