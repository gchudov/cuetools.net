using System;
using System.Collections.Generic;
using System.Text;
using CUETools.Codecs;
using NAudio.CoreAudioApi;
using System.Threading;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace CUETools.Codecs.CoreAudio
{
    /// <summary>
    /// Support for playback using Wasapi
    /// </summary>
    public class WasapiOut : IWavePlayer
    {
        AudioClient audioClient;
        AudioClientShareMode shareMode;
        AudioRenderClient renderClient;
        int latencyMilliseconds;
        int bufferFrameCount;
        bool isUsingEventSync;
        EventWaitHandle frameEventWaitHandle;
		AudioBuffer[] readBuffers;
        volatile PlaybackState playbackState;
        Thread playThread;
		private long _sampleOffset;
		private AudioPCMConfig pcm;
		private NAudio.Wave.WaveFormat outputFormat;
        
        /// <summary>
        /// Playback Stopped
        /// </summary>
        public event EventHandler PlaybackStopped;

        /// <summary>
        /// WASAPI Out using default audio endpoint
        /// </summary>
        /// <param name="shareMode">ShareMode - shared or exclusive</param>
        /// <param name="latency">Desired latency in milliseconds</param>
        public WasapiOut(AudioClientShareMode shareMode, int latency) :
            this(GetDefaultAudioEndpoint(), shareMode, true, latency, AudioPCMConfig.RedBook)
        {

        }

        /// <summary>
        /// WASAPI Out using default audio endpoint
        /// </summary>
        /// <param name="shareMode">ShareMode - shared or exclusive</param>
        /// <param name="useEventSync">true if sync is done with event. false use sleep.</param>
        /// <param name="latency">Desired latency in milliseconds</param>
        public WasapiOut(AudioClientShareMode shareMode, bool useEventSync, int latency) :
			this(GetDefaultAudioEndpoint(), shareMode, useEventSync, latency, AudioPCMConfig.RedBook)
        {

        }

        /// <summary>
        /// Creates a new WASAPI Output
        /// </summary>
        /// <param name="device">Device to use</param>
        /// <param name="shareMode"></param>
        /// <param name="useEventSync">true if sync is done with event. false use sleep.</param>
        /// <param name="latency"></param>
		public WasapiOut(MMDevice device, AudioClientShareMode shareMode, bool useEventSync, int latency, AudioPCMConfig pcm)
        {
            this.audioClient = device.AudioClient;
            this.shareMode = shareMode;
            this.isUsingEventSync = useEventSync;
            this.latencyMilliseconds = latency;
			this.pcm = pcm;
			this.outputFormat = new NAudio.Wave.WaveFormat(pcm.SampleRate, pcm.BitsPerSample, pcm.ChannelCount);
			NAudio.Wave.WaveFormatExtensible closestSampleRateFormat;
            if (!audioClient.IsFormatSupported(shareMode, outputFormat, out closestSampleRateFormat))
				throw new NotSupportedException("PCM format mismatch");
			Init();
			bufferFrameCount = audioClient.BufferSize;
			readBuffers = new AudioBuffer[2];
			readBuffers[0] = new AudioBuffer(pcm, bufferFrameCount);
			readBuffers[1] = new AudioBuffer(pcm, bufferFrameCount);
			//if (this.shareMode == AudioClientShareMode.Exclusive)
			//    this.latencyMilliseconds = (int)(this.audioClient.DefaultDevicePeriod / 10000);
        }


        public static MMDevice GetDefaultAudioEndpoint()
        {
            if (Environment.OSVersion.Version.Major < 6)
            {
                throw new NotSupportedException("WASAPI supported only on Windows Vista and above");
            }
            MMDeviceEnumerator enumerator = new MMDeviceEnumerator();
            return enumerator.GetDefaultAudioEndpoint(DataFlow.Render, Role.Console);
        }

		private bool BufferReady(int iBuf, bool write)
		{
			return write ? readBuffers[iBuf].Length == 0 : readBuffers[iBuf].Length == readBuffers[iBuf].Size;
		}

		public AudioBuffer GetBuffer(bool write)
		{
			lock (this)
			{
				while (!BufferReady(0, write) && !BufferReady(1, write) && playbackState != PlaybackState.Stopped)
					Monitor.Wait(this);
				if (playbackState == PlaybackState.Stopped)
					return null;
				if (BufferReady(0, write))
					return readBuffers[0];
				return readBuffers[1];
			}
		}

		public void ReleaseBuffer(AudioBuffer buff, bool write, int length)
		{
			lock (this)
			{
				buff.Length = length;
				Monitor.Pulse(this);
			}
		}

        private void PlayThread()
        {
			try
			{
				AudioBuffer buff = GetBuffer(false);
				if (buff == null)
				{
					RaisePlaybackStopped();
					return;
				}

				audioClient.Reset();

				// fill a whole buffer	
				IntPtr buffer = renderClient.GetBuffer(buff.Length);
				Marshal.Copy(buff.Bytes, 0, buffer, buff.ByteLength);
				renderClient.ReleaseBuffer(buff.Length, AudioClientBufferFlags.None);
				ReleaseBuffer(buff, false, 0);

				// Create WaitHandle for sync
				WaitHandle[] waitHandles = new WaitHandle[] { frameEventWaitHandle };
				if (frameEventWaitHandle != null)
					frameEventWaitHandle.Reset();
				audioClient.Start();

				if (isUsingEventSync && shareMode == AudioClientShareMode.Exclusive)
				{
					while (playbackState != PlaybackState.Stopped)
					{
						int indexHandle = WaitHandle.WaitAny(waitHandles, 10 * latencyMilliseconds, false);
						if (playbackState == PlaybackState.Playing && indexHandle != WaitHandle.WaitTimeout)
						{
							// In exclusive mode, always ask the max = bufferFrameCount = audioClient.BufferSize
							buff = GetBuffer(false);
							if (buff == null)
								break;
							buffer = renderClient.GetBuffer(buff.Length);
							Marshal.Copy(buff.Bytes, 0, buffer, buff.ByteLength);
							renderClient.ReleaseBuffer(buff.Length, AudioClientBufferFlags.None);
							ReleaseBuffer(buff, false, 0);
						}
					}
				}
				else
				{
					buff = null;
					int offs = 0;
					while (playbackState != PlaybackState.Stopped)
					{
						// If using Event Sync, Wait for notification from AudioClient or Sleep half latency
						int indexHandle = 0;
						if (isUsingEventSync)
						{
							indexHandle = WaitHandle.WaitAny(waitHandles, 3 * latencyMilliseconds, false);
						}
						else
						{
							Thread.Sleep(latencyMilliseconds / 2);
						}

						// If still playing and notification is ok
						if (playbackState == PlaybackState.Playing && indexHandle != WaitHandle.WaitTimeout)
						{
							// See how much buffer space is available.
							int numFramesAvailable = bufferFrameCount - audioClient.CurrentPadding;
							if (numFramesAvailable > 0)
							{
								if (buff == null)
								{
									buff = GetBuffer(false);
									offs = 0;
								}
								if (buff == null)
									break;
								numFramesAvailable = Math.Min(numFramesAvailable, buff.Length - offs);
								buffer = renderClient.GetBuffer(numFramesAvailable);
								Marshal.Copy(buff.Bytes, offs * pcm.BlockAlign, buffer, numFramesAvailable * pcm.BlockAlign);
								renderClient.ReleaseBuffer(numFramesAvailable, AudioClientBufferFlags.None);
								offs += numFramesAvailable;
								if (offs == buff.Length)
								{
									ReleaseBuffer(buff, false, 0);
									buff = null;
								}
							}
						}
					}
				}
				//Thread.Sleep(isUsingEventSync ? latencyMilliseconds : latencyMilliseconds / 2);
				audioClient.Stop();
				if (playbackState == PlaybackState.Stopped)
					audioClient.Reset();
			}
			catch (Exception ex)
			{
				playbackState = PlaybackState.Stopped;
				ReleaseBuffer(readBuffers[0], false, 0);
				ReleaseBuffer(readBuffers[1], false, 0);
				playThread = null;
				try
				{
					audioClient.Stop();
				}
				catch { }
				RaisePlaybackException(ex);
				return;
			}
			ReleaseBuffer(readBuffers[0], false, 0);
			ReleaseBuffer(readBuffers[1], false, 0);
			RaisePlaybackStopped();
        }

		private void RaisePlaybackException(Exception ex)
		{
			RaisePlaybackStopped();
		}

        private void RaisePlaybackStopped()
        {
            if (PlaybackStopped != null)
            {
                PlaybackStopped(this, EventArgs.Empty);
            }
        }

        #region IWavePlayer Members

        /// <summary>
        /// Begin Playback
        /// </summary>
        public void Play()
        {
			switch (playbackState)
			{
				case PlaybackState.Playing:
					return;
				case PlaybackState.Paused:
					playbackState = PlaybackState.Playing;
					return;
				case PlaybackState.Stopped:
					playbackState = PlaybackState.Playing;
					playThread = new Thread(new ThreadStart(PlayThread));
					playThread.Priority = ThreadPriority.Highest;
					playThread.IsBackground = true;
					playThread.Name = "Pro Audio";
					playThread.Start();
					return;
            }
        }

        /// <summary>
        /// Stop playback and flush buffers
        /// </summary>
        public void Stop()
        {
            if (playbackState != PlaybackState.Stopped)
            {
                playbackState = PlaybackState.Stopped;
				if (frameEventWaitHandle != null)
					frameEventWaitHandle.Set();
				playThread.Join();
                playThread = null;
				ReleaseBuffer(readBuffers[0], false, 0);
				ReleaseBuffer(readBuffers[1], false, 0);
				active = null;
				active_offset = 0;
			}
        }

        /// <summary>
        /// Stop playback without flushing buffers
        /// </summary>
        public void Pause()
        {
            if (playbackState == PlaybackState.Playing)
            {
                //playbackState = PlaybackState.Paused;
            }
			if (frameEventWaitHandle != null)
				frameEventWaitHandle.Set();
        }

		private bool inited = false;


        /// <summary>
        /// Initialize for playing the specified format
        /// </summary>
        private void Init()
        {
			if (inited)
				return;

            long latencyRefTimes = latencyMilliseconds * 10000;
            // first attempt uses the WaveFormat from the WaveStream

            // If using EventSync, setup is specific with shareMode
            if (isUsingEventSync)
            {
                // Init Shared or Exclusive
                if (shareMode == AudioClientShareMode.Shared)
                {
                    // With EventCallBack and Shared, both latencies must be set to 0
                    audioClient.Initialize(shareMode, AudioClientStreamFlags.EventCallback, 0, 0,
                        outputFormat, Guid.Empty);

                    // Get back the effective latency from AudioClient
                    latencyMilliseconds = (int)(audioClient.StreamLatency / 10000);
                }
                else
                {
                    // With EventCallBack and Exclusive, both latencies must equals
                    audioClient.Initialize(shareMode, AudioClientStreamFlags.EventCallback, latencyRefTimes, latencyRefTimes,
                                        outputFormat, Guid.Empty);
                }

                // Create the Wait Event Handle
                frameEventWaitHandle = new EventWaitHandle(false, EventResetMode.AutoReset);
                audioClient.SetEventHandle(frameEventWaitHandle);
            }
            else
            {
                // Normal setup for both sharedMode
                audioClient.Initialize(shareMode, AudioClientStreamFlags.None, latencyRefTimes, 0,
                                    outputFormat, Guid.Empty);
            }

            // Get the RenderClient
            renderClient = audioClient.AudioRenderClient;
			inited = true;
        }

        /// <summary>
        /// Playback State
        /// </summary>
        public PlaybackState PlaybackState
        {
            get { return playbackState; }
        }

        /// <summary>
        /// Volume
        /// </summary>
        public float Volume
        {
            get
            {
                return 1.0f;
            }
            set
            {
                if (value != 1.0f)
                {
                    throw new NotImplementedException();
                }
            }
        }

		public void Close()
		{
            if (audioClient != null)
                Stop();
		}

		public void Delete()
		{
			Close();
		}


		private AudioBuffer active = null;
		private int active_offset = 0;

		public void Write(AudioBuffer src)
		{
			if (src.Length == 0)
			{
				Stop();
				return;
			}
			int src_offs = 0;
			do
			{
				if (active == null)
					active = GetBuffer(true);
				if (active == null)
					throw new Exception("done");
				int toCopy = Math.Min(active.Size - active_offset, src.Length - src_offs);
				Array.Copy(src.Bytes, src_offs * pcm.BlockAlign, active.Bytes, active_offset * pcm.BlockAlign, toCopy * pcm.BlockAlign);
				src_offs += toCopy;
				active_offset += toCopy;
				if (active_offset == active.Size)
				{
					ReleaseBuffer(active, true, active.Size);
					active = null;
					active_offset = 0;
				}
			}
			while (src_offs < src.Length);
		}

        #endregion

		#region IAudioDest Members

		public long Position
		{
			get
			{
				return _sampleOffset;
			}
		}

		public long BlockSize
		{
			set { }
		}

		public long FinalSampleCount
		{
			set { ; }
		}

		public int CompressionLevel
		{
			get { return 0; }
			set { }
		}

		public string Options
		{
			set
			{
				if (value == null || value == "") return;
				throw new Exception("Unsupported options " + value);
			}
		}

		public AudioPCMConfig PCM
		{
			get { return pcm; }
		}

		public string Path { get { return null; } }

		#endregion

		#region IDisposable Members

		/// <summary>
        /// Dispose
        /// </summary>
        public void Dispose()
        {
            if (audioClient != null)
            {
                Stop();

                audioClient.Dispose();
                audioClient = null;
                renderClient = null;
            }

        }

        #endregion
    }
}
