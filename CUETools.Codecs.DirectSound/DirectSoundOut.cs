using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Threading;
using System.Windows.Forms;
using Microsoft.DirectX;
using Microsoft.DirectX.DirectSound;

namespace CUETools.Codecs.DirectSound
{
	public class DirectSoundOut : IWavePlayer
	{
		private Device dSound;
		private WaveFormat format;// = new WaveFormat();
		private BufferDescription description = new BufferDescription();
		private SecondaryBuffer secondaryBuffer;
		private Notify notify;
		private MemoryStream pcmStream;
		private int SecBufByteSize;
        private AudioEncoderSettings m_settings;
		PlaybackState playbackState = PlaybackState.Stopped;
		
		/// <summary>
		/// Playback Stopped
		/// </summary>
		public event EventHandler PlaybackStopped;

		AutoResetEvent
			SecBufNotifyAtBegin = new AutoResetEvent(false),
			SecBufNotifyAtOneThird = new AutoResetEvent(false),
			SecBufNotifyAtTwoThirds = new AutoResetEvent(false);
		WaitHandle[] SecBufWaitHandles;

		public DirectSoundOut(Control owner, AudioPCMConfig pcm, int delay)
		{
            this.m_settings = new AudioEncoderSettings(pcm);

			//buffer = new CyclicBuffer(44100*4/10);
			//output = new CycilcBufferOutputStream(buffer);
			//input = new CycilcBufferInputStream(buffer);

			dSound = new Device();
			dSound.SetCooperativeLevel(owner, CooperativeLevel.Priority);
			format.AverageBytesPerSecond = pcm.SampleRate * pcm.BlockAlign;
			format.BitsPerSample = (short)pcm.BitsPerSample;
			format.BlockAlign = (short)pcm.BlockAlign;
			format.Channels = (short)pcm.ChannelCount;
			format.SamplesPerSecond = pcm.SampleRate;
			format.FormatTag = WaveFormatTag.Pcm;
			SecBufByteSize = delay * pcm.SampleRate * pcm.BlockAlign / 1000;
			description.Format = format;
			description.BufferBytes = SecBufByteSize;
			description.CanGetCurrentPosition = true;
			description.ControlPositionNotify = true;
			//description.ControlVolume = true;
			description.GlobalFocus = true;
			secondaryBuffer = new SecondaryBuffer(description, dSound);
			//secondaryBuffer.Volume = 100;

			notify = new Notify(secondaryBuffer);
			BufferPositionNotify[] bufferPositions = new BufferPositionNotify[3];
			bufferPositions[0].Offset = 0;
			bufferPositions[0].EventNotifyHandle = SecBufNotifyAtBegin.Handle;
			bufferPositions[1].Offset = SecBufByteSize / 3;
			bufferPositions[1].EventNotifyHandle = SecBufNotifyAtOneThird.Handle;
			bufferPositions[2].Offset = 2 * SecBufByteSize / 3;
			bufferPositions[2].EventNotifyHandle = SecBufNotifyAtTwoThirds.Handle;
			notify.SetNotificationPositions(bufferPositions);
			pcmStream = new MemoryStream(SecBufByteSize);

			SecBufWaitHandles = new WaitHandle[] { SecBufNotifyAtBegin, SecBufNotifyAtOneThird, SecBufNotifyAtTwoThirds };

			//wavoutput = new WAVWriter("", output, pcm);
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

		//int SecBufNextWritePosition = 0;
		//bool SecBufInitialLoad = false;

		bool playing = false;
		public void Write(AudioBuffer src)
		{
			//wavoutput.Write(src);

			pcmStream.SetLength(0);
			pcmStream.Write(src.Bytes, 0, src.ByteLength);
			pcmStream.Position = 0;
			//pcmStream.Position = 0;

			//while (true)
			//{
			//    if (SecBufInitialLoad)
			//    {
			//        int count = Math.Min(src.ByteLength, SecBufByteSize - SecBufNextWritePosition);
			//        if (count > 0)
			//        {
			//            secondaryBuffer.Write(SecBufNextWritePosition, pcmStream, count, LockFlag.None);
			//            SecBufNextWritePosition += count;
			//            pcmStream.Position += count;
			//        }

			//        if (SecBufByteSize == SecBufNextWritePosition)
			//        {
			//            // Finished filling the buffer
			//            SecBufInitialLoad = false;
			//            SecBufNextWritePosition = 0;

			//            // So start the playback in its own thread
			//            secondaryBuffer.Play(0, BufferPlayFlags.Looping);						

			//            // Yield rest of timeslice so playback can  
			//            // start right away.
			//            Thread.Sleep(0);
			//        }
			//        else
			//        {
			//            continue;  // Get more PCM data
			//        }
			//    }

			// Exhaust the current PCM data by writing the data into secondaryBuffer
			while (pcmStream.Position < pcmStream.Length)
			{
				int PlayPosition, WritePosition;

				secondaryBuffer.GetCurrentPosition(out PlayPosition, out WritePosition);

				int WriteCount = (int)Math.Min(
					(SecBufByteSize + PlayPosition - WritePosition) % SecBufByteSize,
					pcmStream.Length - pcmStream.Position);

				if (WriteCount > 0)
				{
					secondaryBuffer.Write(
						WritePosition,
						pcmStream,
						WriteCount,
						LockFlag.None);
					pcmStream.Position += WriteCount;
					if (!playing)
					{
						secondaryBuffer.Play(0, 0);
						playing = true;
					}
				}
				else
				{
					WaitHandle.WaitAny(SecBufWaitHandles, new TimeSpan(0, 0, 5), true);
				}
			}
		}

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
					//playThread = new Thread(new ThreadStart(PlayThread));
					//playThread.Priority = ThreadPriority.Highest;
					//playThread.IsBackground = true;
					//playThread.Name = "Pro Audio";
					//playThread.Start();
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
				//if (frameEventWaitHandle != null)
				//    frameEventWaitHandle.Set();
				//playThread.Join();
				//playThread = null;
				//ReleaseBuffer(readBuffers[0], false, 0);
				//ReleaseBuffer(readBuffers[1], false, 0);
				//active = null;
				//active_offset = 0;
			}
		}

		/// <summary>
		/// Stop playback without flushing buffers
		/// </summary>
		public void Pause()
		{
			if (playbackState == PlaybackState.Playing)
			{
				playbackState = PlaybackState.Paused;
			}
			//if (frameEventWaitHandle != null)
			//    frameEventWaitHandle.Set();
		}

		/// <summary>
		/// Playback State
		/// </summary>
		public PlaybackState PlaybackState
		{
			get { return playbackState; }
		}

		public void Close()
		{
			if (secondaryBuffer != null)
			{
				secondaryBuffer.Dispose();
				secondaryBuffer = null;
			}
			if (dSound != null)
			{
				dSound.Dispose();
				dSound = null;
			}
			//wavoutput.Close();
		}

		public void Delete()
		{
			Close();
		}

		#region IAudioDest Members

		public long Position
		{
			get
			{
				return 0;
			}
		}

		public long FinalSampleCount
		{
			set { ; }
		}

		public object Settings
		{
			get
			{
				return m_settings;
			}
		}

		public string Path { get { return null; } }

		#endregion

		#region IDisposable Members

		/// <summary>
		/// Dispose
		/// </summary>
		public void Dispose()
		{
			//if (audioClient != null)
			{
				Stop();
			}

		}

		#endregion
	}
}
