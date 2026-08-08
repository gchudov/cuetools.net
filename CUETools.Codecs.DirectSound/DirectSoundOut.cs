using System;
using System.Windows.Forms;
using CUETools.Codecs.CoreAudio;
using NAudio.CoreAudioApi;

namespace CUETools.Codecs.DirectSound
{
    public class DirectSoundOut : CUETools.Codecs.IWavePlayer
    {
        private readonly WasapiOut player;

        public DirectSoundOut(Control owner, AudioPCMConfig pcm, int delay)
        {
            player = new WasapiOut(WasapiOut.GetDefaultAudioEndpoint(), AudioClientShareMode.Shared, true, delay, pcm);
        }

        public event EventHandler PlaybackStopped
        {
            add { player.PlaybackStopped += value; }
            remove { player.PlaybackStopped -= value; }
        }

        public float Volume
        {
            get { return player.Volume; }
            set { player.Volume = value; }
        }

        public CUETools.Codecs.PlaybackState PlaybackState => (CUETools.Codecs.PlaybackState)player.PlaybackState;

        public long Position => player.Position;

        public long FinalSampleCount
        {
            set { player.FinalSampleCount = value; }
        }

        public IAudioEncoderSettings Settings => player.Settings;

        public string Path => player.Path;

        public void Write(AudioBuffer src)
        {
            player.Write(src);
        }

        public void Play()
        {
            player.Play();
        }

        public void Stop()
        {
            player.Stop();
        }

        public void Pause()
        {
            player.Pause();
        }

        public void Close()
        {
            player.Close();
        }

        public void Delete()
        {
            player.Delete();
        }

        public void Dispose()
        {
            player.Dispose();
        }
    }
}
