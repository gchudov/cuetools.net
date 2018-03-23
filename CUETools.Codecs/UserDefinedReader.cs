using System;
using System.Diagnostics;
using System.IO;

namespace CUETools.Codecs
{
    public class CommandLineDecoder : IAudioSource
    {
        string _path;
        Process _decoderProcess;
        WAV.AudioDecoder rdr;

        private CommandLineDecoderSettings m_settings;
        public AudioDecoderSettings Settings => m_settings;

        public long Position
        {
            get
            {
                Initialize();
                return rdr.Position;
            }
            set
            {
                Initialize();
                rdr.Position = value;
            }
        }

        public long Length
        {
            get
            {
                Initialize();
                return rdr.Length;
            }
        }

        public long Remaining
        {
            get
            {
                Initialize();
                return rdr.Remaining;
            }
        }

        public AudioPCMConfig PCM
        {
            get
            {
                Initialize();
                return rdr.PCM;
            }
        }

        public string Path { get { return _path; } }

        public CommandLineDecoder(CommandLineDecoderSettings settings, string path, Stream IO)
        {
            m_settings = settings;
            _path = path;
            _decoderProcess = null;
            rdr = null;
        }

        void Initialize()
        {
            if (_decoderProcess != null)
                return;
            _decoderProcess = new Process();
            _decoderProcess.StartInfo.FileName = m_settings.Path;
            _decoderProcess.StartInfo.Arguments = m_settings.Parameters.Replace("%I", "\"" + _path + "\"");
            _decoderProcess.StartInfo.CreateNoWindow = true;
            _decoderProcess.StartInfo.RedirectStandardOutput = true;
            _decoderProcess.StartInfo.UseShellExecute = false;
            bool started = false;
            Exception ex = null;
            try
            {
                started = _decoderProcess.Start();
                if (started)
                    _decoderProcess.PriorityClass = Process.GetCurrentProcess().PriorityClass;
            }
            catch (Exception _ex)
            {
                ex = _ex;
            }
            if (!started)
            {
                _decoderProcess = null;
                throw new Exception(m_settings.Path + ": " + (ex == null ? "please check the path" : ex.Message));
            }
            rdr = new WAV.AudioDecoder(new WAV.DecoderSettings(), _path, _decoderProcess.StandardOutput.BaseStream);
        }

        public void Close()
        {
            if (rdr != null)
                rdr.Close();
            if (_decoderProcess != null && !_decoderProcess.HasExited)
                try { _decoderProcess.Kill(); _decoderProcess.WaitForExit(); }
                catch { }
        }

        public int Read(AudioBuffer buff, int maxLength)
        {
            Initialize();
            return rdr.Read(buff, maxLength);
        }
    }
}
