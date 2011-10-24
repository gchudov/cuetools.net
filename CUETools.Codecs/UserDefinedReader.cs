using System;
using System.Diagnostics;
using System.IO;

namespace CUETools.Codecs
{
    public class UserDefinedReader : IAudioSource
    {
        string _path, _decoder, _decoderParams;
        Process _decoderProcess;
        WAVReader rdr;

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

        public UserDefinedReader(string path, Stream IO, string decoder, string decoderParams)
        {
            _path = path;
            _decoder = decoder;
            _decoderParams = decoderParams;
            _decoderProcess = null;
            rdr = null;
        }

        void Initialize()
        {
            if (_decoderProcess != null)
                return;
            _decoderProcess = new Process();
            _decoderProcess.StartInfo.FileName = _decoder;
            _decoderProcess.StartInfo.Arguments = _decoderParams.Replace("%I", "\"" + _path + "\"");
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
                throw new Exception(_decoder + ": " + (ex == null ? "please check the path" : ex.Message));
            rdr = new WAVReader(_path, _decoderProcess.StandardOutput.BaseStream);
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
