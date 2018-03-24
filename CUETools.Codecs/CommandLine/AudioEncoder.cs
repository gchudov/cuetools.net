using System;
using System.Diagnostics;
using System.IO;

namespace CUETools.Codecs.CommandLine
{
    public class AudioEncoder : IAudioDest
    {
        string _path;
        Process _encoderProcess;
        WAV.AudioEncoder wrt;
        CyclicBuffer outputBuffer = null;
        bool useTempFile = false;
        string tempFile = null;
        long _finalSampleCount = -1;
        bool closed = false;

        public long Position
        {
            get
            {
                return wrt.Position;
            }
        }

        public long FinalSampleCount
        {
            set { _finalSampleCount = wrt.FinalSampleCount = value; }
        }

        // !!!! Must not start the process in constructor, so that we can set CompressionLevel via Settings!
        private EncoderSettings m_settings;
        public AudioEncoderSettings Settings => m_settings;

        public string Path { get { return _path; } }

        public AudioEncoder(EncoderSettings settings, string path, Stream IO = null)
        {
            m_settings = settings;
            _path = path;
            useTempFile = m_settings.Parameters.Contains("%I");
            tempFile = path + ".tmp.wav";

            _encoderProcess = new Process();
            _encoderProcess.StartInfo.FileName = m_settings.Path;
            _encoderProcess.StartInfo.Arguments = m_settings.Parameters.Replace("%O", "\"" + path + "\"").Replace("%M", m_settings.EncoderMode).Replace("%P", m_settings.Padding.ToString()).Replace("%I", "\"" + tempFile + "\"");
            _encoderProcess.StartInfo.CreateNoWindow = true;
            if (!useTempFile)
                _encoderProcess.StartInfo.RedirectStandardInput = true;
            _encoderProcess.StartInfo.UseShellExecute = false;
            if (!m_settings.Parameters.Contains("%O"))
                _encoderProcess.StartInfo.RedirectStandardOutput = true;
            if (useTempFile)
            {
                wrt = new WAV.AudioEncoder(new WAV.EncoderSettings(settings.PCM), tempFile);
                return;
            }
            bool started = false;
            Exception ex = null;
            try
            {
                started = _encoderProcess.Start();
                if (started)
                    _encoderProcess.PriorityClass = Process.GetCurrentProcess().PriorityClass;
            }
            catch (Exception _ex)
            {
                ex = _ex;
            }
            if (!started)
                throw new Exception(m_settings.Path + ": " + (ex == null ? "please check the path" : ex.Message));
            if (_encoderProcess.StartInfo.RedirectStandardOutput)
            {
                Stream outputStream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
                outputBuffer = new CyclicBuffer(2 * 1024 * 1024, _encoderProcess.StandardOutput.BaseStream, outputStream);
            }
            Stream inputStream = new CyclicBufferOutputStream(_encoderProcess.StandardInput.BaseStream, 128 * 1024);
            wrt = new WAV.AudioEncoder(new WAV.EncoderSettings(settings.PCM), path, inputStream);
        }

        public void Close()
        {
            if (closed)
                return;
            closed = true;
            wrt.Close();
            if (useTempFile && (_finalSampleCount < 0 || wrt.Position == _finalSampleCount))
            {
                bool started = false;
                Exception ex = null;
                try
                {
                    started = _encoderProcess.Start();
                    if (started)
                        _encoderProcess.PriorityClass = Process.GetCurrentProcess().PriorityClass;
                }
                catch (Exception _ex)
                {
                    ex = _ex;
                }
                if (!started)
                    throw new Exception(m_settings.Path + ": " + (ex == null ? "please check the path" : ex.Message));
            }
            wrt = null;
            if (!_encoderProcess.HasExited)
                _encoderProcess.WaitForExit();
            if (useTempFile)
                File.Delete(tempFile);
            if (outputBuffer != null)
                outputBuffer.Close();
            if (_encoderProcess.ExitCode != 0)
                throw new Exception(String.Format("{0} returned error code {1}", m_settings.Path, _encoderProcess.ExitCode));
        }

        public void Delete()
        {
            Close();
            File.Delete(_path);
        }

        public void Write(AudioBuffer buff)
        {
            try
            {
                wrt.Write(buff);
            }
            catch (IOException ex)
            {
                if (_encoderProcess.HasExited)
                    throw new IOException(string.Format("{0} has exited prematurely with code {1}", m_settings.Path, _encoderProcess.ExitCode), ex);
                else
                    throw ex;
            }
            //_sampleLen += sampleCount;
        }
    }
}
