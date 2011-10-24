using System;
using CUETools.AccurateRip;
using CUETools.Codecs;

namespace CUETools.Processor
{
    internal class CUEToolsVerifyTask
    {
        private CUESheet cueSheet;
        public IAudioSource source { get; private set; }
        public int start { get; private set; }
        public int end { get; private set; }
        public AccurateRipVerify ar { get; private set; }
        public IAudioDest hdcd { get; private set; }

        public CUEToolsVerifyTask(CUESheet cueSheet, int start, int end)
            : this(cueSheet, start, end, cueSheet._useAccurateRip || cueSheet._useCUEToolsDB ? new AccurateRipVerify(cueSheet.TOC, null) : null, null)
        {
        }

        public CUEToolsVerifyTask(CUESheet cueSheet, int start, int end, AccurateRipVerify ar)
            : this(cueSheet, start, end, ar, null)
        {
            if (cueSheet.Config.detectHDCD && CUEProcessorPlugins.hdcd != null)
            {
                try { this.hdcd = Activator.CreateInstance(CUEProcessorPlugins.hdcd, 2, 44100, 20, false) as IAudioDest; }
                catch { this.hdcd = null; }
            }
        }

        private CUEToolsVerifyTask(CUESheet cueSheet, int start, int end, AccurateRipVerify ar, IAudioDest hdcd)
        {
            this.cueSheet = cueSheet;
            this.start = start;
            this.end = end;
            this.source = new CUESheetAudio(cueSheet);
            if (cueSheet.IsCD || cueSheet.Config.separateDecodingThread)
                this.source = new AudioPipe(this.source, 0x10000);
            this.source.Position = start;
            this.ar = cueSheet._useAccurateRip ? ar : null;
            this.hdcd = hdcd;
            if (this.ar != null)
                this.ar.Position = start;
        }

        public bool TryClose()
        {
            try { Close(); }
            catch { return false; }
            return true;
        }

        public void Close()
        {
            if (this.source != null)
            {
                this.source.Close();
                this.source = null;
            }
            if (this.ar != null)
            {
                //this.ar.Close(); can't! throws
                this.ar = null;
            }
        }

        public int Step(AudioBuffer sampleBuffer)
        {
            if (Remaining == 0)
                return 0;
            int copyCount = source.Read(sampleBuffer, Remaining);
            if (copyCount == 0)
                return 0;
            if (ar != null)
                ar.Write(sampleBuffer);
            if (hdcd != null)
            {
                hdcd.Write(sampleBuffer);
                if (cueSheet.Config.wait750FramesForHDCD && source.Position > start + 750 * 588 && string.Format("{0:s}", hdcd) == "")
                    hdcd = null;
            }
            return copyCount;
        }

        public void Combine(CUEToolsVerifyTask task)
        {
            if (ar != null)
                ar.Combine(task.ar, task.start, task.end);
        }

        public int Remaining { get { return end - (int)source.Position; } }
    }
}
