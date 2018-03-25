using System;
using CUETools.Codecs;

namespace CUETools.Processor
{
    public class CUESheetAudio : IAudioSource
    {
        private CUESheet cueSheet;
        private IAudioSource currentAudio;
        private int currentSource;
        private long nextPos;
        private long _samplePos, _sampleLen;

        public IAudioDecoderSettings Settings => null;

        public long Length
        {
            get { return _sampleLen; }
        }

        public long Remaining
        {
            get { return _sampleLen - _samplePos; }
        }

        public AudioPCMConfig PCM
        {
            get { return AudioPCMConfig.RedBook; }
        }

        public string Path
        {
            get { return cueSheet.InputPath; }
        }

        public long Position
        {
            get
            {
                return _samplePos;
            }
            set
            {
                SetPosition(value);
            }
        }

        public CUESheetAudio(CUESheet cueSheet)
        {
            this.cueSheet = cueSheet;
            this.currentAudio = null;
            this._samplePos = 0;
            this._sampleLen = 0;
            this.currentSource = -1;
            this.nextPos = 0;
            cueSheet._sources.ForEach(s => this._sampleLen += s.Length);
        }

        private void SetPosition(long value)
        {
            if (value == _samplePos)
                return;
            long sourceStart = 0;
            for (int iSource = 0; iSource < cueSheet._sources.Count; iSource++)
            {
                if (value >= sourceStart && value < sourceStart + cueSheet._sources[iSource].Length)
                {
                    if (iSource != currentSource)
                    {
                        if (currentAudio != null)
                            currentAudio.Close();
                        currentSource = iSource;
                        currentAudio = cueSheet.GetAudioSource(currentSource, false);
                        nextPos = sourceStart + cueSheet._sources[currentSource].Length;
                    }
                    currentAudio.Position = value - sourceStart + cueSheet._sources[currentSource].Offset;
                    _samplePos = value;
                    return;
                }
                sourceStart += cueSheet._sources[iSource].Length;
            }
            throw new Exception("Invalid position");
        }

        public void Close()
        {
            if (currentAudio != null)
            {
                currentAudio.Close();
                currentAudio = null;
            }
        }

        public int Read(AudioBuffer buff, int maxLength)
        {
            buff.Prepare(this, maxLength);
            while (_samplePos >= nextPos)
            {
                currentSource++;
                if (currentSource >= cueSheet._sources.Count)
                {
                    buff.Length = 0;
                    return 0;
                }
                if (currentAudio != null)
                    currentAudio.Close();
                currentAudio = cueSheet.GetAudioSource(currentSource, false);
                int offset = (int)(_samplePos - nextPos);
                if (offset != 0)
                    currentAudio.Position += offset;
                nextPos += cueSheet._sources[currentSource].Length;
            }
            int count = (int)(nextPos - _samplePos);
            if (maxLength >= 0)
                count = Math.Min(count, maxLength);
            count = currentAudio.Read(buff, count);
            _samplePos += count;
            return count;
        }
    }
}
