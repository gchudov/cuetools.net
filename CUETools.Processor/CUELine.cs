using System;
using System.Collections.Generic;
using System.Text;

namespace CUETools.Processor
{
    public class CUELine
    {
        private List<String> _params;
        private List<bool> _quoted;

        public CUELine()
        {
            _params = new List<string>();
            _quoted = new List<bool>();
        }

        public CUELine(string line)
        {
            int start, end, lineLen;
            bool isQuoted;

            _params = new List<string>();
            _quoted = new List<bool>();

            start = 0;
            lineLen = line.Length;

            while (true)
            {
                while ((start < lineLen) && ((line[start] == ' ') || (line[start] == '\t')))
                {
                    start++;
                }
                if (start >= lineLen)
                {
                    break;
                }

                isQuoted = (line[start] == '"');
                if (isQuoted)
                {
                    start++;
                }

                end = line.IndexOf(isQuoted ? '"' : ' ', start);
                if (end == -1)
                {
                    end = lineLen;
                }

                _params.Add(line.Substring(start, end - start));
                _quoted.Add(isQuoted);

                start = isQuoted ? end + 1 : end;
            }
        }

        public List<string> Params
        {
            get
            {
                return _params;
            }
        }

        public List<bool> IsQuoted
        {
            get
            {
                return _quoted;
            }
        }

        public override string ToString()
        {
            if (_params.Count != _quoted.Count)
            {
                throw new Exception("Parameter and IsQuoted lists must match.");
            }

            StringBuilder sb = new StringBuilder();
            int last = _params.Count - 1;

            for (int i = 0; i <= last; i++)
            {
                if (_quoted[i] || _params[i].Contains(" ")) sb.Append('"');
                sb.Append(_params[i].Replace('"', '\''));
                if (_quoted[i] || _params[i].Contains(" ")) sb.Append('"');
                if (i < last) sb.Append(' ');
            }

            return sb.ToString();
        }
    }
}
