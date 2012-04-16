using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.IO;
using System.Text;

namespace CUETools.Processor
{
    public static class General
    {
        public static string GetCUELine(List<CUELine> list, string command)
        {
            var line = General.FindCUELine(list, command);
            return (line == null || line.Params.Count < 2) ? String.Empty : line.Params[1];
        }

        public static string GetCUELine(List<CUELine> list, string command, string command2)
        {
            var line = General.FindCUELine(list, command, command2);
            return (line == null || line.Params.Count < 3) ? String.Empty : line.Params[2];
        }

        public static CUELine FindCUELine(List<CUELine> list, string command)
        {
            command = command.ToUpper();
            foreach (CUELine line in list)
            {
                if (line.Params[0].ToUpper() == command)
                {
                    return line;
                }
            }
            return null;
        }

        public static CUELine FindCUELine(List<CUELine> list, string command, string command2)
        {
            command = command.ToUpper();
            command2 = command2.ToUpper();
            foreach (CUELine line in list)
            {
                if (line.Params.Count > 1 && line.Params[0].ToUpper() == command && line.Params[1].ToUpper() == command2)
                {
                    return line;
                }
            }
            return null;
        }

        //public static CUELine FindCUELine(List<CUELine> list, string [] commands)
        //{
        //    foreach (CUELine line in list)
        //    {
        //        if (line.Params.Count < commands.Length)
        //            continue;
        //        for (int i = 0; i < commands.Length; i++)
        //        {
        //            if (line.Params[i].ToUpper() != commands[i].ToUpper())
        //                break;
        //            if (i == commands.Length - 1)
        //                return line;
        //        }
        //    }
        //    return null;
        //}

        public static void SetCUELine(List<CUELine> list, string command, string value, bool quoted)
        {
            if (value == "")
            {
                General.DelCUELine(list, command);
                return;
            }

            CUELine line = General.FindCUELine(list, command);
            if (line == null)
            {
                line = new CUELine();
                line.Params.Add(command); line.IsQuoted.Add(false);
                line.Params.Add(value); line.IsQuoted.Add(quoted);
                list.Add(line);
            }
            else
            {
                while (line.Params.Count > 1)
                {
                    line.Params.RemoveAt(1);
                    line.IsQuoted.RemoveAt(1);
                }
                line.Params.Add(value); line.IsQuoted.Add(quoted);
            }
        }

        public static void SetCUELine(List<CUELine> list, string command, string command2, string value, bool quoted)
        {
            if (value == "")
            {
                General.DelCUELine(list, command, command2);
                return;
            }

            CUELine line = General.FindCUELine(list, command, command2);
            if (line == null)
            {
                line = new CUELine();
                line.Params.Add(command); line.IsQuoted.Add(false);
                line.Params.Add(command2); line.IsQuoted.Add(false);
                line.Params.Add(value); line.IsQuoted.Add(quoted);
                list.Add(line);
            }
            else
            {
                while (line.Params.Count > 2)
                {
                    line.Params.RemoveAt(2);
                    line.IsQuoted.RemoveAt(2);
                }
                line.Params.Add(value); line.IsQuoted.Add(quoted);
            }
        }

        public static void DelCUELine(List<CUELine> list, string command, string command2)
        {
            CUELine line = General.FindCUELine(list, command, command2);
            if (line == null)
                return;
            list.Remove(line);
        }

        public static void DelCUELine(List<CUELine> list, string command)
        {
            CUELine line = General.FindCUELine(list, command);
            if (line == null)
                return;
            list.Remove(line);
        }

        class TitleFormatFunctionInfo
        {
            public string func;
            public List<int> positions;
            public List<bool> found;

            public TitleFormatFunctionInfo(string _func, int position)
            {
                func = _func;
                positions = new List<int>();
                found = new List<bool>();
                NextArg(position);
            }

            public void Found()
            {
                found[found.Count - 1] = true;
            }

            public void NextArg(int position)
            {
                positions.Add(position);
                found.Add(false);
            }

            public string GetArg(StringBuilder sb, int no)
            {
                return sb.ToString().Substring(positions[no],
                    ((no == positions.Count - 1) ? sb.Length : positions[no + 1]) - positions[no]);
            }

            public int GetIntArg(StringBuilder sb, int no)
            {
                int res;
                return int.TryParse(GetArg(sb, no), out res) ? res : 0;
            }

            void Returns(StringBuilder sb, string res)
            {
                sb.Length = positions[0];
                sb.Append(res);
            }

            public bool Finalise(StringBuilder sb)
            {
                switch (func)
                {
                    case "[":
                        if (positions.Count != 1)
                            return false;
                        if (!found[0])
                            sb.Length = positions[0];
                        return true;
                    case "if":
                        if (positions.Count != 3)
                            return false;
                        Returns(sb, GetArg(sb, found[0] ? 1 : 2));
                        return true;
                    case "if2":
                        if (positions.Count != 2)
                            return false;
                        Returns(sb, GetArg(sb, found[0] ? 0 : 1));
                        return true;
                    case "if3":
                        if (positions.Count < 1)
                            return false;
                        for (int argno = 0; argno < positions.Count; argno++)
                            if (found[argno] || argno == positions.Count - 1)
                            {
                                Returns(sb, GetArg(sb, argno));
                                return true;
                            }
                        return false;
                    case "ifgreater":
                        if (positions.Count != 4)
                            return false;
                        Returns(sb, GetArg(sb, (GetIntArg(sb, 0) > GetIntArg(sb, 1)) ? 2 : 3));
                        return true;
                    case "iflonger":
                        if (positions.Count != 4)
                            return false;
                        Returns(sb, GetArg(sb, (GetArg(sb, 0).Length > GetIntArg(sb, 1)) ? 2 : 3));
                        return true;
                    case "ifequal":
                        if (positions.Count != 4)
                            return false;
                        Returns(sb, GetArg(sb, (GetIntArg(sb, 0) == GetIntArg(sb, 1)) ? 2 : 3));
                        return true;
                    case "len":
                        if (positions.Count != 1)
                            return false;
                        Returns(sb, GetArg(sb, 0).Length.ToString());
                        return true;
                    case "max":
                        if (positions.Count != 2)
                            return false;
                        Returns(sb, Math.Max(GetIntArg(sb, 0), GetIntArg(sb, 1)).ToString());
                        return true;
                    case "directory":
                        if (positions.Count != 1 && positions.Count != 2 && positions.Count != 3)
                            return false;
                        try
                        {
                            int arg3 = positions.Count > 1 ? GetIntArg(sb, 1) : 1;
                            int arg2 = positions.Count > 2 ? GetIntArg(sb, 2) : arg3;
                            Returns(sb, General.GetDirectoryElements(Path.GetDirectoryName(GetArg(sb, 0)), -arg2, -arg3));
                        }
                        catch { return false; }
                        return true;
                    case "directory_path":
                        if (positions.Count != 1)
                            return false;
                        try { Returns(sb, Path.GetDirectoryName(GetArg(sb, 0))); }
                        catch { return false; }
                        return true;
                    case "ext":
                        if (positions.Count != 1)
                            return false;
                        try { Returns(sb, Path.GetExtension(GetArg(sb, 0))); }
                        catch { return false; }
                        return true;
                    case "filename":
                        if (positions.Count != 1)
                            return false;
                        try { Returns(sb, Path.GetFileNameWithoutExtension(GetArg(sb, 0))); }
                        catch { return false; }
                        return true;
                }
                return false;
            }
        }

        public static string GetDirectoryElements(string dir, int first, int last)
        {
            if (dir == null)
                return "";

            string[] dirSplit = dir.Split(Path.DirectorySeparatorChar,
                Path.AltDirectorySeparatorChar);
            int count = dirSplit.Length;

            if ((first == 0) && (last == 0))
            {
                first = 1;
                last = count;
            }

            if (first < 0) first = (count + 1) + first;
            if (last < 0) last = (count + 1) + last;

            if ((first < 1) && (last < 1))
            {
                return String.Empty;
            }
            else if ((first > count) && (last > count))
            {
                return String.Empty;
            }
            else
            {
                int i;
                StringBuilder sb = new StringBuilder();

                if (first < 1) first = 1;
                if (first > count) first = count;
                if (last < 1) last = 1;
                if (last > count) last = count;

                if (last >= first)
                {
                    for (i = first; i <= last; i++)
                    {
                        sb.Append(dirSplit[i - 1]);
                        sb.Append(Path.DirectorySeparatorChar);
                    }
                }
                else
                {
                    for (i = first; i >= last; i--)
                    {
                        sb.Append(dirSplit[i - 1]);
                        sb.Append(Path.DirectorySeparatorChar);
                    }
                }

                return sb.ToString(0, sb.Length - 1);
            }
        }

        public static string ReplaceMultiple(string s, NameValueCollection tags, int maxLen)
        {
            List<string> find = new List<string>();
            List<string> replace = new List<string>();

            foreach (string tag in tags.AllKeys)
            {
                string key = '%' + tag.ToLower() + '%';
                string val = tags[tag];
                if (!find.Contains(key) && val != null && val != "")
                {
                    find.Add(key);
                    replace.Add(val);
                }
            }

            return ReplaceMultiple(s, find, replace, maxLen);
        }

        public delegate bool CheckIfExists(string output);

        public static string ReplaceMultiple(string fmt, NameValueCollection tags, string unique_key, CheckIfExists exists, int maxLen)
        {
            string result = ReplaceMultiple(fmt, tags, maxLen);
            if (result == String.Empty || result == null)
                return result;
            int unique = 1;
            try
            {
                while (exists(result))
                {
                    tags[unique_key] = unique.ToString();
                    string new_result = ReplaceMultiple(fmt, tags, maxLen);
                    if (new_result == result || new_result == String.Empty || new_result == null)
                        break;
                    result = new_result;
                    unique++;
                }
            }
            catch { }
            return result;
        }

        public static string Shorten(string f, string s, int maxLen)
        {
            return maxLen <= 0 || maxLen >= s.Length || f == "music" || f == "path" /*|| f == "filename"*/ || f == "filename_ext" || f == "directoryname" ?
                s : s.Substring(0, maxLen);
        }

        public static string ReplaceMultiple(string s, List<string> find, List<string> replace, int maxLen)
        {
            if (find.Count != replace.Count)
            {
                throw new ArgumentException();
            }
            StringBuilder sb;
            int iChar, iFind;
            string f;
            bool found;
            List<TitleFormatFunctionInfo> formatFunctions = new List<TitleFormatFunctionInfo>();
            bool quote = false;

            sb = new StringBuilder();

            for (iChar = 0; iChar < s.Length; iChar++)
            {
                found = false;

                if (quote)
                {
                    if (s[iChar] == '\'')
                    {
                        if (iChar > 0 && s[iChar - 1] == '\'')
                            sb.Append(s[iChar]);
                        quote = false;
                        continue;
                    }
                    sb.Append(s[iChar]);
                    continue;
                }

                if (s[iChar] == '\'')
                {
                    quote = true;
                    continue;
                }

                if (s[iChar] == '[')
                {
                    formatFunctions.Add(new TitleFormatFunctionInfo("[", sb.Length));
                    continue;
                }

                if (s[iChar] == '$')
                {
                    int funcEnd = s.IndexOf('(', iChar + 1);
                    if (funcEnd < 0)
                        return null;
                    formatFunctions.Add(new TitleFormatFunctionInfo(s.Substring(iChar + 1, funcEnd - iChar - 1), sb.Length));
                    iChar = funcEnd;
                    continue;
                }

                if (s[iChar] == ',')
                {
                    if (formatFunctions.Count < 1)
                        return null;
                    formatFunctions[formatFunctions.Count - 1].NextArg(sb.Length);
                    continue;
                }

                if (s[iChar] == ']')
                {
                    if (formatFunctions.Count < 1 ||
                        formatFunctions[formatFunctions.Count - 1].func != "["
                        || !formatFunctions[formatFunctions.Count - 1].Finalise(sb))
                        return null;
                    formatFunctions.RemoveAt(formatFunctions.Count - 1);
                    continue;
                }

                if (s[iChar] == ')')
                {
                    if (formatFunctions.Count < 1 ||
                        formatFunctions[formatFunctions.Count - 1].func == "["
                        || !formatFunctions[formatFunctions.Count - 1].Finalise(sb))
                        return null;
                    formatFunctions.RemoveAt(formatFunctions.Count - 1);
                    continue;
                }

                for (iFind = 0; iFind < find.Count; iFind++)
                {
                    f = find[iFind];
                    if ((f.Length <= (s.Length - iChar)) && (s.Substring(iChar, f.Length) == f))
                    {
                        if (formatFunctions.Count > 0)
                        {
                            if (replace[iFind] != null)
                            {
                                formatFunctions[formatFunctions.Count - 1].Found();
                                sb.Append(Shorten(f, replace[iFind], maxLen));
                            }
                        }
                        else
                        {
                            if (replace[iFind] != null)
                                sb.Append(Shorten(f, replace[iFind], maxLen));
                            else
                                return null;
                        }
                        iChar += f.Length - 1;
                        found = true;
                        break;
                    }
                }

                if (!found)
                {
                    sb.Append(s[iChar]);
                }
            }

            return sb.ToString();
        }

        public static string EmptyStringToNull(string s)
        {
            return ((s != null) && (s.Length == 0)) ? null : s;
        }
    }
}
