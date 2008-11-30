// Utils.cs
//
// Copyright (c) 2008 Scott Peterson <lunchtimemama@gmail.com>
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

using System;
using System.Text;

namespace MusicBrainz
{
    internal static class Utils
    {
        public static string EnumToString (Enum enumeration)
        {
            string str = enumeration.ToString ();
            StringBuilder builder = new StringBuilder (str.Length);
            EnumToString (builder, str);
            return builder.ToString ();
        }
        
        public static void EnumToString (StringBuilder builder, string str)
        {
            builder.Append (str [0]);
            for (int i = 1; i < str.Length; i++) {
                if (str [i] >= 'A' && str [i] <= 'Z')
                    builder.Append ('-'); 
                builder.Append (str [i]);
            }
        }
        
        public static T StringToEnum<T> (string name) where T : struct
        {
            return StringToEnumOrNull<T> (name) ?? default (T);
        }
        
        public static T? StringToEnumOrNull<T> (string name) where T : struct
        {
            if (name != null)
                foreach (T value in Enum.GetValues (typeof (T)))
                    if (Enum.GetName (typeof (T), value) == name)
                        return value;
            return null;
        }
        
        public static void PercentEncode (StringBuilder builder, string value)
        {
            foreach (char c in value) {
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || 
                    c == '-' || c == '_' || c == '.' || c == '~')
                    builder.Append (c);
                else {
                    builder.Append ('%');
                    foreach (byte b in Encoding.UTF8.GetBytes (new char [] { c }))
                        builder.AppendFormat ("{0:X}", b);
                } 
            }
        }
    }
}
