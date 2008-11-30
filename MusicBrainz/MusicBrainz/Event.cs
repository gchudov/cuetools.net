// Event.cs
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
using System.Xml;

namespace MusicBrainz
{
    public sealed class Event
    {
        string date;
        string country;
        string catalog_number;
        string barcode;
        Label label;
        ReleaseFormat format = ReleaseFormat.None;

        internal Event (XmlReader reader)
        {
            reader.Read ();
            date = reader ["date"];
            country = reader ["country"];
            catalog_number = reader ["catalog-number"];
            barcode = reader ["barcode"];
            format = Utils.StringToEnum<ReleaseFormat> (reader ["format"]);
            if (reader.ReadToDescendant ("label")) {
                label = new Label (reader.ReadSubtree ());
                reader.Read (); // FIXME this is a workaround for Mono bug 334752
            }
            reader.Close ();
        }

        public string Date {
            get { return date; }
        }

        public string Country {
            get { return country; }
        }

        public string CatalogNumber {
            get { return catalog_number; }
        }

        public string Barcode {
            get { return barcode; }
        }

        public Label Label {
            get { return label; }
        }

        public ReleaseFormat Format {
            get { return format; }
        }
    }
}
