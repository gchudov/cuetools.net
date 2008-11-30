// Label.cs
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
using System.Xml;

namespace MusicBrainz
{
    public sealed class Label : MusicBrainzEntity
    {
        
        #region Private
        
        const string EXTENSION = "label";
        string country;
        LabelType? type;
        
        #endregion
        
        #region Constructors

        Label (string id) : base (id, null)
        {
        }

        internal Label (XmlReader reader) : base (reader, false)
        {
        }
        
        #endregion
        
        #region Protected
        
        internal override string UrlExtension {
            get { return EXTENSION; }
        }

        internal override void LoadMissingDataCore ()
        {
            Label label = new Label (Id);
            type = label.GetLabelType ();
            country = label.GetCountry ();
            base.LoadMissingDataCore (label);
        }

        internal override void ProcessAttributes (XmlReader reader)
        {
            type = Utils.StringToEnum<LabelType> (reader ["type"]);
        }

        internal override void ProcessXmlCore (XmlReader reader)
        {
            if (reader.Name == "country") {
                country = reader.ReadString ();
            } else base.ProcessXmlCore (reader);
        }
        
        #endregion

        #region Public
        
        public string GetCountry ()
        {
            return GetPropertyOrNull (ref country);
        }

        public LabelType GetLabelType ()
        {
            return GetPropertyOrDefault (ref type, LabelType.None);
        }
        
        #endregion
        
        #region Static

        public static Label Get (string id)
        {
            if (id == null) throw new ArgumentNullException ("id");
            return new Label (id);
        }

        public static Query<Label> Query (string name)
        {
            if (name == null) throw new ArgumentNullException ("name");
            return new Query<Label> (EXTENSION, CreateNameParameter (name));
        }

        public static Query<Label> QueryLucene (string luceneQuery)
        {
            if (luceneQuery == null) throw new ArgumentNullException ("luceneQuery");
            return new Query<Label> (EXTENSION, CreateLuceneParameter (luceneQuery));
        }

        public static implicit operator string (Label label)
        {
            return label.ToString ();
        }
        
        #endregion

    }
    
    #region Ancillary Types
    
    public enum LabelType
    {
        None,
        Distributor,
        Holding,
        OriginalProduction,
        BootlegProduction,
        ReissueProduction
    }
    
    #endregion
    
}
