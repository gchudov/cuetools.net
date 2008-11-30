// MusicBrainzEntity.cs
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
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Text;
using System.Xml;

namespace MusicBrainz
{
    // A person-like entity, such as an artist or a label.
    public abstract class MusicBrainzEntity : MusicBrainzObject
    {
        
        #region Private
        
        string name;
        string sort_name;
        string disambiguation;
        string begin_date;
        string end_date;
        ReadOnlyCollection<string> aliases;
        
        #endregion
        
        #region Constructors
        
        internal MusicBrainzEntity (string id, string parameters) : base (id, parameters)
        {
        }

        internal MusicBrainzEntity (XmlReader reader, bool all_rels_loaded) : base (reader, all_rels_loaded)
        {
        }
        
        #endregion
        
        #region Protected

        internal override void CreateIncCore (StringBuilder builder)
        {
            if (aliases == null) AppendIncParameters (builder, "aliases");
            base.CreateIncCore (builder);
        }

        internal void LoadMissingDataCore (MusicBrainzEntity entity)
        {
            name = entity.GetName ();
            sort_name = entity.GetSortName ();
            disambiguation = entity.GetDisambiguation ();
            begin_date = entity.GetBeginDate ();
            end_date = entity.GetEndDate ();
            if (aliases == null) aliases = entity.GetAliases ();
            base.LoadMissingDataCore (entity);
        }

        internal override void ProcessXmlCore (XmlReader reader)
        {
            switch (reader.Name) {
            case "name":
                name = reader.ReadString ();
                break;
            case "sort-name":
                sort_name = reader.ReadString ();
                break;
            case "disambiguation":
                disambiguation = reader.ReadString ();
                break;
            case "life-span":
                begin_date = reader ["begin"];
                end_date = reader ["end"];
                break;
            case "alias-list":
                if (reader.ReadToDescendant ("alias")) {
                    List<string> aliases = new List<string> ();
                    do aliases.Add (reader.ReadString ());
                    while (reader.ReadToNextSibling ("alias"));
                    this.aliases = aliases.AsReadOnly ();
                }
                break;
            default:
                base.ProcessXmlCore (reader);
                break;
            }
        }
        
        internal static string CreateNameParameter (string name)
        {
            StringBuilder builder = new StringBuilder (name.Length + 6);
            builder.Append ("&name=");
            Utils.PercentEncode (builder, name);
            return builder.ToString ();
        }
        
        #endregion

        #region Public

        public virtual string GetName ()
        {
            return GetPropertyOrNull (ref name);
        }

        [Queryable ("sortname")]
        public virtual string GetSortName ()
        {
            return GetPropertyOrNull (ref sort_name);
        }

        [Queryable ("comment")]
        public virtual string GetDisambiguation ()
        {
            return GetPropertyOrNull (ref disambiguation);
        }

        [Queryable ("begin")]
        public virtual string GetBeginDate ()
        {
            return GetPropertyOrNull (ref begin_date);
        }

        [Queryable ("end")]
        public virtual string GetEndDate ()
        {
            return GetPropertyOrNull (ref end_date);
        }

        [QueryableMember ("Contains", "alias")]
        public virtual ReadOnlyCollection<string> GetAliases ()
        {
            return GetPropertyOrNew (ref aliases);
        }
        
        public override string ToString ()
        {
            return name;
        }
        
        #endregion
        
    }
}
