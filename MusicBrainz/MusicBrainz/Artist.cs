// Artist.cs
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
    public sealed class Artist : MusicBrainzEntity
    {
        
        #region Private
        
        const string EXTENSION = "artist";
        ArtistReleaseType artist_release_type = DefaultArtistReleaseType;
        ArtistType? type;
        ReadOnlyCollection<Release> releases;
        bool have_all_releases;
        
        #endregion
        
        #region Constructors

        Artist (string id) : base (id, null)
        {
        }

        Artist (string id, ArtistReleaseType artist_release_type)
            : base (id, "&inc=" + artist_release_type.ToString ())
        {
            have_all_releases = true;
            this.artist_release_type = artist_release_type;
        }

        internal Artist (XmlReader reader) : base (reader, false)
        {
        }
        
        #endregion
        
        #region Protected
        
        internal override string UrlExtension {
            get { return EXTENSION; }
        }

        internal override void CreateIncCore (StringBuilder builder)
        {
            AppendIncParameters (builder, artist_release_type.ToString ());
            base.CreateIncCore (builder);
        }

        internal override void LoadMissingDataCore ()
        {
            Artist artist = new Artist (Id);
            type = artist.GetArtistType ();
            base.LoadMissingDataCore (artist);
        }

        internal override void ProcessAttributes (XmlReader reader)
        {
            switch (reader ["type"]) {
            case "Group":
                type = ArtistType.Group;
                break;
            case "Person":
                type = ArtistType.Person;
                break;
            }
        }

        internal override void ProcessXmlCore (XmlReader reader)
        {
            switch (reader.Name) {
            case "release-list":
                if (reader.ReadToDescendant ("release")) {
                    List<Release> releases = new List<Release> ();
                    do releases.Add (new Release (reader.ReadSubtree ()));
                    while (reader.ReadToNextSibling ("release"));
                    this.releases = releases.AsReadOnly ();
                }
                break;
            default:
                base.ProcessXmlCore (reader);
                break;
            }
        }
        
        #endregion

        #region Public
        
        static ArtistReleaseType default_artist_release_type = new ArtistReleaseType (ReleaseStatus.Official, ReleaseArtistType.SingleArtist);
        public static ArtistReleaseType DefaultArtistReleaseType {
            get { return default_artist_release_type; }
            set {
                if (value == null) throw new ArgumentNullException ("value");
                default_artist_release_type = value;
            }
        }
        
        public ArtistReleaseType ArtistReleaseType {
            get { return artist_release_type; }
            set {
                if (artist_release_type == value) {
                    return;
                }
                artist_release_type = value;
                releases = null;
                have_all_releases = false;
            }
        }

        [Queryable ("arid")]
        public override string Id {
            get { return base.Id; }
        }

        [Queryable ("artist")]
        public override string GetName ()
        {
            return base.GetName ();
        }

        [Queryable ("artype")]
        public ArtistType GetArtistType ()
        {
            return GetPropertyOrDefault (ref type, ArtistType.Unknown);
        }
        
        public ReadOnlyCollection<Release> GetReleases ()
        {
            return releases ?? (have_all_releases
                ? releases = new ReadOnlyCollection<Release> (new Release [0])
                : new Artist (Id, ArtistReleaseType).GetReleases ());
        }

        public ReadOnlyCollection<Release> GetReleases (ArtistReleaseType artistReleaseType)
        {
            return new Artist (Id, artistReleaseType).GetReleases ();
        }

        #endregion
        
        #region Static

        public static Artist Get (string id)
        {
            if (id == null) throw new ArgumentNullException ("id");
            return new Artist (id);
        }

        public static Query<Artist> Query (string name)
        {
            if (name == null) throw new ArgumentNullException ("name");
            return new Query<Artist> (EXTENSION, CreateNameParameter (name));
        }

        public static Query<Artist> QueryLucene (string luceneQuery)
        {
            if (luceneQuery == null) throw new ArgumentNullException ("luceneQuery");
            return new Query<Artist> (EXTENSION, CreateLuceneParameter (luceneQuery));
        }

        public static implicit operator string (Artist artist)
        {
            return artist.ToString ();
        }
        
        #endregion
        
    }
    
    #region Ancillary Types
    
    public enum ArtistType
    {
        Unknown,
        Group,
        Person
    }
    
    public enum ReleaseArtistType
    {
        VariousArtists,
        SingleArtist
    }
    
    public sealed class ArtistReleaseType
    {
        string str;

        public ArtistReleaseType (ReleaseType type, ReleaseArtistType artistType) : this ((Enum)type, artistType)
        {
        }

        public ArtistReleaseType (ReleaseStatus status, ReleaseArtistType artistType) : this ((Enum)status, artistType)
        {
        }
        
        public ArtistReleaseType (ReleaseType type, ReleaseStatus status, ReleaseArtistType artistType)
        {
            StringBuilder builder = new StringBuilder ();
            Format (builder, type, artistType);
            builder.Append ('+');
            Format (builder, status, artistType);
            str = builder.ToString ();
        }

        ArtistReleaseType (Enum enumeration, ReleaseArtistType artistType)
        {
            StringBuilder builder = new StringBuilder ();
            Format (builder, enumeration, artistType);
            str = builder.ToString ();
        }
        
        static void Format (StringBuilder builder, Enum enumeration, ReleaseArtistType artistType)
        {
            builder.Append (artistType == ReleaseArtistType.VariousArtists ? "va-" : "sa-");
            Utils.EnumToString (builder, enumeration.ToString ());
        }

        public override string ToString ()
        {
            return str;
        }

        public override bool Equals (object o)
        {
            return this == o as ArtistReleaseType;
        }
        
        public static bool operator ==(ArtistReleaseType artistReleaseType1, ArtistReleaseType artistReleaseType2)
        {
            if (Object.ReferenceEquals (artistReleaseType1, null)) {
                return Object.ReferenceEquals (artistReleaseType2, null);
            }
            return !Object.ReferenceEquals (artistReleaseType2, null) && artistReleaseType1.str == artistReleaseType2.str;
        }
        
        public static bool operator !=(ArtistReleaseType artistReleaseType1, ArtistReleaseType artistReleaseType2)
        {
            return !(artistReleaseType1 == artistReleaseType2);
        }
        
        public override int GetHashCode ()
        {
            return str.GetHashCode ();
        }
    }
    
    #endregion
    
}
