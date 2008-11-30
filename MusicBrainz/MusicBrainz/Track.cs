// Track.cs
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
    public sealed class Track : MusicBrainzItem
    {
        
        #region Private
        
        const string EXTENSION = "track";
        TimeSpan? duration;
        ReadOnlyCollection<Release> releases;
        ReadOnlyCollection<string> puids;
        
        #endregion
        
        #region Constructors

        Track (string id) : base (id)
        {
        }

        internal Track (XmlReader reader) : base (reader, null, false)
        {
        }

        internal Track (XmlReader reader, Artist artist, bool all_rels_loaded) : base (reader, artist, all_rels_loaded)
        {
        }
        
        #endregion

        #region Protected
        
        internal override string UrlExtension {
            get { return EXTENSION; }
        }
        
        internal override void CreateIncCore (StringBuilder builder)
        {
            if (releases == null) AppendIncParameters (builder, "releases");
            if (puids == null) AppendIncParameters (builder, "puids");
            base.CreateIncCore (builder);
        }

        internal override void LoadMissingDataCore ()
        {
            Track track = new Track (Id);
            duration = track.GetDuration ();
            if (releases == null) releases = track.GetReleases ();
            if (puids == null) puids = track.GetPuids ();
            base.LoadMissingDataCore (track);
        }

        internal override void ProcessXmlCore (XmlReader reader)
        {
            switch (reader.Name) {
            case "duration":
                duration = TimeSpan.FromMilliseconds (reader.ReadElementContentAsDouble ());
                break;
            case "release-list":
                if (reader.ReadToDescendant ("release")) {
                    List<Release> releases = new List<Release> ();
                    do releases.Add (new Release (reader.ReadSubtree ()));
                    while (reader.ReadToNextSibling ("release"));
                    this.releases = releases.AsReadOnly ();
                }
                break;
            case "puid-list":
                if (reader.ReadToDescendant ("puid")) {
                    List<string> puids = new List<string> ();
                    do puids.Add (reader["id"]);
                    while (reader.ReadToNextSibling ("puid"));
                    this.puids = puids.AsReadOnly ();
                }
                break;
            default:
                base.ProcessXmlCore (reader);
                break;
            }
        }
        
        #endregion

        #region Public

        [Queryable ("trid")]
        public override string Id {
            get { return base.Id; }
        }

        [Queryable ("track")]
        public override string GetTitle ()
        {
            return base.GetTitle ();
        }

        [Queryable ("dur")]
        public TimeSpan GetDuration ()
        {
            return GetPropertyOrDefault (ref duration);
        }

        [QueryableMember ("Contains", "release")]
        public ReadOnlyCollection<Release> GetReleases ()
        {
            return GetPropertyOrNew (ref releases);
        }

        public ReadOnlyCollection<string> GetPuids ()
        {
            return GetPropertyOrNew (ref puids);
        }

        public int GetTrackNumber (Release release)
        {
            if (release == null) throw new ArgumentNullException ("release");
            
            foreach (Release r in GetReleases ())
                if (r.Equals (release))
                    return r.TrackNumber;
            return -1;
        }

        #endregion
        
        #region Static

        public static Track Get (string id)
        {
            if (id == null) throw new ArgumentNullException ("id");
            return new Track (id);
        }

        public static Query<Track> Query (string title)
        {
            if (title == null) throw new ArgumentNullException ("title");
            
            TrackQueryParameters parameters = new TrackQueryParameters ();
            parameters.Title = title;
            return Query (parameters);
        }

        public static Query<Track> Query (string title, string artist)
        {
            if (title == null) throw new ArgumentNullException ("title");
            if (artist == null) throw new ArgumentNullException ("artist");
            
            TrackQueryParameters parameters = new TrackQueryParameters ();
            parameters.Title = title;
            parameters.Artist = artist;
            return Query (parameters);
        }
        
        public static Query<Track> Query (string title, string artist, string release)
        {
            if (title == null) throw new ArgumentNullException ("title");
            if (artist == null) throw new ArgumentNullException ("artist");
            if (release == null) throw new ArgumentNullException ("release");
            
            TrackQueryParameters parameters = new TrackQueryParameters ();
            parameters.Title = title;
            parameters.Artist = artist;
            parameters.Release = release;
            return Query (parameters);
        }

        public static Query<Track> Query (TrackQueryParameters parameters)
        {
            if (parameters == null) throw new ArgumentNullException ("parameters");
            return new Query<Track> (EXTENSION, parameters.ToString ());
        }

        public static Query<Track> QueryLucene (string luceneQuery)
        {
            if(luceneQuery == null) throw new ArgumentNullException ("luceneQuery"); 
            return new Query<Track> (EXTENSION, CreateLuceneParameter (luceneQuery));
        }

        public static implicit operator string (Track track)
        {
            return track.ToString ();
        }
        
        #endregion

    }
    
    #region Ancillary Types
    
    public sealed class TrackQueryParameters : ItemQueryParameters
    {
        string release;
        public string Release {
            get { return release; }
            set { release = value; }
        }

        string release_id;
        public string ReleaseId {
            get { return release_id; }
            set { release_id = value; }
        }

        uint? duration;
        public uint? Duration {
            get { return duration; }
            set { duration = value; }
        }

        int? track_number;
        public int? TrackNumber {
            get { return track_number; }
            set { track_number = value; }
        }

        string puid;
        public string Puid {
            get { return puid; }
            set { puid = value; }
        }

        internal override void ToStringCore (StringBuilder builder)
        {
            if (release != null) {
                builder.Append ("&release=");
                Utils.PercentEncode (builder, release);
            }
            if (release_id != null) {
                builder.Append ("&releaseid=");
                builder.Append (release_id);
            }
            if (duration != null) {
                builder.Append ("&duration=");
                builder.Append (duration.Value);
            }
            if (track_number != null) {
                builder.Append ("&tracknumber=");
                builder.Append (track_number.Value);
            }
            if (puid != null) {
                builder.Append ("&puid=");
                builder.Append (puid);
            }
        }
    }
    
    #endregion
    
}
