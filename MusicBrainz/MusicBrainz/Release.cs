// Release.cs
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
    public sealed class Release : MusicBrainzItem
    {
        
        #region Private
        
        const string EXTENSION = "release";
        ReleaseType? type;
        ReleaseStatus? status;
        string language;
        string script;
        string asin;
        ReadOnlyCollection<Disc> discs;
        ReadOnlyCollection<Event> events;
        ReadOnlyCollection<Track> tracks;
        int? track_number;
        
        #endregion
        
        #region Constructors

        Release (string id) : base (id)
        {
        }

        internal Release (XmlReader reader) : base (reader, null, false)
        {
        }
        
        #endregion
        
        #region Protected
        
        internal override string UrlExtension {
            get { return EXTENSION; }
        }
        
        static readonly string [] track_params = new string [] { "tracks", "track-level-rels", "artist" };
        
        internal override void CreateIncCore (StringBuilder builder)
        {
            AppendIncParameters (builder, "release-events", "labels");
            if (discs == null) AppendIncParameters (builder, "discs");
            if (tracks == null) {
                AppendIncParameters (builder, track_params);
                AllRelsLoaded = false;
            }
            base.CreateIncCore (builder);
        }

        internal override void LoadMissingDataCore ()
        {
            Release release = new Release (Id);
            type = release.GetReleaseType ();
            status = release.GetReleaseStatus ();
            language = release.GetLanguage ();
            script = release.GetScript ();
            asin = release.GetAsin ();
            events = release.GetEvents ();
            if (discs == null) discs = release.GetDiscs ();
            if (tracks == null) tracks = release.GetTracks ();
            base.LoadMissingDataCore (release);
        }

        internal override void ProcessAttributes (XmlReader reader)
        {
            // How sure am I about getting the type and status in the "Type Status" format?
            // MB really ought to specify these two things seperatly.
            string type_string = reader ["type"];
            if (type_string != null) {
                foreach (string token in type_string.Split (' ')) {
                    if (type == null) {
                        type = Utils.StringToEnumOrNull<ReleaseType> (token);
                        if (type != null) continue;
                    }
                    this.status = Utils.StringToEnumOrNull<ReleaseStatus> (token);
                }
            }
        }

        internal override void ProcessXmlCore (XmlReader reader)
        {
            switch (reader.Name) {
            case "text-representation":
                language = reader["language"];
                script = reader["script"];
                break;
            case "asin":
                asin = reader.ReadString ();
                break;
            case "disc-list":
                if (reader.ReadToDescendant ("disc")) {
                    List<Disc> discs = new List<Disc> ();
                    do discs.Add (new Disc (reader.ReadSubtree ()));
                    while (reader.ReadToNextSibling ("disc"));
                    this.discs = discs.AsReadOnly ();
                }
                break;
            case "release-event-list":
                if (!AllDataLoaded) {
                    reader.Skip (); // FIXME this is a workaround for Mono bug 334752
                    return;
                }
                if (reader.ReadToDescendant ("event")) {
                    List<Event> events = new List<Event> ();
                    do events.Add (new Event (reader.ReadSubtree ()));
                    while (reader.ReadToNextSibling ("event"));
                    this.events = events.AsReadOnly ();
                }
                break;
            case "track-list":
                string offset = reader["offset"];
                if (offset != null)
                    track_number = int.Parse (offset) + 1;
                if (reader.ReadToDescendant ("track")) {
                    List<Track> tracks = new List<Track> ();
                    do tracks.Add (new Track (reader.ReadSubtree (), GetArtist (), AllDataLoaded));
                    while (reader.ReadToNextSibling ("track"));
                    this.tracks = tracks.AsReadOnly ();
                }
                break;
            default:
                base.ProcessXmlCore (reader);
                break;
            }
        }
        
        #endregion

        #region Public

        [Queryable ("reid")]
        public override string Id {
            get { return base.Id; }
        }

        [Queryable ("release")]
        public override string GetTitle ()
        { 
            return base.GetTitle ();
        }

        [Queryable ("type")]
        public ReleaseType GetReleaseType ()
        {
            return GetPropertyOrDefault (ref type, ReleaseType.None);
        }

        [Queryable ("status")]
        public ReleaseStatus GetReleaseStatus ()
        {
            return GetPropertyOrDefault (ref status, ReleaseStatus.None);
        }

        public string GetLanguage ()
        {
            return GetPropertyOrNull (ref language);
        }

        [Queryable ("script")]
        public string GetScript ()
        {
            return GetPropertyOrNull (ref script);
        }

        [Queryable ("asin")]
        public string GetAsin ()
        {
            return GetPropertyOrNull (ref asin);
        }

        [QueryableMember("Count", "discids")]
        public ReadOnlyCollection<Disc> GetDiscs ()
        { 
            return GetPropertyOrNew (ref discs);
        }

        public ReadOnlyCollection<Event> GetEvents ()
        {
            return GetPropertyOrNew (ref events);
        }

        [QueryableMember ("Count", "tracks")]
        public ReadOnlyCollection<Track> GetTracks ()
        {
            return GetPropertyOrNew (ref tracks);
        }

        internal int TrackNumber {
            get { return track_number ?? -1; }
        }

        #endregion
        
        #region Static

        public static Release Get (string id)
        {
            if (id == null) throw new ArgumentNullException ("id");
            return new Release (id);
        }

        public static Query<Release> Query (string title)
        {
            if (title == null) throw new ArgumentNullException ("title");
            
            ReleaseQueryParameters parameters = new ReleaseQueryParameters ();
            parameters.Title = title;
            return Query (parameters);
        }

        public static Query<Release> Query (string title, string artist)
        {
            if (title == null) throw new ArgumentNullException ("title");
            if (artist == null) throw new ArgumentNullException ("artist");
            
            ReleaseQueryParameters parameters = new ReleaseQueryParameters ();
            parameters.Title = title;
            parameters.Artist = artist;
            return Query (parameters);
        }
        
        public static Query<Release> Query (Disc disc)
        {
            if (disc == null) throw new ArgumentNullException ("disc");
            
            ReleaseQueryParameters parameters = new ReleaseQueryParameters ();
            parameters.DiscId = disc.Id;
            return Query (parameters);
        }

        public static Query<Release> Query (ReleaseQueryParameters parameters)
        {
            if (parameters == null) throw new ArgumentNullException ("parameters");
            return new Query<Release> (EXTENSION, parameters.ToString ());
        }

		//public static Query<Release> QueryFromDevice(string device)
		//{
		//    if (device == null) throw new ArgumentNullException ("device");
            
		//    ReleaseQueryParameters parameters = new ReleaseQueryParameters ();
		//    parameters.DiscId = LocalDisc.GetFromDevice (device).Id;
		//    return Query (parameters);
		//}

        public static Query<Release> QueryLucene (string luceneQuery)
        {
            if (luceneQuery == null) throw new ArgumentNullException ("luceneQuery");
            return new Query<Release> (EXTENSION, CreateLuceneParameter (luceneQuery));
        }

        public static implicit operator string (Release release)
        {
            return release.ToString ();
        }
        
        #endregion

    }
    
    #region Ancillary Types
    
    public enum ReleaseType
    {
        None,
        Album,
        Single,
        EP,
        Compilation,
        Soundtrack,
        Spokenword,
        Interview,
        Audiobook,
        Live,
        Remix,
        Other
    }

    public enum ReleaseStatus
    {
        None,
        Official,
        Promotion,
        Bootleg,
        PsudoRelease
    }

    public enum ReleaseFormat
    {
        None,
        Cartridge,
        Cassette,
        CD,
        DAT,
        Digital,
        DualDisc,
        DVD,
        LaserDisc,
        MiniDisc,
        Other,
        ReelToReel,
        SACD,
        Vinyl
    }

    public sealed class ReleaseQueryParameters : ItemQueryParameters
    {
        string disc_id;
        public string DiscId {
            get { return disc_id; }
            set { disc_id = value; }
        }

        string date;
        public string Date {
            get { return date; }
            set { date = value; }
        }

        string asin;
        public string Asin {
            get { return asin; }
            set { asin = value; }
        }

        string language;
        public string Language {
            get { return language; }
            set { language = value; }
        }

        string script;
        public string Script {
            get { return script; }
            set { script = value; }
        }

        internal override void ToStringCore (StringBuilder builder)
        {
            if (disc_id != null) {
                builder.Append ("&discid=");
                builder.Append (disc_id);
            }
            if (date != null) {
                builder.Append ("&date=");
                Utils.PercentEncode (builder, date);
            }
            if (asin != null) {
                builder.Append ("&asin=");
                builder.Append (asin);
            }
            if (language != null) {
                builder.Append ("&lang=");
                builder.Append (language);
            }
            if (script != null) {
                builder.Append ("&script=");
                builder.Append (script);
            }
        }
    }
    
    #endregion
    
}
