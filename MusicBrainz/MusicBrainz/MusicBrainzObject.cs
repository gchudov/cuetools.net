// MusicBrainzObject.cs
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
using System.IO;
using System.Net;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Xml;

namespace MusicBrainz
{
    public abstract class MusicBrainzObject
    {
        
        #region Private Fields
        
        static DateTime last_accessed;
        static readonly TimeSpan min_interval = new TimeSpan (1000000); // 0.1 second
        static readonly object server_mutex = new object ();
        static readonly string [] rels_params = new string [] {
            "artist-rels",
            "release-rels",
            "track-rels",
            "label-rels",
            "url-rels"
        };
        
        bool all_data_loaded;
        bool all_rels_loaded;
        
        string id;
        byte score;
        ReadOnlyCollection<Relation<Artist>> artist_rels;
        ReadOnlyCollection<Relation<Release>> release_rels;
        ReadOnlyCollection<Relation<Track>> track_rels;
        ReadOnlyCollection<Relation<Label>> label_rels;
        ReadOnlyCollection<UrlRelation> url_rels;
        
        #endregion
        
        #region Constructors

        internal MusicBrainzObject (string id, string parameters)
        {
            all_data_loaded = true;
            CreateFromId (id, parameters ?? CreateInc ());
        }

        internal MusicBrainzObject (XmlReader reader, bool all_rels_loaded)
        {
            this.all_rels_loaded = all_rels_loaded;
            CreateFromXml (reader);
        }
        
        #endregion
        
        #region Private Methods

        string CreateInc ()
        {
            StringBuilder builder = new StringBuilder ();
            CreateIncCore (builder);
            return builder.ToString ();
        }
        
        void CreateFromId (string id, string parameters)
        {
            XmlProcessingClosure (
                CreateUrl (UrlExtension, id, parameters),
                delegate (XmlReader reader) {
                    reader.ReadToFollowing ("metadata");
                    reader.Read ();
                    CreateFromXml (reader.ReadSubtree ());
                    reader.Close ();
                }
            );
        }
        
        void CreateFromXml (XmlReader reader)
        {
            reader.Read ();
            id = reader ["id"];
            byte.TryParse (reader ["ext:score"], out score);
            ProcessAttributes (reader);
            while (reader.Read () && reader.NodeType != XmlNodeType.EndElement) {
                if (reader.Name == "relation-list") {
                    all_rels_loaded = true;
                    switch (reader ["target-type"]) {
                    case "Artist":
                        artist_rels = CreateRelation<Artist> (reader.ReadSubtree ());
                        break;
                    case "Release":
                        release_rels = CreateRelation<Release> (reader.ReadSubtree ());
                        break;
                    case "Track":
                        track_rels = CreateRelation<Track> (reader.ReadSubtree ());
                        break;
                    case "Label":
                        label_rels = CreateRelation<Label> (reader.ReadSubtree ());
                        break;
                    case "Url":
                        url_rels = CreateUrlRelation (reader.ReadSubtree ());
                        break;
                    }
                } else
                    ProcessXml (reader.ReadSubtree ());
            }
            reader.Close ();
        }

        void ProcessXml (XmlReader reader)
        {
            reader.Read ();
            ProcessXmlCore (reader);
            reader.Close ();
        }
        
        #endregion
        
        #region Protected
        
        internal bool AllDataLoaded {
            get { return all_data_loaded; }
        }
        
        internal bool AllRelsLoaded {
            get { return all_rels_loaded; }
            set { all_rels_loaded = value; }
        }
        
        internal virtual void CreateIncCore (StringBuilder builder)
        {
            if (!all_rels_loaded)
                AppendIncParameters (builder, rels_params);
        }
        
        internal static void AppendIncParameters (StringBuilder builder, string parameter)
        {
            builder.Append (builder.Length == 0 ? "&inc=" : "+");
            builder.Append (parameter);
        }
        
        internal static void AppendIncParameters (StringBuilder builder, string parameter1, string parameter2)
        {
            builder.Append (builder.Length == 0 ? "&inc=" : "+");
            builder.Append (parameter1);
            builder.Append ('+');
            builder.Append (parameter2);
        }

        internal static void AppendIncParameters (StringBuilder builder, string [] parameters)
        {
            foreach (string parameter in parameters)
                AppendIncParameters (builder, parameter);
        }
        
        internal void LoadMissingData ()
        {
            if (!all_data_loaded) {
                LoadMissingDataCore ();
                all_data_loaded = true;
            }
        }

        internal void LoadMissingDataCore (MusicBrainzObject obj)
        {
            if (!all_rels_loaded) {
                artist_rels = obj.GetArtistRelations ();
                release_rels = obj.GetReleaseRelations ();
                track_rels = obj.GetTrackRelations ();
                label_rels = obj.GetLabelRelations ();
                url_rels = obj.GetUrlRelations ();
            }
        }
        
        internal T GetPropertyOrNull<T> (ref T field_reference) where T : class
        {
            if (field_reference == null) LoadMissingData ();
            return field_reference;
        }

        internal T GetPropertyOrDefault<T> (ref T? field_reference) where T : struct
        {
            return GetPropertyOrDefault (ref field_reference, default (T));
        }
        
        internal T GetPropertyOrDefault<T> (ref T? field_reference, T default_value) where T : struct
        {
            if (field_reference == null) LoadMissingData ();
            return field_reference ?? default_value;
        }
        
        internal ReadOnlyCollection<T> GetPropertyOrNew<T> (ref ReadOnlyCollection<T> field_reference)
        {
            return GetPropertyOrNew (ref field_reference, true);
        }
        
        internal ReadOnlyCollection<T> GetPropertyOrNew<T> (ref ReadOnlyCollection<T> field_reference, bool condition)
        {
            if (field_reference == null && condition) LoadMissingData ();
            return field_reference ?? new ReadOnlyCollection<T> (new T [0]);
        }

        internal virtual void ProcessXmlCore (XmlReader reader)
        {
            reader.Skip (); // FIXME this is a workaround for Mono bug 334752
        }

        internal virtual void ProcessAttributes (XmlReader reader)
        {
        }
        
        internal abstract void LoadMissingDataCore ();
        internal abstract string UrlExtension { get; }
        
        #endregion
        
        #region Public

        public virtual string Id {
            get { return id; }
        }

        public virtual byte Score {
            get { return score; }
        }

        public virtual ReadOnlyCollection<Relation<Artist>> GetArtistRelations ()
        {
            return GetPropertyOrNew (ref artist_rels, !all_rels_loaded);
        }

        public virtual ReadOnlyCollection<Relation<Release>> GetReleaseRelations ()
        {
            return GetPropertyOrNew (ref release_rels, !all_rels_loaded);
        }

        public virtual ReadOnlyCollection<Relation<Track>> GetTrackRelations ()
        {
            return GetPropertyOrNew (ref track_rels, !all_rels_loaded);
        }

        public virtual ReadOnlyCollection<Relation<Label>> GetLabelRelations ()
        {
            return GetPropertyOrNew (ref label_rels, !all_rels_loaded);
        }

        public virtual ReadOnlyCollection<UrlRelation> GetUrlRelations ()
        {
            return GetPropertyOrNew (ref url_rels, !all_rels_loaded);
        }
        
        public override bool Equals (object obj)
        {
            return this == obj as MusicBrainzObject;
        }
        
        public static bool operator ==(MusicBrainzObject obj1, MusicBrainzObject obj2)
        {
            if (Object.ReferenceEquals (obj1, null)) {
                return Object.ReferenceEquals (obj2, null);
            }
            return !Object.ReferenceEquals (obj2, null) && obj1.GetType () == obj2.GetType () && obj1.Id == obj2.Id;
        }
        
        public static bool operator !=(MusicBrainzObject obj1, MusicBrainzObject obj2)
        {
            return !(obj1 == obj2);
        }

        public override int GetHashCode ()
        {
            return (GetType ().Name + Id).GetHashCode ();
        }
        
        #endregion

        #region Static

        static ReadOnlyCollection<Relation<T>> CreateRelation<T> (XmlReader reader) where T : MusicBrainzObject
        {
            List<Relation<T>> relations = new List<Relation<T>> ();
            while (reader.ReadToFollowing ("relation")) {
                string type = reader ["type"];
                RelationDirection direction = RelationDirection.Forward;
                string direction_string = reader ["direction"];
                if (direction_string != null && direction_string == "backward")
                    direction = RelationDirection.Backward;
                string begin = reader ["begin"];
                string end = reader ["end"];
                string attributes_string = reader ["attributes"];
                string [] attributes = attributes_string == null
                    ? null : attributes_string.Split (' ');

                reader.Read ();
                relations.Add (new Relation<T> (
                    type,
                    ConstructMusicBrainzObjectFromXml<T> (reader.ReadSubtree ()),
                    direction,
                    begin,
                    end,
                    attributes));
            }
            reader.Close ();
            return relations.AsReadOnly ();
        }

        static ReadOnlyCollection<UrlRelation> CreateUrlRelation (XmlReader reader)
        {
            List<UrlRelation> url_rels = new List<UrlRelation> ();
            while (reader.ReadToDescendant ("relation")) {
                RelationDirection direction = RelationDirection.Forward;
                string direction_string = reader["direction"];
                if (direction_string != null && direction_string == "backward")
                    direction = RelationDirection.Backward;
                string attributes_string = reader["attributes"];
                string[] attributes = attributes_string == null
                    ? null : attributes_string.Split (' ');
                url_rels.Add (new UrlRelation (
                    reader["type"],
                    reader["target"],
                    direction,
                    reader["begin"],
                    reader["end"],
                    attributes));
            }
            return url_rels.AsReadOnly ();
        }

        static string CreateUrl (string url_extension, int limit, int offset, string parameters)
        {
            StringBuilder builder = new StringBuilder ();
            if (limit != 25) {
                builder.Append ("&limit=");
                builder.Append (limit);
            }
            if (offset != 0) {
                builder.Append ("&offset=");
                builder.Append (offset);
            }
            builder.Append (parameters);
            return CreateUrl (url_extension, string.Empty, builder.ToString ());
        }

        static string CreateUrl (string url_extension, string id, string parameters)
        {
            StringBuilder builder = new StringBuilder (
                MusicBrainzService.ServiceUrl.AbsoluteUri.Length + id.Length + parameters.Length + 9);
            builder.Append (MusicBrainzService.ServiceUrl.AbsoluteUri);
            builder.Append (url_extension);
            builder.Append ('/');
            builder.Append (id);
            builder.Append ("?type=xml");
            builder.Append (parameters);
            return builder.ToString ();
        }

        static void XmlProcessingClosure (string url, XmlProcessingDelegate code)
        {
			lock (server_mutex)
			{
				// Don't access the MB server twice within a second
				if (last_accessed != null)
				{
					TimeSpan time = DateTime.Now - last_accessed;
					if (min_interval > time)
						Thread.Sleep((min_interval - time).Milliseconds);
				}

				WebRequest request = WebRequest.Create(url);
				bool cache_implemented = false;

				request.Proxy = MusicBrainzService.Proxy;

				try
				{
					request.CachePolicy = MusicBrainzService.CachePolicy;
					cache_implemented = true;
				}
				catch (NotImplementedException) { }

				HttpWebResponse response = null;

				try
				{
					response = (HttpWebResponse)request.GetResponse();
				}
				catch (WebException e)
				{
					if (e.Response == null)
						throw e;
					response = (HttpWebResponse)e.Response;
				}

				switch (response.StatusCode)
				{
					case HttpStatusCode.BadRequest:
						throw new MusicBrainzInvalidParameterException();
					case HttpStatusCode.Unauthorized:
						throw new MusicBrainzUnauthorizedException();
					case HttpStatusCode.NotFound:
						throw new MusicBrainzNotFoundException();
					case HttpStatusCode.ServiceUnavailable:
						throw new MusicBrainzUnavailableException(response.StatusDescription);
					case HttpStatusCode.OK:
						break;
					default:
						throw new MusicBrainzUnavailableException(response.StatusDescription);
				}

				bool from_cache = cache_implemented && response.IsFromCache;

				try
				{
					MusicBrainzService.OnXmlRequest(url, from_cache);

					// Should we read the stream into a memory stream and run the XmlReader off of that?
					code(new XmlTextReader(response.GetResponseStream()));
				}
				finally
				{
					response.Close();
					if (!from_cache)
					{
						last_accessed = DateTime.Now;
					}
				}
			}
        }

        #endregion

        #region Query

        internal static string CreateLuceneParameter (string query)
        {
            StringBuilder builder = new StringBuilder (query.Length + 7);
            builder.Append ("&query=");
            Utils.PercentEncode (builder, query);
            return builder.ToString ();
        }

        internal static List<T> Query<T> (string url_extension,
                                          int limit, int offset,
                                          string parameters,
                                          out int? count) where T : MusicBrainzObject
        {
            int count_value = 0;
            List<T> results = new List<T> ();
            XmlProcessingClosure (
                CreateUrl (url_extension, limit, offset, parameters),
                delegate (XmlReader reader) {
                    reader.ReadToFollowing ("metadata");
                    reader.Read ();
                    int.TryParse (reader ["count"], out count_value);
                    while (reader.Read () && reader.NodeType == XmlNodeType.Element)
                        results.Add (ConstructMusicBrainzObjectFromXml<T> (reader.ReadSubtree ()));
                    reader.Close ();
                }
            );
            count = count_value == 0 ? results.Count : count_value;
            return results;
        }

        static T ConstructMusicBrainzObjectFromXml<T> (XmlReader reader) where T : MusicBrainzObject
        {
            ConstructorInfo constructor = typeof (T).GetConstructor (
                BindingFlags.NonPublic | BindingFlags.Instance,
                null,
                new Type [] { typeof (XmlReader) },
                null);
            return (T)constructor.Invoke (new object [] {reader});
        }

        #endregion

    }
    
    internal delegate void XmlProcessingDelegate (XmlReader reader);
}
