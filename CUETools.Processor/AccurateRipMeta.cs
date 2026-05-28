using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using CUETools.AccurateRip;
using CUETools.CDImage;
using CUETools.CTDB;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace CUETools.Processor
{
    public static class AccurateRipMeta
    {
        public const string SourceKey = "accurateripmeta";
        public const string DisplayName = "AccurateRip Meta";
        public const string Endpoint = "http://meta.accuraterip.com/discmatch";

        public static bool TryParseMetadata(string json, CDImageLayout toc, out CUEMetadataEntry entry)
        {
            entry = null;

            if (toc == null || string.IsNullOrWhiteSpace(json))
                return false;

            JObject response;
            try
            {
                response = JObject.Parse(json);
            }
            catch (JsonException ex)
            {
                System.Diagnostics.Trace.WriteLine("AccurateRip Meta JSON parse failed: " + ex.Message);
                return false;
            }

            JArray tracks = response["tracks"] as JArray;
            int audioTracks = (int)toc.AudioTracks;
            if (tracks == null || tracks.Count < audioTracks)
                return false;

            List<JObject> orderedTracks = OrderTracksByTrackNumber(tracks, audioTracks);
            if (orderedTracks == null)
                return false;

            if (string.IsNullOrWhiteSpace(Clean(response["Artist"])) && string.IsNullOrWhiteSpace(Clean(response["Album"])))
                return false;

            var metadata = new CUEMetadata(toc.TOCID, audioTracks)
            {
                Artist = NormalizeProviderList(response["Artist"]),
                Title = Clean(response["Album"]),
                Year = Clean(response["Year"]),
                Genre = FirstNonEmptyLine(response["Genre"]),
                Barcode = Clean(response["UPC"]),
                Label = Clean(response["Label"]),
                LabelNo = Clean(response["CatalogNum"])
            };

            // AccurateRip Meta also exposes Styles, Composers, and Conductors. Keep them unused until CUETools has explicit fields for them; do not overwrite Genre with Styles.
            for (int i = 0; i < audioTracks; i++)
            {
                JObject track = orderedTracks[i];

                metadata.Tracks[i].Title = Clean(track["Title"]);
                metadata.Tracks[i].Artist = NormalizeProviderList(track["Artist"]);
                metadata.Tracks[i].ISRC = Clean(track["ISRC"]);
            }

            string coverUrl = NormalizeCoverUrl(response["_arturl"]);
            if (coverUrl != "")
            {
                metadata.AlbumArt.Add(new CTDBResponseMetaImage
                {
                    uri = coverUrl,
                    uri150 = coverUrl,
                    primary = true
                });
            }

            entry = new CUEMetadataEntry(metadata, toc, SourceKey);
            return true;
        }

        internal static string CreateRequestDiscId(CDImageLayout toc)
        {
            return string.Format(CultureInfo.InvariantCulture, "{0:000}-{1}", toc.AudioTracks, AccurateRipVerify.CalculateAccurateRipId(toc));
        }

        internal static string CreateRequestBody(string accurateRipDiscId)
        {
            return JsonConvert.SerializeObject(new AccurateRipMetaRequest
            {
                accurateripdiscid = accurateRipDiscId
            });
        }

        private static string Clean(JToken value)
        {
            if (value == null || value.Type == JTokenType.Null || value.Type == JTokenType.Array || value.Type == JTokenType.Object)
                return "";

            JValue scalar = value as JValue;
            // The live contract uses strings for identifiers such as UPC. Numeric coercion is fail-soft only and cannot preserve leading zeroes.
            return scalar == null || scalar.Value == null ? "" : Convert.ToString(scalar.Value, CultureInfo.InvariantCulture).Trim();
        }

        private static List<JObject> OrderTracksByTrackNumber(JArray tracks, int audioTracks)
        {
            var byNumber = new Dictionary<int, JObject>();
            foreach (JToken item in tracks)
            {
                JObject track = item as JObject;
                if (track == null)
                    return null;

                int trackNumber;
                if (!int.TryParse(Clean(track["TrackNumber"]), NumberStyles.Integer, CultureInfo.InvariantCulture, out trackNumber))
                    return null;

                if (trackNumber < 1 || trackNumber > audioTracks)
                    continue;

                if (byNumber.ContainsKey(trackNumber))
                    return null;

                byNumber.Add(trackNumber, track);
            }

            if (byNumber.Count != audioTracks)
                return null;

            return Enumerable.Range(1, audioTracks)
                .Select(trackNumber => byNumber[trackNumber])
                .ToList();
        }

        private static string NormalizeCoverUrl(JToken value)
        {
            string cleaned = Clean(value);
            if (cleaned == "")
                return "";

            Uri uri;
            if (!Uri.TryCreate(cleaned, UriKind.Absolute, out uri))
                return "";

            if (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps)
                return "";

            return uri.AbsoluteUri;
        }

        private static string FirstNonEmptyLine(JToken value)
        {
            string text = Clean(value);
            if (string.IsNullOrWhiteSpace(text))
                return "";

            return text
                .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(line => line.Trim())
                .FirstOrDefault(line => line.Length > 0) ?? "";
        }

        private static string NormalizeProviderList(JToken value)
        {
            string text = Clean(value);
            if (string.IsNullOrWhiteSpace(text))
                return "";

            return string.Join(" & ", text
                .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(line => line.Trim())
                .Where(line => line.Length > 0)
                .Distinct()
                .ToArray());
        }
    }

    public sealed class AccurateRipMetaProvider
    {
        private readonly IAccurateRipMetaTransport transport;

        public AccurateRipMetaProvider()
            : this(new AccurateRipMetaHttpTransport())
        {
        }

        public AccurateRipMetaProvider(IAccurateRipMetaTransport transport)
        {
            this.transport = transport ?? throw new ArgumentNullException(nameof(transport));
        }

        public CUEMetadataEntry Lookup(CDImageLayout toc, IWebProxy proxy)
        {
            if (toc == null)
                return null;

            try
            {
                string accurateRipDiscId = AccurateRipMeta.CreateRequestDiscId(toc);
                string body = AccurateRipMeta.CreateRequestBody(accurateRipDiscId);
                string json = transport.Post(AccurateRipMeta.Endpoint, body, proxy);

                return AccurateRipMeta.TryParseMetadata(json, toc, out CUEMetadataEntry entry) ? entry : null;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine("AccurateRip Meta lookup failed: " + ex.Message);
                return null;
            }
        }
    }

    public interface IAccurateRipMetaTransport
    {
        string Post(string url, string body, IWebProxy proxy);
    }

    public sealed class AccurateRipMetaHttpTransport : IAccurateRipMetaTransport
    {
        public string Post(string url, string body, IWebProxy proxy)
        {
            var request = (HttpWebRequest)WebRequest.Create(url);
            request.Method = "POST";
            request.UserAgent = "Mozilla/5.0";
            request.ContentType = "application/x-www-form-urlencoded";
            request.AllowAutoRedirect = false;
            request.Timeout = 15000;
            request.ReadWriteTimeout = 30000;
            if (proxy != null)
                request.Proxy = proxy;

            byte[] bytes = Encoding.UTF8.GetBytes(body);
            request.ContentLength = bytes.Length;
            using (Stream requestStream = request.GetRequestStream())
                requestStream.Write(bytes, 0, bytes.Length);

            using (var response = (HttpWebResponse)request.GetResponse())
            using (Stream responseStream = response.GetResponseStream())
            using (var reader = new StreamReader(responseStream, Encoding.UTF8))
                return reader.ReadToEnd();
        }
    }

    internal sealed class AccurateRipMetaRequest
    {
        public string accurateripdiscid { get; set; }
    }
}
