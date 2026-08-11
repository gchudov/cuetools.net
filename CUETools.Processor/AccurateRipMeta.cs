using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading;
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
        internal const string ContentType = "application/x-www-form-urlencoded";
        internal const string CoverArtHost = "meta.accuraterip.com";
        // HTTPS on meta.accuraterip.com:443 was refused during contract review, so the live endpoint remains HTTP-only.
        public static readonly string Endpoint = "http://meta.accuraterip.com/discmatch";
        internal const int LookupTimeoutMilliseconds = 5000;
        internal const int ReadWriteTimeoutMilliseconds = 15000;

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

            List<JObject> orderedTracks = OrderTracksByTrackNumber(tracks, toc.FirstAudio, audioTracks);
            if (orderedTracks == null)
                return false;

            if (string.IsNullOrWhiteSpace(Clean(response["Artist"])) && string.IsNullOrWhiteSpace(Clean(response["Album"])))
                return false;

            var metadata = new CUEMetadata(toc.TOCID, audioTracks)
            {
                Artist = NormalizeProviderList(response["Artist"]),
                Title = Clean(response["Album"]),
                Year = Clean(response["Year"]),
                Genre = NormalizeGenreList(response["Genre"]),
                Barcode = CleanIdentifier(response["UPC"], "UPC"),
                Label = Clean(response["Label"]),
                LabelNo = CleanIdentifier(response["CatalogNum"], "CatalogNum")
            };

            // AccurateRip Meta also exposes Styles, Composers, Conductors, _albumid, and _tocid. Keep them unused until CUETools has explicit fields for them; do not overwrite Genre with Styles.
            for (int i = 0; i < audioTracks; i++)
            {
                JObject track = orderedTracks[i];

                metadata.Tracks[i].Title = Clean(track["Title"]);
                string trackArtist = NormalizeProviderList(track["Artist"]);
                metadata.Tracks[i].Artist = trackArtist != "" ? trackArtist : metadata.Artist;
                metadata.Tracks[i].ISRC = CleanIdentifier(track["ISRC"], "ISRC");
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
                AccurateRipDiscId = accurateRipDiscId
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

        private static string CleanIdentifier(JToken value, string fieldName)
        {
            if (value == null || value.Type == JTokenType.Null)
                return "";

            if (value.Type != JTokenType.String)
            {
                if (value.Type != JTokenType.Array && value.Type != JTokenType.Object)
                    System.Diagnostics.Trace.WriteLine("AccurateRip Meta ignored non-string " + fieldName + " value.");
                return "";
            }

            return Clean(value);
        }

        private static List<JObject> OrderTracksByTrackNumber(JArray tracks, int firstAudioTrackNumber, int audioTracks)
        {
            List<JObject> orderedTracks = OrderTracksByNumberRange(tracks, firstAudioTrackNumber, audioTracks);
            if (orderedTracks != null || firstAudioTrackNumber == 1)
                return orderedTracks;

            // Some providers normalize mixed-mode discs to audio-relative track numbers.
            return OrderTracksByNumberRange(tracks, 1, audioTracks);
        }

        private static List<JObject> OrderTracksByNumberRange(JArray tracks, int firstTrackNumber, int audioTracks)
        {
            var byNumber = new Dictionary<int, JObject>();
            foreach (JToken item in tracks)
            {
                JObject track = item as JObject;
                if (track == null)
                    continue;

                int trackNumber;
                if (!int.TryParse(Clean(track["TrackNumber"]), NumberStyles.Integer, CultureInfo.InvariantCulture, out trackNumber))
                    continue;

                if (trackNumber < firstTrackNumber || trackNumber >= firstTrackNumber + audioTracks)
                    continue;

                int audioIndex = trackNumber - firstTrackNumber;
                if (byNumber.ContainsKey(audioIndex))
                    return null;

                byNumber.Add(audioIndex, track);
            }

            if (byNumber.Count != audioTracks)
                return null;

            return Enumerable.Range(0, audioTracks)
                .Select(audioIndex => byNumber[audioIndex])
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

            if (!string.Equals(uri.Host, CoverArtHost, StringComparison.OrdinalIgnoreCase))
            {
                System.Diagnostics.Trace.WriteLine("AccurateRip Meta ignored cover art URL from unexpected host: " + uri.Host);
                return "";
            }

            return uri.AbsoluteUri;
        }

        private static string NormalizeGenreList(JToken value)
        {
            string text = Clean(value);
            if (string.IsNullOrWhiteSpace(text))
                return "";

            return string.Join("; ", text
                .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(line => line.Trim())
                .Where(line => line.Length > 0)
                .Distinct()
                .ToArray());
        }

        private static string NormalizeProviderList(JToken value)
        {
            string text = Clean(value);
            if (string.IsNullOrWhiteSpace(text))
                return "";

            string[] names = text
                .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(line => line.Trim())
                .Where(line => line.Length > 0)
                .Distinct()
                .ToArray();

            if (names.Length == 0)
                return "";
            if (names.Length == 1)
                return names[0];
            if (names.Length == 2)
                return names[0] + " & " + names[1];

            return string.Join(", ", names.Take(names.Length - 1).ToArray()) + " & " + names[names.Length - 1];
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
            return Lookup(toc, proxy, null);
        }

        public CUEMetadataEntry Lookup(CDImageLayout toc, IWebProxy proxy, Action checkStop)
        {
            if (toc == null)
                return null;

            try
            {
                string accurateRipDiscId = AccurateRipMeta.CreateRequestDiscId(toc);
                string body = AccurateRipMeta.CreateRequestBody(accurateRipDiscId);
                string json = transport.Post(AccurateRipMeta.Endpoint, body, proxy, checkStop);

                return AccurateRipMeta.TryParseMetadata(json, toc, out CUEMetadataEntry entry) ? entry : null;
            }
            catch (Exception ex) when (IsCancellation(ex))
            {
                throw;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.WriteLine("AccurateRip Meta lookup failed: " + ex.Message);
                return null;
            }
        }

        private static bool IsCancellation(Exception ex)
        {
            return ex is StopException || ex is OperationCanceledException;
        }
    }

    public interface IAccurateRipMetaTransport
    {
        string Post(string url, string body, IWebProxy proxy, Action checkStop);
    }

    public sealed class AccurateRipMetaHttpTransport : IAccurateRipMetaTransport
    {
        public string Post(string url, string body, IWebProxy proxy, Action checkStop)
        {
            var request = (HttpWebRequest)WebRequest.Create(url);
            int[] stopped = new[] { 0 };
            int[] stopWatcherFinished = new[] { 0 };
            Thread stopWatcher = null;
            try
            {
                stopWatcher = StartStopWatcher(request, checkStop, stopped, stopWatcherFinished);
                request.Method = "POST";
                // The undocumented endpoint matched the EasyEACGUI client only with this browser-style UA during contract review.
                request.UserAgent = "Mozilla/5.0";
                // The server expects the EasyEACGUI-compatible JSON body with this form content type.
                request.ContentType = AccurateRipMeta.ContentType;
                request.AllowAutoRedirect = false;
                request.ServicePoint.Expect100Continue = false;
                request.AutomaticDecompression = DecompressionMethods.GZip | DecompressionMethods.Deflate;
                request.Timeout = AccurateRipMeta.LookupTimeoutMilliseconds;
                request.ReadWriteTimeout = AccurateRipMeta.ReadWriteTimeoutMilliseconds;
                request.Proxy = proxy;

                byte[] bytes = Encoding.UTF8.GetBytes(body);
                request.ContentLength = bytes.Length;
                using (Stream requestStream = request.GetRequestStream())
                    requestStream.Write(bytes, 0, bytes.Length);

                using (var response = (HttpWebResponse)request.GetResponse())
                using (Stream responseStream = response.GetResponseStream())
                    return ReadResponseBody(responseStream);
            }
            catch (WebException ex)
            {
                var response = ex.Response as HttpWebResponse;
                if (response != null && (int)response.StatusCode >= 300 && (int)response.StatusCode < 400)
                    System.Diagnostics.Trace.WriteLine("AccurateRip Meta moved; update CUETools (got " + response.StatusCode + " -> " + response.Headers["Location"] + ")");
                if (response != null)
                    response.Dispose();
                request.Abort();
                if (Volatile.Read(ref stopped[0]) != 0)
                    throw new StopException();
                throw;
            }
            catch
            {
                request.Abort();
                if (Volatile.Read(ref stopped[0]) != 0)
                    throw new StopException();
                throw;
            }
            finally
            {
                Volatile.Write(ref stopWatcherFinished[0], 1);
                if (stopWatcher != null)
                    stopWatcher.Join(500);
            }
        }

        private static Thread StartStopWatcher(HttpWebRequest request, Action checkStop, int[] stopped, int[] finished)
        {
            if (checkStop == null)
                return null;

            var stopWatcher = new Thread(() =>
            {
                while (Volatile.Read(ref finished[0]) == 0)
                {
                    try
                    {
                        checkStop();
                    }
                    catch (Exception ex) when (IsCancellation(ex))
                    {
                        Interlocked.Exchange(ref stopped[0], 1);
                        request.Abort();
                        return;
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.WriteLine("AccurateRip Meta stop watcher failed: " + ex.Message);
                        return;
                    }

                    for (int i = 0; i < 10 && Volatile.Read(ref finished[0]) == 0; i++)
                        Thread.Sleep(10);
                }
            });
            stopWatcher.IsBackground = true;
            stopWatcher.Name = "AccurateRip Meta stop watcher";
            stopWatcher.Start();
            return stopWatcher;
        }

        private static string ReadResponseBody(Stream responseStream)
        {
            if (responseStream == null)
                return "";

            const int maxResponseBytes = 1024 * 1024;
            byte[] buffer = new byte[8192];
            using (var memory = new MemoryStream())
            {
                int read;
                while ((read = responseStream.Read(buffer, 0, buffer.Length)) > 0)
                {
                    if (memory.Length + read > maxResponseBytes)
                        throw new InvalidDataException("AccurateRip Meta response exceeded the maximum supported size.");
                    memory.Write(buffer, 0, read);
                }

                return Encoding.UTF8.GetString(memory.ToArray());
            }
        }

        private static bool IsCancellation(Exception ex)
        {
            return ex is StopException || ex is OperationCanceledException;
        }
    }

    internal sealed class AccurateRipMetaRequest
    {
        [JsonProperty("accurateripdiscid")]
        public string AccurateRipDiscId { get; set; }
    }
}
