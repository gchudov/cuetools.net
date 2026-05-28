using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;
using CUETools.AccurateRip;
using CUETools.CDImage;
using CUETools.CTDB;
using CUETools.Processor;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace CUETools.TestProcessor
{
    [TestClass]
    public class AccurateRipMetaTest
    {
        [TestMethod]
        public void TryParseMetadataMapsRecordedAmarokFixture()
        {
            string json = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Fixtures", "accuraterip-meta-amarok.json"));

            bool parsed = AccurateRipMeta.TryParseMetadata(json, CreateAmarokToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.IsNotNull(entry);
            Assert.AreEqual(AccurateRipMeta.SourceKey, entry.ImageKey);
            Assert.AreEqual("Mike Oldfield", entry.metadata.Artist);
            Assert.AreEqual("Amarok", entry.metadata.Title);
            Assert.AreEqual("1990", entry.metadata.Year);
            Assert.AreEqual("Parlata", entry.metadata.Genre);
            Assert.AreEqual("Virgin", entry.metadata.Label);
            Assert.AreEqual("CDV 2640", entry.metadata.LabelNo);
            Assert.AreEqual("5012981264024", entry.metadata.Barcode);
            Assert.AreEqual(1, entry.metadata.Tracks.Count);
            Assert.AreEqual("Amarok", entry.metadata.Tracks[0].Title);
            Assert.AreEqual("Mike Oldfield", entry.metadata.Tracks[0].Artist);
            Assert.AreEqual("GBAAA0000417", entry.metadata.Tracks[0].ISRC);
            Assert.AreEqual(1, entry.metadata.AlbumArt.Count);
            Assert.AreEqual("http://meta.accuraterip.com/albumart/0000001F89DA34F26C1A", entry.metadata.AlbumArt[0].uri);
        }

        [TestMethod]
        public void TryParseMetadataMapsKnownFields()
        {
            bool parsed = AccurateRipMeta.TryParseMetadata(SampleJson(trackCount: 2), CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.IsNotNull(entry);
            Assert.AreEqual(AccurateRipMeta.SourceKey, entry.ImageKey);
            Assert.AreEqual("The Artist & Guest", entry.metadata.Artist);
            Assert.AreEqual("The Album", entry.metadata.Title);
            Assert.AreEqual("1999", entry.metadata.Year);
            Assert.AreEqual("Rock", entry.metadata.Genre);
            Assert.AreEqual("0123456789012", entry.metadata.Barcode);
            Assert.AreEqual("The Label", entry.metadata.Label);
            Assert.AreEqual("CAT-123", entry.metadata.LabelNo);
            Assert.AreEqual("", entry.metadata.Comment);
            Assert.AreEqual(2, entry.metadata.Tracks.Count);
            Assert.AreEqual("Track One", entry.metadata.Tracks[0].Title);
            Assert.AreEqual("Track Artist 1 & Other", entry.metadata.Tracks[0].Artist);
            Assert.AreEqual("USAAA0100001", entry.metadata.Tracks[0].ISRC);
            Assert.AreEqual("", entry.metadata.Tracks[0].Comment);
            Assert.AreEqual("Track Two", entry.metadata.Tracks[1].Title);
            Assert.AreEqual("Track Artist 2", entry.metadata.Tracks[1].Artist);
            Assert.AreEqual("USAAA0100002", entry.metadata.Tracks[1].ISRC);
            Assert.AreEqual(1, entry.metadata.AlbumArt.Count);
            Assert.AreEqual("http://covers.example/front.jpg", entry.metadata.AlbumArt[0].uri);
            Assert.AreEqual("http://covers.example/front.jpg", entry.metadata.AlbumArt[0].uri150);
            Assert.IsTrue(entry.metadata.AlbumArt[0].primary);
        }

        [TestMethod]
        public void TryParseMetadataIgnoresFieldsWithoutStorage()
        {
            bool parsed = AccurateRipMeta.TryParseMetadata(SampleJson(trackCount: 2), CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.AreEqual("Rock", entry.metadata.Genre);
            Assert.IsFalse(entry.metadata.Genre.Contains("Fusion"));
            Assert.AreEqual("", entry.metadata.Comment);
            Assert.IsFalse(entry.metadata.Comment.Contains("Fusion"));
            Assert.IsFalse(entry.metadata.Tracks.Any(track => track.Comment.Contains("Composer")));
            Assert.IsFalse(entry.metadata.Tracks.Any(track => track.Comment.Contains("Conductor")));
        }

        [TestMethod]
        public void TryParseMetadataOrdersTracksByTrackNumber()
        {
            JObject json = JObject.Parse(SampleJson(trackCount: 2));
            JArray tracks = (JArray)json["tracks"];
            json["tracks"] = new JArray(tracks[1], tracks[0]);

            bool parsed = AccurateRipMeta.TryParseMetadata(json.ToString(), CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.AreEqual("Track One", entry.metadata.Tracks[0].Title);
            Assert.AreEqual("Track Two", entry.metadata.Tracks[1].Title);
            Assert.AreEqual("USAAA0100001", entry.metadata.Tracks[0].ISRC);
            Assert.AreEqual("USAAA0100002", entry.metadata.Tracks[1].ISRC);
        }

        [TestMethod]
        public void TryParseMetadataRejectsMissingTrackNumber()
        {
            JObject json = JObject.Parse(SampleJson(trackCount: 2));
            JArray tracks = (JArray)json["tracks"];
            tracks[1]["TrackNumber"].Parent.Remove();

            bool parsed = AccurateRipMeta.TryParseMetadata(json.ToString(), CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsFalse(parsed);
            Assert.IsNull(entry);
        }

        [TestMethod]
        public void TryParseMetadataRejectsTrackCountMismatch()
        {
            bool parsed = AccurateRipMeta.TryParseMetadata(SampleJson(trackCount: 1), CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsFalse(parsed);
            Assert.IsNull(entry);
        }

        [TestMethod]
        public void TryParseMetadataRejectsNonArrayTracksShape()
        {
            JObject json = JObject.Parse(SampleJson(trackCount: 2));
            json["tracks"] = new JObject { ["unexpected"] = "shape" };

            bool parsed = AccurateRipMeta.TryParseMetadata(json.ToString(), CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsFalse(parsed);
            Assert.IsNull(entry);
        }

        [TestMethod]
        public void TryParseMetadataAllowsMissingCoverArt()
        {
            string json = SampleJson(trackCount: 2).Replace("\"_arturl\":\"http://covers.example/front.jpg\",", "");

            bool parsed = AccurateRipMeta.TryParseMetadata(json, CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.AreEqual(0, entry.metadata.AlbumArt.Count);
        }

        [TestMethod]
        public void TryParseMetadataAllowsHttpsCoverArtUrl()
        {
            string coverArtUrl = "https://covers.example/front.jpg";
            string json = SampleJsonWithCoverArtUrl(coverArtUrl);

            bool parsed = AccurateRipMeta.TryParseMetadata(json, CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.AreEqual(1, entry.metadata.AlbumArt.Count);
            Assert.AreEqual(coverArtUrl, entry.metadata.AlbumArt[0].uri);
            Assert.AreEqual(coverArtUrl, entry.metadata.AlbumArt[0].uri150);
        }

        [DataTestMethod]
        [DataRow("not a url")]
        [DataRow("file:///C:/covers/front.jpg")]
        [DataRow("ftp://covers.example/front.jpg")]
        [DataRow("//covers.example/front.jpg")]
        [DataRow("/covers/front.jpg")]
        public void TryParseMetadataRejectsInvalidCoverArtUrl(string coverArtUrl)
        {
            string json = SampleJsonWithCoverArtUrl(coverArtUrl);

            bool parsed = AccurateRipMeta.TryParseMetadata(json, CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.AreEqual(0, entry.metadata.AlbumArt.Count);
        }

        [DataTestMethod]
        [DataRow("2", "2")]
        [DataRow("abc", "2")]
        [DataRow("0", "2")]
        [DataRow("1", "3")]
        public void TryParseMetadataRejectsInvalidTrackNumberSet(string firstTrackNumber, string secondTrackNumber)
        {
            string json = SampleJsonWithTrackNumbers(firstTrackNumber, secondTrackNumber);

            bool parsed = AccurateRipMeta.TryParseMetadata(json, CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsFalse(parsed);
            Assert.IsNull(entry);
        }

        [TestMethod]
        public void TryParseMetadataAllowsEmptyOptionalFields()
        {
            string json = SampleJson(trackCount: 2)
                .Replace("\"Year\":\"1999\",", "\"Year\":\"\",")
                .Replace("\"Genre\":\"Rock\\rAlternative\",", "\"Genre\":\"\",")
                .Replace("\"Label\":\"The Label\",", "\"Label\":\"\",")
                .Replace("\"CatalogNum\":\"CAT-123\",", "\"CatalogNum\":\"\",")
                .Replace("\"UPC\":\"0123456789012\",", "\"UPC\":\"\",");

            bool parsed = AccurateRipMeta.TryParseMetadata(json, CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.AreEqual("", entry.metadata.Year);
            Assert.AreEqual("", entry.metadata.Genre);
            Assert.AreEqual("", entry.metadata.Label);
            Assert.AreEqual("", entry.metadata.LabelNo);
            Assert.AreEqual("", entry.metadata.Barcode);
        }

        [TestMethod]
        public void TryParseMetadataAllowsExtraProviderTracks()
        {
            bool parsed = AccurateRipMeta.TryParseMetadata(SampleJson(trackCount: 3), CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.AreEqual(2, entry.metadata.Tracks.Count);
            Assert.AreEqual("Track One", entry.metadata.Tracks[0].Title);
            Assert.AreEqual("Track Two", entry.metadata.Tracks[1].Title);
        }

        [TestMethod]
        public void TryParseMetadataAllowsUnorderedExtraProviderTracks()
        {
            JObject json = JObject.Parse(SampleJson(trackCount: 3));
            JArray tracks = (JArray)json["tracks"];
            json["tracks"] = new JArray(tracks[2], tracks[0], tracks[1]);

            bool parsed = AccurateRipMeta.TryParseMetadata(json.ToString(), CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.AreEqual(2, entry.metadata.Tracks.Count);
            Assert.AreEqual("Track One", entry.metadata.Tracks[0].Title);
            Assert.AreEqual("Track Two", entry.metadata.Tracks[1].Title);
        }

        [TestMethod]
        public void TryParseMetadataAllowsSurplusTrackWithoutTrackNumber()
        {
            JObject json = JObject.Parse(SampleJson(trackCount: 2));
            JArray tracks = (JArray)json["tracks"];
            tracks.Add(new JObject
            {
                ["Title"] = "Malformed Provider Extra Track",
                ["Artist"] = "Ignored Extra Artist"
            });

            bool parsed = AccurateRipMeta.TryParseMetadata(json.ToString(), CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.AreEqual(2, entry.metadata.Tracks.Count);
            Assert.AreEqual("Track One", entry.metadata.Tracks[0].Title);
            Assert.AreEqual("Track Two", entry.metadata.Tracks[1].Title);
        }

        [TestMethod]
        public void TryParseMetadataAllowsSurplusNonObjectTrack()
        {
            JObject json = JObject.Parse(SampleJson(trackCount: 2));
            JArray tracks = (JArray)json["tracks"];
            tracks.Add("not a track object");

            bool parsed = AccurateRipMeta.TryParseMetadata(json.ToString(), CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.AreEqual(2, entry.metadata.Tracks.Count);
            Assert.AreEqual("Track One", entry.metadata.Tracks[0].Title);
            Assert.AreEqual("Track Two", entry.metadata.Tracks[1].Title);
        }

        [TestMethod]
        public void TryParseMetadataAllowsOddScalarShapesWithoutDroppingMetadata()
        {
            string json = SampleJson(trackCount: 2)
                .Replace("\"Year\":\"1999\"", "\"Year\":1999")
                .Replace("\"UPC\":\"0123456789012\"", "\"UPC\":123456789012");

            bool parsed = AccurateRipMeta.TryParseMetadata(json, CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.AreEqual("1999", entry.metadata.Year);
            Assert.AreEqual("123456789012", entry.metadata.Barcode);
        }

        [TestMethod]
        public void TryParseMetadataIgnoresOddCoverArtShapeWithoutDroppingMetadata()
        {
            string json = SampleJson(trackCount: 2).Replace("\"_arturl\":\"http://covers.example/front.jpg\"", "\"_arturl\":[\"http://covers.example/front.jpg\"]");

            bool parsed = AccurateRipMeta.TryParseMetadata(json, CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.AreEqual("The Album", entry.metadata.Title);
            Assert.AreEqual(0, entry.metadata.AlbumArt.Count);
        }

        [TestMethod]
        public void LookupPostsSerializedAccurateRipIdAndUsesProxy()
        {
            var toc = CreateTwoTrackToc();
            var proxy = new WebProxy("127.0.0.1", 8888);
            var transport = new FakeTransport { Response = SampleJson(trackCount: 2) };
            var provider = new AccurateRipMetaProvider(transport);

            CUEMetadataEntry entry = provider.Lookup(toc, proxy);

            Assert.IsNotNull(entry);
            Assert.AreEqual(1, transport.PostCount);
            Assert.AreEqual(AccurateRipMeta.Endpoint, transport.Url);
            Assert.AreSame(proxy, transport.Proxy);
            var body = JsonConvert.DeserializeObject<Dictionary<string, string>>(transport.Body);
            Assert.AreEqual(string.Format("{0:000}-{1}", toc.AudioTracks, AccurateRipVerify.CalculateAccurateRipId(toc)), body["accurateripdiscid"]);
            Assert.AreEqual(AccurateRipMeta.SourceKey, entry.ImageKey);
            Assert.AreEqual("The Album", entry.metadata.Title);
        }

        [TestMethod]
        public async Task HttpTransportPostsLiveRequestShapeToLoopbackServer()
        {
            const string requestBody = "{\"accurateripdiscid\":\"001-test\"}";
            const string responseBody = "{}";

            using (var listener = new TcpListener(IPAddress.Loopback, 0))
            {
                listener.Start();
                int port = ((IPEndPoint)listener.LocalEndpoint).Port;
                Task<CapturedHttpRequest> serverTask = ReadSingleRequestAsync(listener, responseBody);

                string response = new AccurateRipMetaHttpTransport().Post("http://127.0.0.1:" + port + "/", requestBody, null);
                CapturedHttpRequest request = await serverTask;

                Assert.AreEqual(responseBody, response);
                Assert.AreEqual("POST", request.Method);
                Assert.AreEqual("Mozilla/5.0", request.UserAgent);
                StringAssert.StartsWith(request.ContentType, "application/x-www-form-urlencoded");
                Assert.AreEqual(requestBody, request.Body);
            }
        }

        [TestMethod]
        public void LookupReturnsNullWhenTransportThrows()
        {
            var transport = new FakeTransport { Exception = new WebException("network unavailable") };
            var provider = new AccurateRipMetaProvider(transport);

            CUEMetadataEntry entry = provider.Lookup(CreateTwoTrackToc(), null);

            Assert.IsNull(entry);
            Assert.AreEqual(1, transport.PostCount);
        }

        [TestMethod]
        public void LookupReturnsNullForInvalidJson()
        {
            var transport = new FakeTransport { Response = "{not-json" };
            var provider = new AccurateRipMetaProvider(transport);

            CUEMetadataEntry entry = provider.Lookup(CreateTwoTrackToc(), null);

            Assert.IsNull(entry);
            Assert.AreEqual(1, transport.PostCount);
        }

        [TestMethod]
        public void LookupReturnsNullForNullTocWithoutCallingTransport()
        {
            var transport = new FakeTransport { Response = SampleJson(trackCount: 2) };
            var provider = new AccurateRipMetaProvider(transport);

            CUEMetadataEntry entry = provider.Lookup(null, new WebProxy("127.0.0.1", 8888));

            Assert.IsNull(entry);
            Assert.AreEqual(0, transport.PostCount);
        }

        [TestMethod]
        public void LookupReturnsNullForInvalidMappedJson()
        {
            var transport = new FakeTransport { Response = SampleJson(trackCount: 1) };
            var provider = new AccurateRipMetaProvider(transport);

            CUEMetadataEntry entry = provider.Lookup(CreateTwoTrackToc(), null);

            Assert.IsNull(entry);
            Assert.AreEqual(1, transport.PostCount);
        }

        private static CDImageLayout CreateAmarokToc()
        {
            var sheet = new CUESheet(new CUEConfig());
            sheet.Open("Amarok\\Amarok.cue");
            return sheet.TOC;
        }

        private static CDImageLayout CreateTwoTrackToc()
        {
            var toc = new CDImageLayout();
            toc.AddTrack(new CDTrack(1, 0, 15000, true, false));
            toc.AddTrack(new CDTrack(2, 15000, 18000, true, false));
            return toc;
        }

        private static string SampleJsonWithCoverArtUrl(string coverArtUrl)
        {
            JObject json = JObject.Parse(SampleJson(trackCount: 2));
            json["_arturl"] = coverArtUrl;
            return json.ToString();
        }

        private static string SampleJsonWithTrackNumbers(params string[] trackNumbers)
        {
            JObject json = JObject.Parse(SampleJson(trackNumbers.Length));
            JArray tracks = (JArray)json["tracks"];

            for (int i = 0; i < trackNumbers.Length; i++)
            {
                tracks[i]["TrackNumber"] = trackNumbers[i];
            }

            return json.ToString();
        }

        private static string SampleJson(int trackCount)
        {
            var tracks = new List<object>
            {
                new
                {
                    TrackNumber = "1",
                    Title = "Track One",
                    Artist = "Track Artist 1\rOther",
                    Composers = "Ignored Composer",
                    Conductors = "Ignored Conductor",
                    ISRC = "USAAA0100001"
                }
            };

            if (trackCount > 1)
            {
                tracks.Add(new
                {
                    TrackNumber = "2",
                    Title = "Track Two",
                    Artist = "Track Artist 2",
                    Composers = "Second Composer",
                    Conductors = "Second Conductor",
                    ISRC = "USAAA0100002"
                });
            }

            if (trackCount > 2)
            {
                tracks.Add(new
                {
                    TrackNumber = "3",
                    Title = "Provider Extra Track",
                    Artist = "Ignored Extra Artist",
                    Composers = "Ignored Extra Composer",
                    Conductors = "Ignored Extra Conductor",
                    ISRC = "USAAA0100003"
                });
            }

            return JsonConvert.SerializeObject(new
            {
                _tocid = "toc-id",
                Artist = "The Artist\rGuest",
                Album = "The Album",
                Year = "1999",
                Genre = "Rock\rAlternative",
                Label = "The Label",
                Styles = "Fusion",
                CatalogNum = "CAT-123",
                UPC = "0123456789012",
                _albumid = "album-id",
                _arturl = "http://covers.example/front.jpg",
                tracks = tracks
            });
        }

        private static async Task<CapturedHttpRequest> ReadSingleRequestAsync(TcpListener listener, string responseBody)
        {
            using (TcpClient client = await listener.AcceptTcpClientAsync())
            using (NetworkStream stream = client.GetStream())
            {
                byte[] rawRequest = await ReadRequestBytesThroughHeadersAsync(stream);
                int headerEnd = FindHeaderEnd(rawRequest, rawRequest.Length);
                string headerText = Encoding.ASCII.GetString(rawRequest, 0, headerEnd);
                string[] headerLines = headerText.Split(new[] { "\r\n" }, StringSplitOptions.None);

                var captured = new CapturedHttpRequest();
                captured.Method = headerLines[0].Split(' ')[0];

                int contentLength = 0;
                bool expectsContinue = false;
                for (int i = 1; i < headerLines.Length; i++)
                {
                    int separator = headerLines[i].IndexOf(':');
                    if (separator < 0)
                        continue;

                    string name = headerLines[i].Substring(0, separator);
                    string value = headerLines[i].Substring(separator + 1).Trim();
                    if (string.Equals(name, "User-Agent", StringComparison.OrdinalIgnoreCase))
                        captured.UserAgent = value;
                    else if (string.Equals(name, "Content-Type", StringComparison.OrdinalIgnoreCase))
                        captured.ContentType = value;
                    else if (string.Equals(name, "Content-Length", StringComparison.OrdinalIgnoreCase))
                        int.TryParse(value, out contentLength);
                    else if (string.Equals(name, "Expect", StringComparison.OrdinalIgnoreCase) && value.IndexOf("100-continue", StringComparison.OrdinalIgnoreCase) >= 0)
                        expectsContinue = true;
                }

                if (expectsContinue)
                {
                    byte[] continueBytes = Encoding.ASCII.GetBytes("HTTP/1.1 100 Continue\r\n\r\n");
                    await stream.WriteAsync(continueBytes, 0, continueBytes.Length);
                }

                int bodyStart = headerEnd + 4;
                byte[] bodyBytes = new byte[contentLength];
                int bufferedBodyBytes = Math.Min(contentLength, rawRequest.Length - bodyStart);
                if (bufferedBodyBytes > 0)
                    Buffer.BlockCopy(rawRequest, bodyStart, bodyBytes, 0, bufferedBodyBytes);

                int offset = bufferedBodyBytes;
                while (offset < contentLength)
                {
                    int read = await stream.ReadAsync(bodyBytes, offset, contentLength - offset);
                    if (read == 0)
                        break;

                    offset += read;
                }

                captured.Body = Encoding.UTF8.GetString(bodyBytes, 0, offset);

                byte[] responseBodyBytes = Encoding.UTF8.GetBytes(responseBody);
                string responseHeaders = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " + responseBodyBytes.Length + "\r\nConnection: close\r\n\r\n";
                byte[] responseHeaderBytes = Encoding.ASCII.GetBytes(responseHeaders);
                await stream.WriteAsync(responseHeaderBytes, 0, responseHeaderBytes.Length);
                await stream.WriteAsync(responseBodyBytes, 0, responseBodyBytes.Length);

                return captured;
            }
        }

        private static async Task<byte[]> ReadRequestBytesThroughHeadersAsync(NetworkStream stream)
        {
            var request = new MemoryStream();
            byte[] buffer = new byte[1024];
            while (true)
            {
                int read = await stream.ReadAsync(buffer, 0, buffer.Length);
                if (read == 0)
                    break;

                request.Write(buffer, 0, read);
                byte[] bytes = request.ToArray();
                if (FindHeaderEnd(bytes, bytes.Length) >= 0)
                    return bytes;
            }

            return request.ToArray();
        }

        private static int FindHeaderEnd(byte[] bytes, int length)
        {
            for (int i = 0; i <= length - 4; i++)
            {
                if (bytes[i] == '\r' && bytes[i + 1] == '\n' && bytes[i + 2] == '\r' && bytes[i + 3] == '\n')
                    return i;
            }

            return -1;
        }

        [TestMethod]
        public void LookupAlbumInfoExtensiveAddsAccurateRipMeta()
        {
            var transport = new FakeTransport { Response = SampleJson(trackCount: 1) };
            var sheet = new CUESheet(new CUEConfig())
            {
                AccurateRipMetaProvider = new AccurateRipMetaProvider(transport)
            };
            sheet.Open("Amarok\\Amarok.cue");

            var releases = sheet.LookupAlbumInfo(false, false, false, CUETools.CTDB.CTDBMetadataSearch.Extensive)
                .OfType<CUEMetadataEntry>()
                .ToList();

            Assert.AreEqual(1, releases.Count(entry => entry.ImageKey == AccurateRipMeta.SourceKey));
        }

        [TestMethod]
        public void LookupAlbumInfoExtensiveAddsAccurateRipMetaWhenCtdbReturnedMetadata()
        {
            var transport = new FakeTransport { Response = SampleJson(trackCount: 1) };
            var sheet = new CtdbStubCUESheet(new CUEConfig(), CreateCtdbMetadata())
            {
                AccurateRipMetaProvider = new AccurateRipMetaProvider(transport)
            };
            sheet.Open("Amarok\\Amarok.cue");

            var releases = sheet.LookupAlbumInfo(false, false, true, CTDBMetadataSearch.Extensive)
                .OfType<CUEMetadataEntry>()
                .ToList();

            Assert.AreEqual(1, releases.Count(entry => entry.ImageKey == "ctdb"));
            Assert.AreEqual(1, releases.Count(entry => entry.ImageKey == AccurateRipMeta.SourceKey));
            Assert.AreEqual(1, transport.PostCount);
        }

        [TestMethod]
        public void LookupAlbumInfoDefaultDoesNotQueryAccurateRipMeta()
        {
            var transport = new FakeTransport { Response = SampleJson(trackCount: 1) };
            var sheet = new CUESheet(new CUEConfig())
            {
                AccurateRipMetaProvider = new AccurateRipMetaProvider(transport)
            };
            sheet.Open("Amarok\\Amarok.cue");

            var releases = sheet.LookupAlbumInfo(false, false, false, CUETools.CTDB.CTDBMetadataSearch.Default)
                .OfType<CUEMetadataEntry>()
                .ToList();

            Assert.AreEqual(0, releases.Count(entry => entry.ImageKey == AccurateRipMeta.SourceKey));
            Assert.IsNull(transport.Url);
        }

        private sealed class CtdbStubCUESheet : CUESheet
        {
            private readonly IEnumerable<CTDBResponseMeta> ctdbMetadata;

            public CtdbStubCUESheet(CUEConfig config, IEnumerable<CTDBResponseMeta> ctdbMetadata)
                : base(config)
            {
                this.ctdbMetadata = ctdbMetadata;
            }

            protected override IEnumerable<CTDBResponseMeta> LookupCtdbMetadata(CTDBMetadataSearch metadataSearch)
            {
                return ctdbMetadata;
            }
        }

        private static IEnumerable<CTDBResponseMeta> CreateCtdbMetadata()
        {
            return new[]
            {
                new CTDBResponseMeta
                {
                    source = "ctdb",
                    artist = "CTDB Artist",
                    album = "CTDB Album",
                    track = new[]
                    {
                        new CTDBResponseMetaTrack
                        {
                            name = "CTDB Track",
                            artist = "CTDB Artist"
                        }
                    }
                }
            };
        }

        private sealed class CapturedHttpRequest
        {
            public string Method { get; set; }
            public string UserAgent { get; set; }
            public string ContentType { get; set; }
            public string Body { get; set; }
        }

        private sealed class FakeTransport : IAccurateRipMetaTransport
        {
            public string Url { get; private set; }
            public string Body { get; private set; }
            public IWebProxy Proxy { get; private set; }
            public int PostCount { get; private set; }
            public string Response { get; set; }
            public Exception Exception { get; set; }

            public string Post(string url, string body, IWebProxy proxy)
            {
                PostCount++;
                Url = url;
                Body = body;
                Proxy = proxy;

                if (Exception != null)
                    throw Exception;

                return Response;
            }
        }
    }
}
