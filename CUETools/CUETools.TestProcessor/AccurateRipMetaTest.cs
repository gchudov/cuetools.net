using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Net;
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
            string json = SampleJson(trackCount: 2).Replace("\"TrackNumber\":\"2\",", "");

            bool parsed = AccurateRipMeta.TryParseMetadata(json, CreateTwoTrackToc(), out CUEMetadataEntry entry);

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
        public void TryParseMetadataRejectsInvalidCoverArtUrl()
        {
            string json = SampleJson(trackCount: 2).Replace("http://covers.example/front.jpg", "not a url");

            bool parsed = AccurateRipMeta.TryParseMetadata(json, CreateTwoTrackToc(), out CUEMetadataEntry entry);

            Assert.IsTrue(parsed);
            Assert.AreEqual(0, entry.metadata.AlbumArt.Count);
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
    }
}
