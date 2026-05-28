# AccurateRip Meta Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace active FreeDB metadata lookup with AccurateRip Meta while keeping the old FreeDB code dormant and unreachable from normal UI.

**Architecture:** Add a focused AccurateRip Meta provider in `CUETools.Processor` with a testable JSON mapper and injectable HTTP transport. Ground the request and response contract in the local EasyEACGUI sample plus a recorded live Amarok fixture before writing the mapper; the live endpoint expects a prefixed id such as `001-00041f6d-00083ece-020e1201`, not the bare `AccurateRipVerify.CalculateAccurateRipId` value. The endpoint is HTTP and expects `Content-Type: application/x-www-form-urlencoded` with a JSON body; this deliberately mirrors the deployed EasyEACGUI contract and should not be changed to HTTPS or `application/json` without live verification. Treat returned metadata and artwork as ordinary unauthenticated metadata, matching the existing AccurateRip HTTP precedent, not as security-sensitive input. Integrate it into `CUESheet.LookupAlbumInfo` and CUERipper reload/fallback lookup, using the existing `CUEMetadata` and `AlbumArt` models while validating provider cover URLs before consumers fetch them. In CUETools, `CTDBMetadataSearch.Extensive` must query AccurateRip Meta even when CTDB already returned metadata. In CUERipper, preserve the current fallback/reload shape: query AccurateRip Meta when no release was found or when the user requests all metadata/reload, not on every ordinary rip where CTDB already populated the release list. Hide FreeDB submission/settings paths, intentionally retire FreeDB submit/repair actions because AccurateRip Meta has no submission API, and preserve old FreeDB code as fenced source, not as a flip-a-symbol reactivation path, because project references are removed from the normal solution.

**Tech Stack:** C#/.NET 10 Windows projects, `netstandard2.1` processor library, MSTest, `Newtonsoft.Json`, WinForms, existing CUETools metadata and CTDB cover-art models.

---

## File Structure

- Create `CUETools.Processor/AccurateRipMeta.cs`
  - Owns AccurateRip Meta constants, response contracts, mapper, provider, and HTTP transport.
  - Public surface: `AccurateRipMeta.SourceKey`, `AccurateRipMeta.DisplayName`, `AccurateRipMeta.TryParseMetadata(string json, CDImageLayout toc, out CUEMetadataEntry entry)`, `AccurateRipMetaProvider.Lookup(CDImageLayout toc, IWebProxy proxy)`.

- Create `CUETools/CUETools.TestProcessor/AccurateRipMetaTest.cs`
  - Tests recorded fixture mapping, JSON mapping, unused fields, track-number ordering, track-count/shape rejection, extra-track tolerance, invalid cover URL rejection, odd scalar/cover-art shapes, transport request body, proxy forwarding, fail-soft provider behavior, CTDB-succeeded Extensive behavior, and FreeDB settings dormancy.

- Create `CUETools/CUETools.TestProcessor/Fixtures/accuraterip-meta-amarok.json`
  - Recorded live response for `{"accurateripdiscid":"001-00041f6d-00083ece-020e1201"}` from `http://meta.accuraterip.com/discmatch`, captured while reviewing this plan.

- Modify `CUETools/CUETools.TestProcessor/CUETools.TestProcessor.csproj`
  - Include `AccurateRipMetaTest.cs`.
  - Remove the FreeDB project reference after FreeDB-dependent production code is isolated.

- Modify `CUETools/CUETools.TestCodecs/CUETools.TestCodecs.csproj`
  - Remove the FreeDB project reference so the active solution build no longer pulls in `Freedb/Freedb.csproj` through codec tests.

- Modify `CUETools.Processor/CUESheet.cs`
  - Remove active `using Freedb`.
  - Add an injectable `AccurateRipMetaProvider` property for tests.
  - Add a protected CTDB metadata lookup seam so tests can cover the CTDB-succeeded Extensive branch without network I/O.
  - Replace the active FreeDB fallback with AccurateRip Meta lookup during `CTDBMetadataSearch.Extensive`.
  - Keep old FreeDB lookup and `TocFromCDEntry` dormant under `#if DORMANT_FREEDB`.

- Modify `CUETools.Processor/CUEMetadata.cs`
  - Keep active metadata behavior unchanged.
  - Move `FillFromFreedb`, `FreedbToEncoding`, and `FreedbToVarious` under `#if DORMANT_FREEDB`.

- Modify `CUETools.Processor/CUEConfigAdvanced.cs`
  - Move `FreedbUser`, `FreedbDomain`, and `FreedbSiteAddress` under `#if DORMANT_FREEDB`.
  - This removes the normal CUETools settings exposure because `CUETools/frmSettings.cs` binds `propertyGrid1.SelectedObject` directly to `_config.advanced`.

- Modify `CUETools.Processor/CUETools.Processor.csproj`
  - Remove the FreeDB project reference once the active processor code no longer compiles against it.

- Modify `CUETools/frmChoice.cs`
  - Register the `accurateripmeta` image key with the existing AccurateRip icon at runtime.
  - Remove active FreeDB encoding duplicate insertion; keep it dormant under `#if DORMANT_FREEDB`.

- Modify `CUETools/CUETools.csproj`
  - Remove the FreeDB project reference once active UI code no longer compiles against it.

- Modify `CUERipper/frmCUERipper.cs`
  - Remove active `using Freedb`.
  - Register the `accurateripmeta` image key with the existing AccurateRip icon at runtime.
  - Replace active FreeDB lookup with AccurateRip Meta lookup during no-result and reload/all-metadata lookup.
  - Fetch `AlbumArt` entries for every release in `backgroundWorkerArtwork_DoWork` before the CTDB-only artwork scan so AccurateRip Meta cover URLs appear in rip-time preview/download even when the user selects a non-default release later.
  - Hide FreeDB submit and FreeDB repair buttons from normal runtime.
  - Keep old FreeDB lookup/submission/repair handlers dormant under `#if DORMANT_FREEDB`, with non-dormant event-handler stubs where designer wiring still requires a method.
  - Leave `frmFreedbSubmit` compiled but unreachable from normal UI; the dormant submit code is the only remaining constructor path. Removing the form/designer/resx files is a later cleanup after the dormant retention period.

- Modify `CUERipper/Options.cs`
  - Remove the normal `FreedbSiteAddress` property-grid entry.
  - Keep a dormant copy under `#if DORMANT_FREEDB`.

- Modify `CUERipper/CUERipper.csproj`
  - Remove the FreeDB project reference once active CUERipper code no longer compiles against it.

- Modify `CUETools.sln`
  - Remove `Freedb/Freedb.csproj` from the active solution so FreeDB is not part of the normal build.
  - Leave the `Freedb/` source directory intact for future adaptation. Defining `DORMANT_FREEDB` later will also require restoring the `Freedb` project references.

- Exclude from this plan: `CUETools.CTDB.EACPlugin`
  - It has its own CTDB metadata UI with `source == "freedb"` repair logic and a `freedb` image key, but no `Freedb.csproj` dependency. Keep it unchanged because the user scoped this replacement to CUETools/CUERipper active FreeDB provider behavior, not the EAC plugin's CTDB metadata display.

---

### Task 1: Record AccurateRip Meta Contract Fixture And Mapper Tests

**Files:**
- Create: `CUETools/CUETools.TestProcessor/AccurateRipMetaTest.cs`
- Create: `CUETools/CUETools.TestProcessor/Fixtures/accuraterip-meta-amarok.json`
- Modify: `CUETools/CUETools.TestProcessor/CUETools.TestProcessor.csproj`

- [ ] **Step 1: Record the verified Amarok response fixture**

Create `CUETools/CUETools.TestProcessor/Fixtures/accuraterip-meta-amarok.json`:

```json
{"_tocid":"858576","Artist":"Mike Oldfield","Album":"Amarok","Year":"1990","Genre":"Parlata","Label":"Virgin","Styles":"Progressive Electronic\rProg-Rock\rArt Rock","CatalogNum":"CDV 2640","UPC":"5012981264024","_albumid":"155298","_arturl":"http://meta.accuraterip.com/albumart/0000001F89DA34F26C1A","tracks":[{"TrackNumber":"1","Title":"Amarok","Artist":"Mike Oldfield","Composers":"","Conductors":"","ISRC":"GBAAA0000417"}]}
```

This fixture is the response returned by a live POST captured on 2026-05-28 with the local EasyEACGUI request shape from `H:\CProjekte\EasyEACGUI\MetadataProvider\AccurateRipMetadata.cs` and `H:\CProjekte\EasyEACGUI\MetadataProvider\BasicTools.cs`. `Label`, `CatalogNum`, `UPC`, and per-track `ISRC` values in the fixture are from that live response, not inferred from the EasyEACGUI consumer code.

```http
POST http://meta.accuraterip.com/discmatch
User-Agent: Mozilla/5.0
Content-Type: application/x-www-form-urlencoded

{"accurateripdiscid":"001-00041f6d-00083ece-020e1201"}
```

- [ ] **Step 2: Write the failing mapper tests**

Create `CUETools/CUETools.TestProcessor/AccurateRipMetaTest.cs`:

```csharp
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
```

Modify `CUETools/CUETools.TestProcessor/CUETools.TestProcessor.csproj` and add the compile include beside the existing test files:

```xml
  <ItemGroup>
    <Compile Include="AccurateRipMetaTest.cs" />
    <Compile Include="FileGroupInfoTest.cs" />
    <Compile Include="ProcessorTest.cs" />
    <Compile Include="Properties\AssemblyInfo.cs" />
  </ItemGroup>

  <ItemGroup>
    <None Update="Fixtures\accuraterip-meta-amarok.json" CopyToOutputDirectory="PreserveNewest" />
  </ItemGroup>
```

- [ ] **Step 3: Run tests to verify they fail for the expected reason**

Run:

```powershell
dotnet test CUETools\CUETools.TestProcessor\CUETools.TestProcessor.csproj --filter AccurateRipMetaTest -v:minimal
```

Expected: FAIL at compile time with errors that `AccurateRipMeta` does not exist.

- [ ] **Step 4: Commit the failing tests and recorded fixture**

```powershell
git add CUETools/CUETools.TestProcessor/AccurateRipMetaTest.cs CUETools/CUETools.TestProcessor/Fixtures/accuraterip-meta-amarok.json CUETools/CUETools.TestProcessor/CUETools.TestProcessor.csproj
git commit -m "test: cover AccurateRip Meta mapping"
```

---

### Task 2: Implement AccurateRip Meta Mapper

**Files:**
- Create: `CUETools.Processor/AccurateRipMeta.cs`
- Test: `CUETools/CUETools.TestProcessor/AccurateRipMetaTest.cs`

- [ ] **Step 1: Add the mapper implementation**

Create `CUETools.Processor/AccurateRipMeta.cs`:

```csharp
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
```

- [ ] **Step 2: Run mapper tests**

Run:

```powershell
dotnet test CUETools\CUETools.TestProcessor\CUETools.TestProcessor.csproj --filter "FullyQualifiedName~AccurateRipMetaTest.TryParseMetadata" -v:minimal
```

Expected: PASS for all ten mapper tests whose names start with `TryParseMetadata`.

- [ ] **Step 3: Commit the mapper**

```powershell
git add CUETools.Processor/AccurateRipMeta.cs
git commit -m "feat: map AccurateRip Meta responses"
```

---

### Task 3: Add Transport And Provider Tests

**Files:**
- Modify: `CUETools/CUETools.TestProcessor/AccurateRipMetaTest.cs`
- Modify: `CUETools.Processor/AccurateRipMeta.cs`

- [ ] **Step 1: Add failing provider tests**

Append these tests and helper class to `AccurateRipMetaTest` before the final closing brace of the class:

```csharp
        [TestMethod]
        public void LookupPostsSerializedAccurateRipIdAndUsesProxy()
        {
            var toc = CreateTwoTrackToc();
            var proxy = new WebProxy("127.0.0.1", 8888);
            var transport = new FakeTransport { Response = SampleJson(trackCount: 2) };
            var provider = new AccurateRipMetaProvider(transport);

            CUEMetadataEntry entry = provider.Lookup(toc, proxy);

            Assert.IsNotNull(entry);
            Assert.AreEqual(AccurateRipMeta.Endpoint, transport.Url);
            Assert.AreSame(proxy, transport.Proxy);
            var body = JsonConvert.DeserializeObject<Dictionary<string, string>>(transport.Body);
            Assert.AreEqual(string.Format("{0:000}-{1}", toc.AudioTracks, AccurateRipVerify.CalculateAccurateRipId(toc)), body["accurateripdiscid"]);
        }

        [TestMethod]
        public void LookupReturnsNullWhenTransportThrows()
        {
            var transport = new FakeTransport { Exception = new WebException("network unavailable") };
            var provider = new AccurateRipMetaProvider(transport);

            CUEMetadataEntry entry = provider.Lookup(CreateTwoTrackToc(), null);

            Assert.IsNull(entry);
        }

        [TestMethod]
        public void LookupReturnsNullForInvalidJson()
        {
            var transport = new FakeTransport { Response = "{not-json" };
            var provider = new AccurateRipMetaProvider(transport);

            CUEMetadataEntry entry = provider.Lookup(CreateTwoTrackToc(), null);

            Assert.IsNull(entry);
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
```

- [ ] **Step 2: Run tests to verify current implementation passes**

Run:

```powershell
dotnet test CUETools\CUETools.TestProcessor\CUETools.TestProcessor.csproj --filter AccurateRipMetaTest -v:minimal
```

Expected: PASS for all `AccurateRipMetaTest` tests. If this fails because Task 2 did not expose `IAccurateRipMetaTransport` publicly, make the interface public as shown in Task 2 and rerun.

- [ ] **Step 3: Commit provider tests**

```powershell
git add CUETools/CUETools.TestProcessor/AccurateRipMetaTest.cs CUETools.Processor/AccurateRipMeta.cs
git commit -m "test: cover AccurateRip Meta transport behavior"
```

---

### Task 4: Integrate AccurateRip Meta Into CUESheet Lookup

**Files:**
- Modify: `CUETools.Processor/CUESheet.cs`
- Modify: `CUETools/CUETools.TestProcessor/AccurateRipMetaTest.cs`

Behavior change to preserve: `CTDBMetadataSearch.Extensive` must query AccurateRip Meta even when CTDB has already returned metadata. The focused unit tests use a CTDB metadata seam to avoid network I/O while still proving the CTDB-succeeded branch.

- [ ] **Step 1: Add failing CUESheet lookup tests**

Append these tests to `AccurateRipMetaTest` before the `FakeTransport` class:

```csharp
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
```

- [ ] **Step 2: Run tests to verify they fail for the expected reason**

Run:

```powershell
dotnet test CUETools\CUETools.TestProcessor\CUETools.TestProcessor.csproj --filter "FullyQualifiedName~LookupAlbumInfo" -v:minimal
```

Expected: FAIL at compile time because `CUESheet.AccurateRipMetaProvider` does not exist, or FAIL at runtime because `LookupAlbumInfo` does not add AccurateRip Meta.

- [ ] **Step 3: Add provider property and CTDB metadata seam in `CUESheet`**

In `CUETools.Processor/CUESheet.cs`, replace the active `using Freedb;` line with a dormant-only using:

```csharp
#if DORMANT_FREEDB
using Freedb;
#endif
```

Add this property near the existing public properties:

```csharp
        public AccurateRipMetaProvider AccurateRipMetaProvider { get; set; }
```

In the `CUESheet(CUEConfig config)` constructor, add this assignment after `_config = config;`:

```csharp
            AccurateRipMetaProvider = new AccurateRipMetaProvider();
```

Add this protected helper near `LookupAlbumInfo` so tests can simulate CTDB metadata without network I/O:

```csharp
        protected virtual IEnumerable<CTDBResponseMeta> LookupCtdbMetadata(CTDBMetadataSearch metadataSearch)
        {
            var ctdb = new CUEToolsDB(TOC, proxy);
            ctdb.ContactDB(_config.advanced.CTDBServer, "CUETools " + CUEToolsVersion, null, false, false, metadataSearch);
            return ctdb.Metadata;
        }
```

- [ ] **Step 4: Replace active FreeDB fallback in `LookupAlbumInfo`**

In the active CTDB section of `LookupAlbumInfo`, replace the direct `CUEToolsDB` construction and contact:

```csharp
                var ctdb = new CUEToolsDB(TOC, proxy);
                ctdb.ContactDB(_config.advanced.CTDBServer, "CUETools " + CUEToolsVersion, null, false, false, metadataSearch);
                foreach (var meta in ctdb.Metadata)
```

with:

```csharp
                foreach (var meta in LookupCtdbMetadata(metadataSearch))
```

Replace the current block starting with:

```csharp
            if (!ctdbFound && metadataSearch == CTDBMetadataSearch.Extensive)
            {
                ShowProgress("Looking up album via Freedb...", 0.0, null, null);
```

and ending just before:

```csharp
            ShowProgress("", 0, null, null);
```

with:

```csharp
            if (metadataSearch == CTDBMetadataSearch.Extensive)
            {
                ShowProgress("Looking up album via AccurateRip Meta...", 0.0, null, null);
                CheckStop();

                AccurateRipMetaProvider provider = AccurateRipMetaProvider ?? new AccurateRipMetaProvider();
                CUEMetadataEntry accurateRipMetaEntry = provider.Lookup(TOC, proxy);
                if (accurateRipMetaEntry != null)
                    Releases.Add(accurateRipMetaEntry);
            }
```

- [ ] **Step 5: Keep old FreeDB lookup dormant**

Move the complete FreeDB lookup block removed in Step 4 to immediately above `ShowProgress("", 0, null, null);`. Put `#if DORMANT_FREEDB` immediately before the moved `if (!ctdbFound && metadataSearch == CTDBMetadataSearch.Extensive)` line and `#endif` immediately after the moved block's closing brace. Add this comment as the first line inside the dormant region:

```csharp
            // FreeDB lookup is dormant. AccurateRip Meta is the active replacement provider.
```

Keep the exact removed FreeDB code inside the wrapper. Do not leave an active `FreedbHelper`, `QueryResult`, `QueryResultCollection`, or `CDEntry` reference outside the wrapper.

- [ ] **Step 6: Wrap `TocFromCDEntry` as dormant**

In `CUETools.Processor/CUESheet.cs`, place `#if DORMANT_FREEDB` immediately before the current `public CDImageLayout TocFromCDEntry(CDEntry cdEntry)` declaration and `#endif` immediately after that method's closing brace. Add this comment as the first line inside the dormant region:

```csharp
        // FreeDB TOC reconstruction is dormant with the FreeDB provider.
```

Keep the existing `TocFromCDEntry` method body unchanged inside that dormant region.

- [ ] **Step 7: Run CUESheet lookup tests**

Run:

```powershell
dotnet test CUETools\CUETools.TestProcessor\CUETools.TestProcessor.csproj --filter "FullyQualifiedName~LookupAlbumInfo" -v:minimal
```

Expected: PASS for both CUESheet lookup tests.

- [ ] **Step 8: Commit CUESheet integration**

```powershell
git add CUETools.Processor/CUESheet.cs CUETools/CUETools.TestProcessor/AccurateRipMetaTest.cs
git commit -m "feat: use AccurateRip Meta in processor lookup"
```

---

### Task 5: Isolate FreeDB Metadata Helpers

**Files:**
- Modify: `CUETools.Processor/CUEMetadata.cs`
- Test: `CUETools/CUETools.TestProcessor/AccurateRipMetaTest.cs`

Tasks 5, 6, and 7 form one atomic build-safe implementation group. Do not commit, hand off, or intentionally stop after Task 5 or Task 6, because the helper methods are removed from the active processor API before all active UI callers are wrapped. Commit the group only after Task 7 builds CUERipper.

- [ ] **Step 1: Move FreeDB-specific helpers behind a dormant compile symbol**

In `CUETools.Processor/CUEMetadata.cs`, create two separate `#if DORMANT_FREEDB` regions so the active `FillFromCtdb` method remains compiled:

- Region 1 starts immediately before the current `public void FillFromFreedb(Freedb.CDEntry cdEntry, int firstAudio)` declaration and ends immediately after that method's closing brace.
- Region 2 starts immediately before the current `private static string FreedbToEncoding(Encoding iso, Encoding def, ref bool changed, ref bool error, string s)` declaration and ends immediately after the current `public bool FreedbToVarious()` method's closing brace.

Add this comment as the first line inside both dormant regions:

```csharp
        // FreeDB metadata import and repair helpers are dormant. AccurateRip Meta is the active replacement provider.
```

The dormant regions together must contain these existing declarations with their current bodies unchanged:

```csharp
        public void FillFromFreedb(Freedb.CDEntry cdEntry, int firstAudio)
        private static string FreedbToEncoding(Encoding iso, Encoding def, ref bool changed, ref bool error, string s)
        public bool FreedbToEncoding()
        public bool FreedbToVarious()
```

Do not include `public void FillFromCtdb(CUETools.CTDB.CTDBResponseMeta cdEntry, int firstAudio)` in either dormant region. `CUESheet.LookupAlbumInfo` and CUERipper still call `FillFromCtdb` in the active CTDB metadata path.

- [ ] **Step 2: Continue without committing**

Run this focused test only to catch accidental mapper regressions while the broader working tree is temporarily between caller updates:

```powershell
dotnet test CUETools\CUETools.TestProcessor\CUETools.TestProcessor.csproj --filter AccurateRipMetaTest -v:minimal
```

Expected: PASS. Do not run or claim a whole-solution build yet; CUETools and CUERipper callers are updated in Tasks 6 and 7 before the commit.

- [ ] **Step 3: Do not commit yet**

Expected: no commit. Keep `CUETools.Processor/CUEMetadata.cs` in the working tree and continue directly to Task 6.

---

### Task 6: Update CUETools Metadata Chooser UI

**Files:**
- Modify: `CUETools/frmChoice.cs`

- [ ] **Step 1: Register an AccurateRip Meta image key**

Change the `frmChoice` constructor from:

```csharp
        public frmChoice()
        {
            InitializeComponent();
        }
```

to:

```csharp
        public frmChoice()
        {
            InitializeComponent();
            if (!imageList1.Images.ContainsKey(AccurateRipMeta.SourceKey))
                using (Image accurateRipIcon = Properties.Resources.accuraterip16)
                    imageList1.Images.Add(AccurateRipMeta.SourceKey, accurateRipIcon);
        }
```

- [ ] **Step 2: Make FreeDB duplicate insertion dormant**

In `AddItem`, replace this active block:

```csharp
                if (entry.ImageKey == "freedb")
                {
                    // check if the entry contains non-iso characters,
                    // and add a second one if it does
                    CUEMetadata copy = new CUEMetadata(entry.metadata);
                    if (copy.FreedbToEncoding())
                    {
                        entry = new CUEMetadataEntry(copy, entry.TOC, entry.ImageKey);
                        item = new ListViewItem(entry.ToString(), entry.ImageKey);
                        item.Tag = entry;
                        listChoices.Items.Add(item);
                    }
                }
```

with:

```csharp
#if DORMANT_FREEDB
                // FreeDB encoding repair is dormant. AccurateRip Meta entries are already Unicode JSON.
                if (entry.ImageKey == "freedb")
                {
                    CUEMetadata copy = new CUEMetadata(entry.metadata);
                    if (copy.FreedbToEncoding())
                    {
                        entry = new CUEMetadataEntry(copy, entry.TOC, entry.ImageKey);
                        item = new ListViewItem(entry.ToString(), entry.ImageKey);
                        item.Tag = entry;
                        listChoices.Items.Add(item);
                    }
                }
#endif
```

- [ ] **Step 3: Build CUETools project**

Run:

```powershell
dotnet build CUETools\CUETools.csproj -v:minimal '/clp:ErrorsOnly;Summary'
```

Expected: Build succeeds. `frmChoice` should compile without any active FreeDB helper calls.

- [ ] **Step 4: Do not commit yet**

Expected: no commit. Keep `CUETools/frmChoice.cs` in the working tree and continue directly to Task 7.

---

### Task 7: Update CUERipper Lookup And UI

**Files:**
- Modify: `CUERipper/frmCUERipper.cs`
- Modify: `CUERipper/Options.cs`

- [ ] **Step 1: Make FreeDB using dormant and add LINQ**

At the top of `CUERipper/frmCUERipper.cs`, replace:

```csharp
using Freedb;
```

with:

```csharp
#if DORMANT_FREEDB
using Freedb;
#endif
```

Also add:

```csharp
using System.Linq;
```

- [ ] **Step 2: Register an AccurateRip Meta image key**

In the `frmCUERipper` constructor, immediately after `InitializeComponent();`, add:

```csharp
            if (!imageListMetadataSource.Images.ContainsKey(AccurateRipMeta.SourceKey))
                using (Image accurateRipIcon = Properties.Resources.accuraterip16)
                    imageListMetadataSource.Images.Add(AccurateRipMeta.SourceKey, accurateRipIcon);
```

- [ ] **Step 3: Hide FreeDB-only controls in `SetupControls`**

Replace these lines in `SetupControls`:

```csharp
            buttonFreedbSubmit.Enabled = data.selectedRelease != null && !running;
            buttonVA.Enabled = data.selectedRelease != null && !running &&
                data.selectedRelease.ImageKey == "freedb" && !data.selectedRelease.metadata.IsVarious() && (new CUEMetadata(data.selectedRelease.metadata)).FreedbToVarious();
            buttonEncoding.Enabled = data.selectedRelease != null && !running &&
                data.selectedRelease.ImageKey == "freedb" && (new CUEMetadata(data.selectedRelease.metadata)).FreedbToEncoding();
```

with:

```csharp
            buttonFreedbSubmit.Visible = false;
            buttonFreedbSubmit.Enabled = false;
            buttonVA.Visible = false;
            buttonVA.Enabled = false;
            buttonEncoding.Visible = false;
            buttonEncoding.Enabled = false;
```

- [ ] **Step 4: Add an AccurateRip Meta progress helper**

Add this active helper at the current `FreeDB_LookupProgress` location:

```csharp
        private void AccurateRipMetaLookupProgress(string detail)
        {
            CheckStop();
            string text = Properties.Resources.LookingUpVia + " " + AccurateRipMeta.DisplayName + "..." + (string.IsNullOrEmpty(detail) ? "" : " " + detail);
            this.BeginInvoke((MethodInvoker)delegate()
            {
                toolStripStatusLabel1.Text = text;
                toolStripProgressBar1.Value = (100 + 2 * toolStripProgressBar1.Value) / 3;
            });
        }
```

Preserve the old `FreeDB_LookupProgress` body under `#if DORMANT_FREEDB`, because the dormant FreeDB lookup block still calls it when the dormant symbol is enabled. Put `#if DORMANT_FREEDB` immediately before its current declaration and `#endif` immediately after that method's closing brace. Add this comment as the first line inside the dormant region:

```csharp
        // FreeDB progress UI is dormant with the FreeDB provider.
```

- [ ] **Step 5: Replace CUERipper FreeDB lookup block**

Keep this replacement inside the existing `if (data.Releases.Count == 0 || loadAllMetadata)` gate. CUERipper intentionally preserves the current fallback/reload lookup semantics: it does not issue an AccurateRip Meta request on ordinary lookups where CTDB already populated `data.Releases`, but reload/all-metadata still queries it even when CTDB returned entries.

In `Lookup`, replace the active block from:

```csharp
                this.BeginInvoke((MethodInvoker)delegate() { toolStripStatusLabel1.Text = Properties.Resources.LookingUpVia + " Freedb..."; });

                FreedbHelper m_freedb = new FreedbHelper();
```

through the end of the FreeDB `try/catch` block with:

```csharp
                AccurateRipMetaLookupProgress(null);
                try
                {
                    var provider = new AccurateRipMetaProvider();
                    CUEMetadataEntry accurateRipMeta = provider.Lookup(audioSource.TOC, _config.GetProxy());
                    if (accurateRipMeta != null && !data.Releases.Any(r =>
                        r.ImageKey == AccurateRipMeta.SourceKey &&
                        r.metadata.Contains(accurateRipMeta.metadata)))
                    {
                        data.Releases.Add(accurateRipMeta);
                    }
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Trace.WriteLine(ex.Message);
                }
```

The duplicate check is intentionally narrow: it suppresses repeated AccurateRip Meta entries on reload, but it does not hide an AccurateRip Meta release just because an existing CTDB/local release has the same metadata text.

Move the complete removed FreeDB lookup block immediately below the new AccurateRip Meta block. Put `#if DORMANT_FREEDB` immediately before the moved block and `#endif` immediately after it. Add this comment as the first line inside the dormant region:

```csharp
                // FreeDB lookup is dormant. AccurateRip Meta is the active replacement provider.
```

- [ ] **Step 6: Fetch every release's artwork before CTDB-only artwork scan**

Replace the artwork worker launch:

```csharp
                    backgroundWorkerArtwork.RunWorkerAsync(new BackgroundWorkerArtworkArgs() { cueSheet = cueSheet, meta = data.selectedRelease });
```

with:

```csharp
                    backgroundWorkerArtwork.RunWorkerAsync(new BackgroundWorkerArtworkArgs()
                    {
                        cueSheet = cueSheet,
                        meta = data.selectedRelease,
                        releases = data.Releases.ToList()
                    });
```

Replace the full `backgroundWorkerArtwork_DoWork` method in `CUERipper/frmCUERipper.cs` with:

```csharp
        private void backgroundWorkerArtwork_DoWork(object sender, DoWorkEventArgs e)
        {
            var args = e.Argument as BackgroundWorkerArtworkArgs;
            var cueSheet = args.cueSheet;
            albumArt.Clear();
            currentAlbumArt = 0;
            var knownUrls = new List<string>();
            var firstUrls = new List<string>();
            var releaseCovers = new List<CTDBResponseMetaImage>();

            var releases = args.releases ?? new List<CUEMetadataEntry>();
            if (releases.Count == 0 && args.meta != null)
                releases.Add(args.meta);

            foreach (var release in releases)
            {
                if (release == null || release.metadata == null || release.metadata.AlbumArt == null)
                    continue;

                foreach (var releaseCover in release.metadata.AlbumArt)
                {
                    if (releaseCover == null)
                        continue;

                    releaseCovers.Add(releaseCover);
                    if (object.ReferenceEquals(release, args.meta) && !string.IsNullOrEmpty(releaseCover.uri))
                        firstUrls.Add(releaseCover.uri);
                }
            }

            foreach (var releaseCover in releaseCovers)
            {
                string fetchUrl = !string.IsNullOrEmpty(releaseCover.uri150) ? releaseCover.uri150 : releaseCover.uri;
                if (string.IsNullOrEmpty(fetchUrl) || knownUrls.Contains(fetchUrl))
                    continue;

                var ms = new MemoryStream();
                if (!cueSheet.CTDB.FetchFile(fetchUrl, ms))
                    continue;

                lock (this.albumArt)
                {
                    if (backgroundWorkerArtwork.CancellationPending)
                    {
                        e.Cancel = true;
                        return;
                    }
                    this.albumArt.Add(new AlbumArt(releaseCover, ms.ToArray()));
                }
                knownUrls.Add(fetchUrl);
                backgroundWorkerArtwork.ReportProgress(0);
            }

            for (int i = 0; i < 2; i++)
            {
                foreach (var metadata in cueSheet.CTDB.Metadata)
                {
                    if (metadata.coverart == null)
                        continue;
                    foreach (var coverart in metadata.coverart)
                    {
                        var uri = coverart.uri150;
                        if (knownUrls.Contains(uri) ||
                            (_config.advanced.coversSearch == CUEConfigAdvanced.CTDBCoversSearch.Primary && !coverart.primary))
                            continue;
                        if (i == 0 && !firstUrls.Contains(coverart.uri))
                            continue;
                        var ms = new MemoryStream();
                        if (!cueSheet.CTDB.FetchFile(uri, ms))
                            continue;
                        lock (this.albumArt)
                        {
                            if (backgroundWorkerArtwork.CancellationPending)
                            {
                                e.Cancel = true;
                                return;
                            }
                            this.albumArt.Add(new AlbumArt(coverart, ms.ToArray()));
                        }
                        knownUrls.Add(uri);
                        backgroundWorkerArtwork.ReportProgress(0);
                    }
                }
            }
        }
```

Replace `BackgroundWorkerArtworkArgs` with:

```csharp
    internal class BackgroundWorkerArtworkArgs
    {
        public CUESheet cueSheet;
        public CUEMetadataEntry meta;
        public List<CUEMetadataEntry> releases;
    }
```

- [ ] **Step 7: Make `CreateCUESheet(ICDRipper, CDEntry)` dormant**

In `CUERipper/frmCUERipper.cs`, place `#if DORMANT_FREEDB` immediately before the current `private CUEMetadataEntry CreateCUESheet(ICDRipper audioSource, CDEntry cdEntry)` declaration and `#endif` immediately after that method's closing brace. Add this comment as the first line inside the dormant region:

```csharp
        // FreeDB CDEntry conversion is dormant with the FreeDB provider.
```

Keep the existing overload body unchanged inside that dormant region.

- [ ] **Step 8: Keep designer-required FreeDB button handlers as stubs**

Replace `buttonVA_Click`, `buttonEncoding_Click`, and `buttonFreedbSubmit_Click` bodies with non-dormant stubs:

```csharp
        private void buttonVA_Click(object sender, EventArgs e)
        {
            // FreeDB various-artist repair is dormant. AccurateRip Meta does not need this action.
        }

        private void buttonEncoding_Click(object sender, EventArgs e)
        {
            // FreeDB encoding repair is dormant. AccurateRip Meta responses are Unicode JSON.
        }

        private void buttonFreedbSubmit_Click(object sender, EventArgs e)
        {
            // FreeDB submission is dormant. AccurateRip Meta currently has no submission API.
        }
```

Move the old bodies below the stubs under `#if DORMANT_FREEDB` with their original method names changed to avoid duplicate definitions. The old `buttonVA_Click` body moves into `DormantFreedbVariousArtistRepair`, the old `buttonEncoding_Click` body moves into `DormantFreedbEncodingRepair`, and the existing `FreedbSubmit(object o)` body stays under the same dormant region.

- [ ] **Step 9: Remove FreeDB setting from normal CUERipper options**

In `CUERipper/Options.cs`, replace:

```csharp
        [DefaultValue("gnudb.gnudb.org"), Category("Various"), DisplayName("Freedb site address")]
        public string FreedbSiteAddress  { get { return config.advanced.FreedbSiteAddress ; } set { config.advanced.FreedbSiteAddress  = value; } }
```

with:

```csharp
#if DORMANT_FREEDB
        [DefaultValue("gnudb.gnudb.org"), Category("Various"), DisplayName("Freedb site address")]
        public string FreedbSiteAddress  { get { return config.advanced.FreedbSiteAddress ; } set { config.advanced.FreedbSiteAddress  = value; } }
#endif
```

- [ ] **Step 10: Build CUERipper project**

Run:

```powershell
dotnet build CUERipper\CUERipper.csproj -v:minimal '/clp:ErrorsOnly;Summary'
```

Expected: Build succeeds. No active `Freedb` namespace references should remain in `frmCUERipper.cs`.

- [ ] **Step 11: Commit CUERipper integration**

```powershell
git add CUETools.Processor/CUEMetadata.cs CUETools/frmChoice.cs CUERipper/frmCUERipper.cs CUERipper/Options.cs
git commit -m "feat: make FreeDB UI and helpers dormant"
```

---

### Task 8: Hide FreeDB Advanced Settings

**Files:**
- Modify: `CUETools.Processor/CUEConfigAdvanced.cs`
- Inspect: `CUETools/frmSettings.cs:43-48`
- Test: `CUETools/CUETools.TestProcessor/AccurateRipMetaTest.cs`

- [ ] **Step 1: Add a failing settings exposure test**

Append this test to `AccurateRipMetaTest` before the `FakeTransport` class:

```csharp
        [TestMethod]
        public void CUEConfigAdvancedDoesNotExposeFreedbSettings()
        {
            var freedbProperties = TypeDescriptor.GetProperties(new CUEConfigAdvanced())
                .Cast<PropertyDescriptor>()
                .Where(property => property.Category == "Freedb")
                .Select(property => property.Name)
                .ToList();

            CollectionAssert.AreEqual(new string[0], freedbProperties);
        }
```

`System.ComponentModel` was added to the test file in Task 1 so `TypeDescriptor` and `PropertyDescriptor` resolve.

- [ ] **Step 2: Run the settings exposure test to verify it fails**

Run:

```powershell
dotnet test CUETools\CUETools.TestProcessor\CUETools.TestProcessor.csproj --filter CUEConfigAdvancedDoesNotExposeFreedbSettings -v:minimal
```

Expected: FAIL because `CUEConfigAdvanced` still exposes `FreedbUser`, `FreedbDomain`, and `FreedbSiteAddress` with `Category("Freedb")`.

- [ ] **Step 3: Make FreeDB advanced settings dormant**

In `CUETools.Processor/CUEConfigAdvanced.cs`, replace the active FreeDB settings properties:

```csharp
        [DefaultValue("i"), Category("Freedb"), DisplayName("Email user")]
        public string FreedbUser { get; set; }

        [DefaultValue("wont.tell"), Category("Freedb"), DisplayName("Email domain")]
        public string FreedbDomain { get; set; }

        [DefaultValue("gnudb.gnudb.org"), Category("Freedb"), DisplayName("Site address")]
        public string FreedbSiteAddress { get; set; }
```

with:

```csharp
#if DORMANT_FREEDB
        [DefaultValue("i"), Category("Freedb"), DisplayName("Email user")]
        public string FreedbUser { get; set; }

        [DefaultValue("wont.tell"), Category("Freedb"), DisplayName("Email domain")]
        public string FreedbDomain { get; set; }

        [DefaultValue("gnudb.gnudb.org"), Category("Freedb"), DisplayName("Site address")]
        public string FreedbSiteAddress { get; set; }
#endif
```

- [ ] **Step 4: Confirm CUETools settings UI now follows the hidden properties**

Run:

```powershell
rg -n 'propertyGrid1.SelectedObject = _config\.advanced|Category\("Freedb"\)|FreedbUser|FreedbDomain|FreedbSiteAddress' CUETools\frmSettings.cs CUETools.Processor\CUEConfigAdvanced.cs
```

Expected: `CUETools/frmSettings.cs` still has `propertyGrid1.SelectedObject = _config.advanced`; FreeDB properties and `Category("Freedb")` remain only inside the `#if DORMANT_FREEDB` region in `CUEConfigAdvanced.cs`.

- [ ] **Step 5: Run the settings exposure test**

Run:

```powershell
dotnet test CUETools\CUETools.TestProcessor\CUETools.TestProcessor.csproj --filter CUEConfigAdvancedDoesNotExposeFreedbSettings -v:minimal
```

Expected: PASS.

- [ ] **Step 6: Commit FreeDB settings dormancy**

```powershell
git add CUETools.Processor/CUEConfigAdvanced.cs CUETools/CUETools.TestProcessor/AccurateRipMetaTest.cs
git commit -m "refactor: make FreeDB settings dormant"
```

---

### Task 9: Remove Active FreeDB Project References

**Files:**
- Modify: `CUETools.Processor/CUETools.Processor.csproj`
- Modify: `CUETools/CUETools.csproj`
- Modify: `CUERipper/CUERipper.csproj`
- Modify: `CUETools/CUETools.TestProcessor/CUETools.TestProcessor.csproj`
- Modify: `CUETools/CUETools.TestCodecs/CUETools.TestCodecs.csproj`
- Modify: `CUETools.sln`

- [ ] **Step 1: Confirm no active source references FreeDB**

Run:

```powershell
rg -n "using Freedb|FreedbHelper|FillFromFreedb|FreedbToEncoding|FreedbToVarious|FreedbUser|FreedbDomain|FreedbSiteAddress|Category\(\"Freedb\"\)|CDEntry|QueryResult" CUETools.Processor CUETools CUERipper CUETools\CUETools.TestProcessor CUETools\CUETools.TestCodecs -g "*.cs"
```

Expected: Matches only inside `#if DORMANT_FREEDB` regions, comments, dormant forms, or resource/designer artifacts. No normal active code path should require the `Freedb` project or expose FreeDB settings.

- [ ] **Step 2: Remove FreeDB project references**

Remove these XML lines:

From `CUETools.Processor/CUETools.Processor.csproj`:

```xml
    <ProjectReference Include="..\Freedb\Freedb.csproj" />
```

From `CUETools/CUETools.csproj`:

```xml
    <ProjectReference Include="..\Freedb\Freedb.csproj" />
```

From `CUERipper/CUERipper.csproj`:

```xml
    <ProjectReference Include="..\Freedb\Freedb.csproj" />
```

From `CUETools/CUETools.TestProcessor/CUETools.TestProcessor.csproj`:

```xml
    <ProjectReference Include="..\..\Freedb\Freedb.csproj" />
```

From `CUETools/CUETools.TestCodecs/CUETools.TestCodecs.csproj`:

```xml
    <ProjectReference Include="..\..\Freedb\Freedb.csproj" />
```

- [ ] **Step 3: Remove FreeDB project from solution**

Run:

```powershell
dotnet sln CUETools.sln remove Freedb\Freedb.csproj
```

Expected: command reports that `Freedb\Freedb.csproj` was removed from the solution.

- [ ] **Step 4: Check for remaining solution-level FreeDB references**

Run:

```powershell
rg -n "Freedb|freedb" CUETools.sln
```

Expected: no active project reference remains. If `ThirdParty\Freedb.dll` remains as a solution item, remove that line from `CUETools.sln` because the active solution no longer ships FreeDB as a provider.

- [ ] **Step 5: Build affected projects**

Run:

```powershell
dotnet build CUETools.Processor\CUETools.Processor.csproj -v:minimal '/clp:ErrorsOnly;Summary'
dotnet build CUETools\CUETools.csproj -v:minimal '/clp:ErrorsOnly;Summary'
dotnet build CUERipper\CUERipper.csproj -v:minimal '/clp:ErrorsOnly;Summary'
dotnet build CUETools\CUETools.TestCodecs\CUETools.TestCodecs.csproj -v:minimal '/clp:ErrorsOnly;Summary'
dotnet test CUETools\CUETools.TestProcessor\CUETools.TestProcessor.csproj --filter AccurateRipMetaTest -v:minimal
```

Expected: all commands succeed.

- [ ] **Step 6: Commit dependency cleanup**

```powershell
git add CUETools.Processor/CUETools.Processor.csproj CUETools/CUETools.csproj CUERipper/CUERipper.csproj CUETools/CUETools.TestProcessor/CUETools.TestProcessor.csproj CUETools/CUETools.TestCodecs/CUETools.TestCodecs.csproj CUETools.sln
git commit -m "refactor: remove active FreeDB dependencies"
```

---

### Task 10: Final Verification

**Files:**
- No planned source edits

- [ ] **Step 1: Run focused tests**

Run:

```powershell
dotnet test CUETools\CUETools.TestProcessor\CUETools.TestProcessor.csproj --filter AccurateRipMetaTest -v:minimal
```

Expected: all AccurateRip Meta tests pass.

- [ ] **Step 2: Run solution inventory**

Run:

```powershell
dotnet sln CUETools.sln list
```

Expected: the solution lists `dotnet10-update` projects and no `Freedb\Freedb.csproj` entry.

- [ ] **Step 3: Build solution**

Run:

```powershell
dotnet build CUETools.sln -v:minimal '/clp:ErrorsOnly;Summary'
```

Expected: solution build succeeds. If it fails before reaching this feature because clean third-party submodules are missing CUETools patches, apply the documented parent patches temporarily:

```powershell
git apply ThirdParty/submodule_taglib-sharp_CUETools.patch
git apply ThirdParty/submodule_openclnet_CUETools.patch
git apply ThirdParty/submodule_WindowsMediaLib_CUETools.patch
dotnet build CUETools.sln -v:minimal '/clp:ErrorsOnly;Summary'
git apply -R ThirdParty/submodule_WindowsMediaLib_CUETools.patch
git apply -R ThirdParty/submodule_openclnet_CUETools.patch
git apply -R ThirdParty/submodule_taglib-sharp_CUETools.patch
```

Expected after temporary patches: build succeeds, and `git status --short` does not show retained third-party patch changes.

- [ ] **Step 4: Run solution tests without rebuilding**

Run:

```powershell
dotnet test CUETools.sln --no-build -v:minimal
```

Expected: all runnable tests pass; skipped tests remain skipped.

- [ ] **Step 5: Verify FreeDB is dormant**

Run:

```powershell
rg -n "Looking up album via Freedb|Submit to freedb|Freedb site address|Category\(\"Freedb\"\)|FreedbUser|FreedbDomain|FreedbSiteAddress|ImageKey == \"freedb\"|using Freedb|FreedbHelper" CUETools.Processor CUETools CUERipper -g "*.cs" -g "*.resx"
```

Expected: remaining matches are either in `#if DORMANT_FREEDB` code, comments, resource text for dormant forms, or files deliberately retained for possible future reactivation. No normal runtime path should display or call FreeDB, and no normal property grid should expose FreeDB settings.

- [ ] **Step 6: Verify FreeDB submit form is orphaned from normal UI**

Run:

```powershell
rg -n "new frmFreedbSubmit|frmFreedbSubmit|buttonFreedbSubmit_Click|FreedbSubmit\(" CUERipper -g "*.cs" -g "*.Designer.cs" -g "*.resx"
```

Expected: `frmFreedbSubmit` form/designer/resource files still exist, and the only construction path is inside the dormant submit code. Normal event-handler stubs must not create or show `frmFreedbSubmit`.

- [ ] **Step 7: Verify no active FreeDB project references remain**

Run:

```powershell
rg -n "Freedb\.csproj" -g "*.csproj" -g "*.sln"
```

Expected: no matches.

- [ ] **Step 8: Run manual smoke checks**

Run CUETools and CUERipper from the built output and check these behaviors with a disc or image that can return AccurateRip Meta data:

- CUETools metadata chooser shows `AccurateRip Meta` as a selectable source when the provider returns metadata.
- CUETools `CTDBMetadataSearch.Extensive` lookup still posts to AccurateRip Meta when CTDB has already returned metadata.
- CUETools metadata chooser shows cover preview when the provider returns a valid `_arturl`.
- CUERipper ordinary lookup preserves fallback semantics: if CTDB already populated releases, AccurateRip Meta is not queried until reload/all-metadata is requested.
- CUERipper reload/all-metadata lookup can show an `AccurateRip Meta` release.
- CUERipper artwork preview downloads the AccurateRip Meta cover URL when it is initially selected and when the user selects it later from a non-default release entry.
- CUERipper toolbar/control layout remains acceptable with the FreeDB submit/repair buttons hidden.
- CUERipper lookup progress text uses the localized `LookingUpVia` prefix and the `AccurateRip Meta` provider suffix in the default UI and, if available on the test machine, de-DE or ru-RU resources.
- FreeDB submit/settings are not reachable in normal CUETools or CUERipper UI.

Expected: all nine checks pass.

- [ ] **Step 9: Commit final verification cleanup if needed**

Run:

```powershell
git status --short
```

Expected: no uncommitted files from final verification. If this command lists cleanup files changed by the final verification task, stage exactly those listed cleanup files and commit them with:

```powershell
git commit -m "chore: finish AccurateRip Meta verification cleanup"
```

If no files changed during verification, do not create a commit.
