# AccurateRip Meta Replacement Design

## Goal

Replace CUETools' active FreeDB metadata implementation with AccurateRip Meta while leaving other metadata providers unchanged. The replacement should appear to users as `AccurateRip Meta`, provide cover art through the existing album-art flow, and avoid copying any API key or secret from the EasyEACGUI sample project.

Implementation work must happen on a feature branch based on `dotnet10-update` so the .NET 10 migration changes are included.

## Scope

In scope:

- Add a first-party AccurateRip Meta lookup path in `CUETools.Processor`.
- Replace active FreeDB lookup behavior in CUETools and CUERipper.
- Show AccurateRip Meta as a selectable metadata source in the UI.
- Preserve cover URLs from AccurateRip Meta using the existing `CUEMetadata.AlbumArt` model.
- Hide or disable FreeDB submit/settings UI in normal runtime.
- Mark old FreeDB-specific code sections as dormant and deletion/adaptation candidates.

Out of scope:

- Changing CTDB, MusicBrainz, local cache, CUE, tag, or AccurateRip verification behavior.
- Adding AccurateRip Meta submission support, because the provider currently does not support submissions.
- Adding new persistent metadata fields for provider values that do not have clear CUETools storage targets.
- Copying auxiliary EasyEACGUI infrastructure wholesale.

## Architecture

Add a small AccurateRip Meta client/mapper in `CUETools.Processor`, near the existing metadata lookup orchestration. The client calculates the provider id with `AccurateRipVerify.CalculateAccurateRipId(toc)`, posts that id to `http://meta.accuraterip.com/discmatch`, deserializes the JSON response with the existing `Newtonsoft.Json` dependency, and returns a `CUEMetadataEntry` with source key `accurateripmeta`.

`CUESheet.LookupAlbumInfo` will keep the current local/CUE/tag/CTDB flow, but `CTDBMetadataSearch.Extensive` will also query AccurateRip Meta and add its result as an additional selectable entry, even if CTDB already returned metadata. This differs from the current FreeDB fallback, which only runs if CTDB found nothing.

CUERipper will use the same processor-level provider for no-result fallback and explicit reload/all-metadata lookup. Active FreeDB calls will be replaced by AccurateRip Meta calls. FreeDB submit/config code will remain dormant and clearly marked rather than physically removed in the first pass.

## Data Mapping

Map AccurateRip Meta fields into the existing CUETools metadata model:

- `Artist` -> `CUEMetadata.Artist`
- `Album` -> `CUEMetadata.Title`
- `Year` -> `CUEMetadata.Year`
- first non-empty line of `Genre` -> `CUEMetadata.Genre`
- `UPC` -> `CUEMetadata.Barcode`
- `Label` -> `CUEMetadata.Label`
- `CatalogNum` -> `CUEMetadata.LabelNo`
- `tracks[].Title` -> `CUETrackMetadata.Title`
- `tracks[].Artist` -> `CUETrackMetadata.Artist`
- `tracks[].ISRC` -> `CUETrackMetadata.ISRC`
- `_arturl` -> `CUEMetadata.AlbumArt`

`CatalogNum` maps to `LabelNo` because CUETools already uses `LabelNo` as the catalog number slot: it is displayed as `Label#`, read from cue `REM CATALOGNUMBER`, and written to TagLib `CatalogNo`.

Cover art will be represented as a `CTDBResponseMetaImage` with `uri` and `uri150` set to `_arturl`, `primary = true`, and dimensions left at `0` when the provider does not supply dimensions. This lets the existing CUETools and CUERipper cover preview/download paths continue to work.

The AccurateRip Meta response also exposes `Styles`, `tracks[].Composers`, and `tracks[].Conductors`. These fields are available from the provider but intentionally unused until CUETools has explicit fields for them. Add a short code comment near the mapping to document that decision.

## UI Behavior

User-facing source text should be `AccurateRip Meta`. Internal source identity should use a new key such as `accurateripmeta`; do not reuse `freedb` for new results.

CUETools metadata chooser:

- Shows AccurateRip Meta entries as selectable releases.
- Uses the existing cover preview path when `_arturl` is present.
- Uses an existing AccurateRip icon where practical, or another existing resource if resource churn would be disproportionate.
- Does not offer FreeDB-specific encoding repair for AccurateRip Meta results.

CUERipper:

- Shows AccurateRip Meta entries during fallback/reload lookup.
- Hides or disables the FreeDB submit button because AccurateRip Meta has no submission API.
- Hides or disables FreeDB-specific encoding and various-artist repair actions unless they still apply only to dormant FreeDB entries.
- Removes normal UI access to FreeDB advanced settings if those settings are not useful for AccurateRip Meta.

## Error Handling And Networking

AccurateRip Meta lookup should fail soft. Network errors, empty responses, invalid JSON, missing required album fields, or incompatible track counts should trace/log the issue and add no AccurateRip Meta entry. Existing local/CUE/tag/CTDB results must remain available.

The request body should be built with JSON serialization rather than string concatenation. The implementation must reuse CUETools proxy settings where the existing metadata flow provides a proxy. No API key or secret should be introduced.

If `_arturl` is empty or invalid, the metadata entry can still be shown without cover art. Image bytes should not be downloaded during the metadata mapping step; existing album-art consumers should fetch URLs as they do for CTDB cover art.

## FreeDB Dormancy

FreeDB is no longer an active provider after this change. The old FreeDB lookup, submit form, submit handler, advanced settings, and encoding/various-artist helper paths should be marked as dormant and deletion/adaptation candidates.

Dormant code should not remain reachable from normal UI. If retaining dormant code prevents removal of the `Freedb` project reference, isolate it so the active solution can build cleanly and the dependency boundary is obvious.

## Testing And Verification

Add focused tests for the AccurateRip Meta mapper:

- Valid JSON maps album, track, UPC, label, catalog number, ISRC, and cover URL correctly.
- `CatalogNum` maps to `LabelNo`.
- Empty optional fields do not throw.
- `Styles`, `Composers`, and `Conductors` do not leak into comments or tags.
- Track-count mismatches fail soft.

Add lookup behavior tests where practical:

- `CTDBMetadataSearch.Extensive` adds an AccurateRip Meta entry even when CTDB returns metadata.
- Non-extensive lookup does not query AccurateRip Meta.
- Provider failure does not remove existing results.

Verification should include affected project builds and the broader solution build/test gate. If the current .NET 10 solution still needs temporary third-party patch application to build, apply and revert those patches only for verification.

Manual smoke checks:

- CUETools metadata chooser shows `AccurateRip Meta` with cover preview when `_arturl` exists.
- CUERipper reload/all-metadata lookup can show AccurateRip Meta.
- FreeDB submit/settings are not reachable in normal UI.
