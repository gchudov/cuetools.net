# CUE Ripper for Linux

Experimental Linux console port of the `console-ripper-flac` fork. Uses the same
CD reader, correction modes, drive offsets, AccurateRip checks, CTDB metadata,
managed Flake encoder, and embedded CUE/log output as the Windows console ripper.
Windows projects and their SCSI transport remain separate.

Unknown drive offsets are calibrated automatically at the start of a rip using
AccurateRip's frame-450 checksums. Three tracks must agree on one correction;
multiple possible pressings, silence and missing database data are inconclusive.
The helper reads eleven sectors per sample with two secure passes, tries at most
six samples, stops starting samples after five seconds, and limits each command
to two seconds. Normal metadata queries do not perform calibration. A failed
calibration retains the previous zero-offset fallback and reports that explicitly.
Known drive offsets and explicit `--offset` values (including zero) take priority.

The correction applies to this disc only, not a permanently guessed offset for a
generic model name. The normal full-track AccurateRip verification still runs.
This shared calibration also builds for Windows with the existing .NET Framework
4.7 projects and native SCSI transport:

```powershell
./Windows/build.ps1 -DestinationDirectory C:/temp/cueripper-update
```

That command tests and builds the two changed Windows components for overlay onto
the existing CUETools distribution. Shared regressions compare the new checksum
scanner against the original full-disc AccurateRip verifier on both platforms.

Build requirements: .NET 10 SDK, a C compiler and Linux development headers for
the ABI check. Initialize the pinned metadata library first:

```sh
git submodule update --init ThirdParty/taglib-sharp
bash Linux/build.sh --test
bash Linux/build.sh --publish /tmp/cueripper-linux
```

Usage:

```sh
/tmp/cueripper-linux/CUERipper --list-drives
/tmp/cueripper-linux/CUERipper --info --drive /dev/sr0
# Run inside the directory where the new FLAC, CUE and log should be written.
/tmp/cueripper-linux/CUERipper --flac --secure --drive /dev/sr0 --filename Album
```

The logged-in user needs read/write access to the optical device. Desktop
distributions normally grant local-session access or use the `cdrom` group.
Do not run the music player as root. An audio CD does not need to be mounted.
Only `/dev/srN` devices identified as optical by sysfs (including aliases resolving
to them) are accepted. The helper uses `SG_IO`; it does not depend on Wine or Mono.

Exit codes: 0 success, 1 operation failure, 2 invalid arguments, 3 no optical
drives, 4 no audio tracks. Missing metadata or an absent AccurateRip database
entry is not itself a ripping failure. Results must still be reviewed in the log.

Software checks cover the native SCSI ABI, command layouts, sense errors, short
audio transfers, non-optical device rejection, and sample-exact FLAC output after
Unicode metadata/CUE/log tagging. Gap checks replay stale Q packets captured from
a Pioneer BDR-XS06 and inject invalid checksums/transport failures. Linux retries
an unusable gap-search response at that same position up to twice, retaining
CRC/track/position validation. Persistent invalid data still fails, including in
small search ranges; it is not replaced with guessed gap boundaries. Normal reads
and the Windows reader are unchanged. These checks do not replace physical testing.
Before release, test real discs in secure/burst/paranoid modes, cancel/eject,
disc replacement, mixed-mode discs, drive offsets and AccurateRip results.

This executable and its modifications are distributed under GPL-3.0-or-later
(the underlying GPL-2.0-or-later code permits that choice). Keep the original
copyright and license notices, LGPL notices for the codec/tagging components,
and MIT notices for the .NET runtime, DeviceId and Newtonsoft.Json. Distribute
the corresponding sources, including the pinned TagLib submodule and these
Linux build files, alongside binary packages.
