# CUE Ripper for Linux

Experimental Linux console port of the `console-ripper-flac` fork. Uses the same
CD reader, correction modes, drive offsets, AccurateRip checks, CTDB metadata,
managed Flake encoder, and embedded CUE/log output as the Windows console ripper.
Windows projects and their SCSI transport remain separate.

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
Unicode metadata/CUE/log tagging. They do not replace physical drive testing.
Before release, test real discs in secure/burst/paranoid modes, cancel/eject,
disc replacement, mixed-mode discs, drive offsets and AccurateRip results.

This executable and its modifications are distributed under GPL-3.0-or-later
(the underlying GPL-2.0-or-later code permits that choice). Keep the original
copyright and license notices, LGPL notices for the codec/tagging components,
and MIT notices for the .NET runtime, DeviceId and Newtonsoft.Json. Distribute
the corresponding sources, including the pinned TagLib submodule and these
Linux build files, alongside binary packages.
