Release date: 2024-06-28

**Version 2.2.6 Prerequisites**
- Microsoft .NET Framework 4.7  - preinstalled in Windows 10 (since version 1703)
- Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017, 2019 and 2022 - needed for some plugins

Further information concerning installation:
http://cue.tools/wiki/CUETools_Download

**CUETools 2.2.6 Changelog**

- Update MAC_SDK from 10.37 to 10.74
- EACPlugin: Allow coverart search to be stopped
- Fix C2 mode for further drives:
  LG GH24NSD5, LITEON DH-20A4P
- Preserve encoding of localized EAC log files
- Add setting for UTF-8-BOM
  settings.txt: WriteUTF8BOM
  The setting is enabled by default, to preserve previous behavior
- CUETools, Correct filenames: Support UTF-8
- CUERipper: Fix incorrect TOC entry of first track
- CUERipper: Add setting to force ReadCDCommand
  settings.txt: 0 (ReadCdBEh), 1 (ReadCdD8h), 2 (Unknown/AutoDetect)
  Default: 2
- Add WavPack 5.7.0 encoder multithreading support
- Update WavPack from 5.6.0 to 5.7.0
- Avoid short HTOA files
- CUERipper: Detect too large album art earlier

**Links to ffmpeg dlls (from [v2.2.5](https://github.com/gchudov/cuetools.net/releases/tag/v2.2.5)):**
- [ffmpeg_6.1_dlls_win32.zip](https://github.com/gchudov/cuetools.net/releases/download/v2.2.5/ffmpeg_6.1_dlls_win32.zip)
- [ffmpeg_6.1_dlls_x64.zip](https://github.com/gchudov/cuetools.net/releases/download/v2.2.5/ffmpeg_6.1_dlls_x64.zip)