# CUETools
CUETools is a tool for lossless audio/CUE sheet format conversion. The goal is to make sure the album image is preserved accurately. A lossless disc image must be lossless not only in preserving contents of the audio tracks, but also in preserving gaps and CUE sheet contents. Many applications lose vital information upon conversion, and don't support all possible CUE sheet styles. For example, foobar2000 loses disc pre-gap information when converting an album image, and doesn't support gaps appended (noncompliant) CUE sheets.
# Supported formats
Supports WAV, FLAC, APE, LossyWAV, ALAC, TTA, and WavPack audio input/output. Audio must be 16-bit, 44.1kHz samples stereo (i.e. CD PCM). Supports every CUE sheet style (embedded, single file, gaps appended/prepended/left out). It is also possible to process a set of audio files in a directory without a CUE sheet, or use a RAR archive as an input without unpacking it.
# CUERipper
CUERipper is a utility for extracting digital audio from CDs, an open-source alternative to EAC. It has a lot fewer configuration options, so is somewhat easier to use, and is included in CUETools package. It supports MusicBrainz and freeDB metadata databases, AccurateRip and CTDB.
# Installing
Prebuilt binaries can be downloaded from [CUETools Download](http://cue.tools/wiki/CUETools_Download).
## Installing from sources
* Get the CUETools sources from GitHub ([https://github.com/gchudov/cuetools.net](https://github.com/gchudov/cuetools.net)):
`git clone https://github.com/gchudov/cuetools.net.git`
* Get the required submodules using:
`git submodule update --init --recursive`
* Apply patches to ThirdParty modules:
`git apply --directory=ThirdParty/flac ThirdParty/submodule_flac_CUETools.patch --whitespace=nowarn`
`powershell -c "Expand-Archive ThirdParty/MAC_SDK/MAC_904_SDK.zip -DestinationPath ThirdParty/MAC_SDK/"`
`git apply --directory=ThirdParty/MAC_SDK ThirdParty/ThirdParty_MAC_SDK_CUETools.patch`
`git apply --directory=ThirdParty/taglib-sharp ThirdParty/submodule_taglib-sharp_CUETools.patch`
`git apply --directory=ThirdParty/WavPack ThirdParty/submodule_WavPack_CUETools.patch`
`git apply --directory=ThirdParty/WindowsMediaLib ThirdParty/submodule_WindowsMediaLib_CUETools.patch`
* The solution can be built using Microsoft Visual Studio 2017 or newer (Community Edition will work)
  * Install the required .NET development tools (currently .NET Framework 4.7 and .NET Core 2.0)
  * Install an appropriate Windows SDK version (e.g. 10.0.16299.0 or newer)
  * Install the Microsoft Visual Studio Installer Projects
* Open cuetools.net\CUETools.sln
* Select 'Any CPU' under 'Solution Platforms'
* Build solution
