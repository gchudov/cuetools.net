@ECHO OFF
REM This script collects the built .exe, .dll etc. files required for running CUETools
REM Wolfgang Stöggl <c72578@yahoo.de>, 2020-2022.

REM The script is located in the subdirectory CUETools
echo %~dp0
pushd %~dp0
SET base_dir=..

REM Get version of CUETools
for /f "tokens=7 delims= " %%a in ('find "CUEToolsVersion =" %base_dir%\CUETools.Processor\CUESheet.cs') do set PRODUCTVER=%%a
REM echo %PRODUCTVER%
REM "2.1.7";

REM Remove double quotes and semicolon
set PRODUCTVER=%PRODUCTVER:"=%
set PRODUCTVER=%PRODUCTVER:;=%
echo CUETools version: %PRODUCTVER%

SET release_dir=..\bin\Release\CUETools_%PRODUCTVER%

mkdir %release_dir%

REM use xcopy instead of copy. xcopy creates directories if necessary and outputs the copied file
REM /Y Suppresses prompting to confirm that you want to overwrite an existing destination file.
REM /D xcopy copies all Source files that are newer than existing Destination files

REM 32 files from net47
xcopy /Y /D %base_dir%\bin\Release\net47\BluTools.exe %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\BluTools.exe.config %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUEControls.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUERipper.exe %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUERipper.exe.config %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.exe %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.AccurateRip.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.ALACEnc.exe %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.ALACEnc.exe.config %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.ARCUE.exe %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.CDImage.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.Codecs.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.Compression.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.Converter.exe %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.CTDB.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.CTDB.Types.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.eac3to.exe %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.eac3to.exe.config %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.exe.config %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.FLACCL.cmd.exe %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.FLACCL.cmd.exe.config %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.Flake.exe %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.Flake.exe.config %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.Parity.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.Processor.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.Ripper.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.Ripper.Console.exe %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.Ripper.Console.exe.config %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\DeviceId.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Freedb.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\ProgressODoom.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\TagLibSharp.dll %release_dir%

xcopy /Y /D %base_dir%\CUETools\License.txt %release_dir%
xcopy /Y /D %base_dir%\CUETools\user_profiles_enabled %release_dir%

xcopy /Y /D %base_dir%\bin\Release\net47\de-DE\* %release_dir%\de-DE\
xcopy /Y /D %base_dir%\bin\Release\net47\ru-RU\* %release_dir%\ru-RU\

xcopy /Y /D %base_dir%\bin\Release\net47\plugins\Bwg.Hardware.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\Bwg.Logging.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\Bwg.Scsi.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\CUETools.Codecs.ALAC.dll %release_dir%\plugins\
REM No more win32\CUETools.Codecs.BDLPCM.dll and win32\CUETools.Codecs.BDLPCM.dll. Instead: CUETools.Codecs.MPEG.dll
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\CUETools.Codecs.MPEG.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\CUETools.Codecs.FLACCL.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\CUETools.Codecs.Flake.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\CUETools.Codecs.HDCD.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\CUETools.Codecs.libFLAC.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\CUETools.Codecs.libmp3lame.dll %release_dir%\plugins\
REM No more win32\CUETools.Codecs.WavPack.dll and x64\CUETools.Codecs.WavPack.dll. Instead: CUETools.Codecs.libwavpack.dll
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\CUETools.Codecs.libwavpack.dll %release_dir%\plugins\
REM Nor more win32\CUETools.Codecs.APE.dll and x64\CUETools.Codecs.APE.dll. Instead: CUETools.Codecs.MACLib.dll
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\CUETools.Codecs.MACLib.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\CUETools.Codecs.WMA.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\CUETools.Compression.Zip.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\CUETools.Ripper.SCSI.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\flac.cl %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\OpenCLNet.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\WindowsMediaLib.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\CUETools.Codecs.ffmpegdll.dll %release_dir%\plugins\

xcopy /Y /D %base_dir%\bin\Release\net47\plugins\win32\CUETools.Codecs.TTA.dll %release_dir%\plugins\win32\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\win32\CUETools.Compression.Rar.dll %release_dir%\plugins\win32\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\x64\CUETools.Codecs.TTA.dll %release_dir%\plugins\x64\
REM CUETools.Compression.Rar.dll is the same in the win32 and x64 directory of 2.1.7. Copy from win32
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\win32\CUETools.Compression.Rar.dll %release_dir%\plugins\x64\

REM plugins translation files
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\de-DE\* %release_dir%\plugins\de-DE\
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\ru-RU\* %release_dir%\plugins\ru-RU\

REM ThirdParty
xcopy /Y /D %base_dir%\ThirdParty\FFmpeg.AutoGen\FFmpeg.AutoGen\bin\Release\net472\FFmpeg.AutoGen.dll %release_dir%\plugins\
xcopy /Y /D %base_dir%\ThirdParty\ICSharpCode.SharpZipLib.dll %release_dir%\plugins\

REM ThirdParty\Win32 plugins
xcopy /Y /D %base_dir%\ThirdParty\Win32\hdcd.dll %release_dir%\plugins\win32\
xcopy /Y /D %base_dir%\ThirdParty\Win32\libFLAC_dynamic.dll %release_dir%\plugins\win32\
xcopy /Y /D %base_dir%\ThirdParty\Win32\libmp3lame.dll %release_dir%\plugins\win32\
xcopy /Y /D %base_dir%\ThirdParty\Win32\MACLibDll.dll %release_dir%\plugins\win32\
xcopy /Y /D %base_dir%\ThirdParty\Win32\unrar.dll %release_dir%\plugins\win32\
xcopy /Y /D %base_dir%\ThirdParty\Win32\wavpackdll.dll %release_dir%\plugins\win32\

REM ThirdParty\x64 plugins
xcopy /Y /D %base_dir%\ThirdParty\x64\hdcd.dll %release_dir%\plugins\x64\
xcopy /Y /D %base_dir%\ThirdParty\x64\libFLAC_dynamic.dll %release_dir%\plugins\x64\
xcopy /Y /D %base_dir%\ThirdParty\x64\libmp3lame.dll %release_dir%\plugins\x64\
xcopy /Y /D %base_dir%\ThirdParty\x64\MACLibDll.dll %release_dir%\plugins\x64\
xcopy /Y /D %base_dir%\ThirdParty\x64\Unrar.dll %release_dir%\plugins\x64\
xcopy /Y /D %base_dir%\ThirdParty\x64\wavpackdll.dll %release_dir%\plugins\x64\

REM EAC Plugin
REM CUETools.CTDB.Types.dll is also required now
xcopy /Y /D %base_dir%\bin\Release\interop\EAC\CUETools.AccurateRip.dll %release_dir%\interop\EAC\
xcopy /Y /D %base_dir%\bin\Release\interop\EAC\CUETools.CDImage.dll %release_dir%\interop\EAC\
xcopy /Y /D %base_dir%\bin\Release\interop\EAC\CUETools.Codecs.dll %release_dir%\interop\EAC\
xcopy /Y /D %base_dir%\bin\Release\interop\EAC\CUETools.CTDB.dll %release_dir%\interop\EAC\
xcopy /Y /D %base_dir%\bin\Release\interop\EAC\CUETools.CTDB.EACPlugin.dll %release_dir%\interop\EAC\
xcopy /Y /D %base_dir%\bin\Release\interop\EAC\CUETools.CTDB.Types.dll %release_dir%\interop\EAC\
xcopy /Y /D %base_dir%\bin\Release\interop\EAC\CUETools.Parity.dll %release_dir%\interop\EAC\
xcopy /Y /D %base_dir%\bin\Release\interop\EAC\Newtonsoft.Json.dll %release_dir%\interop\EAC\

REM required for running CUERipper:
REM Newtonsoft.Json.dll
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\Newtonsoft.Json.dll %release_dir%

REM CUETools.LossyWAV.exe was not in 2.1.7 release, added
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.Codecs.LossyWAV.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.LossyWAV.exe %release_dir%

popd
