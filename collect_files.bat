@ECHO OFF
REM This script collects the built .exe, .dll etc. files required for running CUETools
REM Wolfgang Stöggl <c72578@yahoo.de>, 2020-2025.

REM The script is located in the subdirectory CUETools
echo %~dp0
pushd %~dp0
SET base_dir=.

REM Get version of CUETools
for /f "tokens=6 delims= " %%a in ('find "CUEToolsVersion =" %base_dir%\CUETools.Processor\CUESheet.cs') do set PRODUCTVER=%%a
REM echo %PRODUCTVER%
REM "2.1.7";

REM Remove double quotes and semicolon
set PRODUCTVER=%PRODUCTVER:"=%
set PRODUCTVER=%PRODUCTVER:;=%
echo CUETools version: %PRODUCTVER%

SET release_dir=%base_dir%\bin\Release\CUETools_%PRODUCTVER%

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

xcopy /Y /D %base_dir%\License.txt %release_dir%
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
xcopy /Y /D %base_dir%\bin\Release\net47\plugins\FFmpeg.AutoGen.dll %release_dir%\plugins\
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

REM CUERipper Avalonia
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Base.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Controls.DataGrid.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Controls.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.DesignerSupport.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Desktop.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Dialogs.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Fonts.Inter.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.FreeDesktop.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Markup.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Markup.Xaml.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Metal.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.MicroCom.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Native.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.OpenGL.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Remote.Protocol.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Skia.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Themes.Fluent.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Vulkan.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.Win32.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Avalonia.X11.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\av_libglesv2.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CommunityToolkit.Mvvm.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUERipper.Avalonia.exe %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUERipper.Avalonia.exe.config %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\CUETools.Interop.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\DeviceId.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Freedb.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\HarfBuzzSharp.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\libHarfBuzzSharp.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\libSkiaSharp.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\MicroCom.Runtime.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Microsoft.Bcl.AsyncInterfaces.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Microsoft.Extensions.DependencyInjection.Abstractions.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Microsoft.Extensions.DependencyInjection.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Microsoft.Extensions.Localization.Abstractions.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Microsoft.Extensions.Localization.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Microsoft.Extensions.Logging.Abstractions.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Microsoft.Extensions.Logging.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Microsoft.Extensions.Options.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Microsoft.Extensions.Primitives.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Microsoft.Win32.Primitives.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\netstandard.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Newtonsoft.Json.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Serilog.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Serilog.Extensions.Logging.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Serilog.Sinks.File.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\SkiaSharp.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.AppContext.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Buffers.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Collections.Concurrent.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Collections.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Collections.Immutable.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Collections.NonGeneric.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Collections.Specialized.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.ComponentModel.Annotations.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.ComponentModel.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.ComponentModel.EventBasedAsync.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.ComponentModel.Primitives.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.ComponentModel.TypeConverter.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Console.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Data.Common.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Diagnostics.Contracts.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Diagnostics.Debug.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Diagnostics.DiagnosticSource.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Diagnostics.FileVersionInfo.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Diagnostics.Process.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Diagnostics.StackTrace.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Diagnostics.TextWriterTraceListener.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Diagnostics.Tools.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Diagnostics.TraceSource.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Diagnostics.Tracing.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Drawing.Primitives.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Dynamic.Runtime.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Globalization.Calendars.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Globalization.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Globalization.Extensions.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.IO.Compression.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.IO.Compression.ZipFile.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.IO.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.IO.FileSystem.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.IO.FileSystem.DriveInfo.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.IO.FileSystem.Primitives.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.IO.FileSystem.Watcher.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.IO.IsolatedStorage.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.IO.MemoryMappedFiles.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.IO.Pipelines.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.IO.Pipes.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.IO.UnmanagedMemoryStream.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Linq.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Linq.Expressions.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Linq.Parallel.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Linq.Queryable.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Memory.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Net.Http.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Net.NameResolution.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Net.NetworkInformation.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Net.Ping.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Net.Primitives.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Net.Requests.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Net.Security.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Net.Sockets.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Net.WebHeaderCollection.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Net.WebSockets.Client.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Net.WebSockets.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Numerics.Vectors.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.ObjectModel.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Reflection.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Reflection.Extensions.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Reflection.Primitives.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Resources.Reader.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Resources.ResourceManager.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Resources.Writer.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Runtime.CompilerServices.Unsafe.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Runtime.CompilerServices.VisualC.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Runtime.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Runtime.Extensions.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Runtime.Handles.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Runtime.InteropServices.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Runtime.InteropServices.RuntimeInformation.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Runtime.Numerics.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Runtime.Serialization.Formatters.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Runtime.Serialization.Json.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Runtime.Serialization.Primitives.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Runtime.Serialization.Xml.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Security.Claims.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Security.Cryptography.Algorithms.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Security.Cryptography.Csp.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Security.Cryptography.Encoding.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Security.Cryptography.Primitives.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Security.Cryptography.X509Certificates.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Security.Principal.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Security.SecureString.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Text.Encoding.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Text.Encoding.Extensions.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Text.RegularExpressions.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Threading.Channels.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Threading.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Threading.Overlapped.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Threading.Tasks.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Threading.Tasks.Extensions.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Threading.Tasks.Parallel.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Threading.Thread.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Threading.ThreadPool.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Threading.Timer.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.ValueTuple.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Xml.ReaderWriter.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Xml.XDocument.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Xml.XmlDocument.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Xml.XmlSerializer.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Xml.XPath.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\System.Xml.XPath.XDocument.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\TagLibSharp.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\Tmds.DBus.Protocol.dll %release_dir%
xcopy /Y /D %base_dir%\bin\Release\net47\arm64\libHarfBuzzSharp.dll %release_dir%\arm64\
xcopy /Y /D %base_dir%\bin\Release\net47\arm64\libSkiaSharp.dll %release_dir%\arm64\
xcopy /Y /D %base_dir%\bin\Release\net47\de-DE\Bwg.Scsi.resources.dll %release_dir%\de-DE\
xcopy /Y /D %base_dir%\bin\Release\net47\de-DE\CUETools.Ripper.SCSI.resources.dll %release_dir%\de-DE\
xcopy /Y /D %base_dir%\bin\Release\net47\nl-NL\CUERipper.Avalonia.resources.dll %release_dir%\nl-NL\
xcopy /Y /D %base_dir%\bin\Release\net47\ru-RU\Bwg.Scsi.resources.dll %release_dir%\ru-RU\
xcopy /Y /D %base_dir%\bin\Release\net47\ru-RU\CUETools.Ripper.SCSI.resources.dll %release_dir%\ru-RU\
xcopy /Y /D %base_dir%\bin\Release\net47\x64\libHarfBuzzSharp.dll %release_dir%\x64\
xcopy /Y /D %base_dir%\bin\Release\net47\x64\libSkiaSharp.dll %release_dir%\x64\
xcopy /Y /D %base_dir%\bin\Release\net47\x86\libHarfBuzzSharp.dll %release_dir%\x86\
xcopy /Y /D %base_dir%\bin\Release\net47\x86\libSkiaSharp.dll %release_dir%\x86\

popd
