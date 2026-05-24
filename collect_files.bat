@ECHO OFF
REM Collect the built .exe, .dll, .json, and config files required for running CUETools.
REM Wolfgang Stoeggl <c72578@yahoo.de>, 2020-2026.

REM The script is located in the CUETools repository root.
echo %~dp0
pushd %~dp0
SET base_dir=.

REM Get version of CUETools
for /f "tokens=7 delims= " %%a in ('find "CUEToolsVersion =" %base_dir%\CUETools.Processor\CUESheet.cs') do set PRODUCTVER=%%a
REM echo %PRODUCTVER%
REM "2.1.7";

REM Remove double quotes and semicolon
set PRODUCTVER=%PRODUCTVER:"=%
set PRODUCTVER=%PRODUCTVER:;=%
echo CUETools version: %PRODUCTVER%

SET release_dir=%base_dir%\bin\Release\CUETools_%PRODUCTVER%
SET win_dir=%base_dir%\bin\Release\net10.0-windows
SET cli_dir=%base_dir%\bin\Release\net10.0
SET lib_dir=%base_dir%\bin\Release\netstandard2.1

if not exist "%release_dir%" mkdir "%release_dir%"

REM use xcopy instead of copy. xcopy creates directories if necessary and outputs the copied file.
REM /Y Suppresses prompting to confirm that you want to overwrite an existing destination file.
REM /D xcopy copies all Source files that are newer than existing Destination files.
REM /I treats the destination as a directory and avoids interactive prompts in batch mode.

REM Windows applications and their runtime metadata.
xcopy /Y /D /I "%win_dir%\*.exe" "%release_dir%\"
xcopy /Y /D /I "%win_dir%\*.dll" "%release_dir%\"
xcopy /Y /D /I "%win_dir%\*.deps.json" "%release_dir%\"
xcopy /Y /D /I "%win_dir%\*.runtimeconfig.json" "%release_dir%\"
xcopy /Y /D /I "%win_dir%\*.config" "%release_dir%\"
xcopy /Y /D /I "%win_dir%\de-DE\*" "%release_dir%\de-DE\"
xcopy /Y /D /I "%win_dir%\ru-RU\*" "%release_dir%\ru-RU\"

REM Console tools and their runtime metadata.
xcopy /Y /D /I "%cli_dir%\*.exe" "%release_dir%\"
xcopy /Y /D /I "%cli_dir%\*.dll" "%release_dir%\"
xcopy /Y /D /I "%cli_dir%\*.deps.json" "%release_dir%\"
xcopy /Y /D /I "%cli_dir%\*.runtimeconfig.json" "%release_dir%\"
xcopy /Y /D /I "%cli_dir%\*.config" "%release_dir%\"

xcopy /Y /D /I "%base_dir%\License.txt" "%release_dir%\"
xcopy /Y /D /I "%base_dir%\CUETools\user_profiles_enabled" "%release_dir%\"

REM Managed plugins. Most plugins are libraries under netstandard2.1; FLACCL remains Windows-only.
xcopy /Y /D /I /S "%lib_dir%\plugins\*.dll" "%release_dir%\plugins\"
xcopy /Y /D /I /S "%lib_dir%\plugins\*.deps.json" "%release_dir%\plugins\"
xcopy /Y /D /I /S "%win_dir%\plugins\*.dll" "%release_dir%\plugins\"
xcopy /Y /D /I /S "%win_dir%\plugins\*.deps.json" "%release_dir%\plugins\"
xcopy /Y /D /I /S "%win_dir%\plugins\*.cl" "%release_dir%\plugins\"

REM ThirdParty
IF EXIST "%base_dir%\ThirdParty\ICSharpCode.SharpZipLib.dll" (
    xcopy /Y /D /I "%base_dir%\ThirdParty\ICSharpCode.SharpZipLib.dll" "%release_dir%\plugins\"
) ELSE echo WARNING: Missing %base_dir%\ThirdParty\ICSharpCode.SharpZipLib.dll

REM ThirdParty\Win32 plugins
IF EXIST "%base_dir%\ThirdParty\Win32\hdcd.dll" (
    xcopy /Y /D /I "%base_dir%\ThirdParty\Win32\hdcd.dll" "%release_dir%\plugins\win32\"
) ELSE echo WARNING: Missing %base_dir%\ThirdParty\Win32\hdcd.dll
IF EXIST "%base_dir%\ThirdParty\Win32\libFLAC_dynamic.dll" (
    xcopy /Y /D /I "%base_dir%\ThirdParty\Win32\libFLAC_dynamic.dll" "%release_dir%\plugins\win32\"
) ELSE echo WARNING: Missing %base_dir%\ThirdParty\Win32\libFLAC_dynamic.dll
IF EXIST "%base_dir%\ThirdParty\Win32\libmp3lame.dll" (
    xcopy /Y /D /I "%base_dir%\ThirdParty\Win32\libmp3lame.dll" "%release_dir%\plugins\win32\"
) ELSE echo WARNING: Missing %base_dir%\ThirdParty\Win32\libmp3lame.dll
IF EXIST "%base_dir%\ThirdParty\Win32\MACLibDll.dll" (
    xcopy /Y /D /I "%base_dir%\ThirdParty\Win32\MACLibDll.dll" "%release_dir%\plugins\win32\"
) ELSE echo WARNING: Missing %base_dir%\ThirdParty\Win32\MACLibDll.dll
IF EXIST "%base_dir%\ThirdParty\Win32\unrar.dll" (
    xcopy /Y /D /I "%base_dir%\ThirdParty\Win32\unrar.dll" "%release_dir%\plugins\win32\"
) ELSE echo WARNING: Missing %base_dir%\ThirdParty\Win32\unrar.dll
IF EXIST "%base_dir%\ThirdParty\Win32\wavpackdll.dll" (
    xcopy /Y /D /I "%base_dir%\ThirdParty\Win32\wavpackdll.dll" "%release_dir%\plugins\win32\"
) ELSE echo WARNING: Missing %base_dir%\ThirdParty\Win32\wavpackdll.dll

REM ThirdParty\x64 plugins
IF EXIST "%base_dir%\ThirdParty\x64\hdcd.dll" (
    xcopy /Y /D /I "%base_dir%\ThirdParty\x64\hdcd.dll" "%release_dir%\plugins\x64\"
) ELSE echo WARNING: Missing %base_dir%\ThirdParty\x64\hdcd.dll
IF EXIST "%base_dir%\ThirdParty\x64\libFLAC_dynamic.dll" (
    xcopy /Y /D /I "%base_dir%\ThirdParty\x64\libFLAC_dynamic.dll" "%release_dir%\plugins\x64\"
) ELSE echo WARNING: Missing %base_dir%\ThirdParty\x64\libFLAC_dynamic.dll
IF EXIST "%base_dir%\ThirdParty\x64\libmp3lame.dll" (
    xcopy /Y /D /I "%base_dir%\ThirdParty\x64\libmp3lame.dll" "%release_dir%\plugins\x64\"
) ELSE echo WARNING: Missing %base_dir%\ThirdParty\x64\libmp3lame.dll
IF EXIST "%base_dir%\ThirdParty\x64\MACLibDll.dll" (
    xcopy /Y /D /I "%base_dir%\ThirdParty\x64\MACLibDll.dll" "%release_dir%\plugins\x64\"
) ELSE echo WARNING: Missing %base_dir%\ThirdParty\x64\MACLibDll.dll
IF EXIST "%base_dir%\ThirdParty\x64\Unrar.dll" (
    xcopy /Y /D /I "%base_dir%\ThirdParty\x64\Unrar.dll" "%release_dir%\plugins\x64\"
) ELSE echo WARNING: Missing %base_dir%\ThirdParty\x64\Unrar.dll
IF EXIST "%base_dir%\ThirdParty\x64\wavpackdll.dll" (
    xcopy /Y /D /I "%base_dir%\ThirdParty\x64\wavpackdll.dll" "%release_dir%\plugins\x64\"
) ELSE echo WARNING: Missing %base_dir%\ThirdParty\x64\wavpackdll.dll

REM EAC Plugin
xcopy /Y /D /I "%win_dir%\interop\EAC\*.dll" "%release_dir%\interop\EAC\"
xcopy /Y /D /I "%win_dir%\interop\EAC\*.deps.json" "%release_dir%\interop\EAC\"
xcopy /Y /D /I "%win_dir%\interop\EAC\*.config" "%release_dir%\interop\EAC\"
xcopy /Y /D /I "%win_dir%\Newtonsoft.Json.dll" "%release_dir%\interop\EAC\"

popd
