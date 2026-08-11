@ECHO OFF
REM Collect native ThirdParty .dll and .pdb files for debugging CUETools.
REM Wolfgang Stoeggl <c72578@yahoo.de>, 2020-2026.

REM The script is located in the CUETools repository root.
echo %~dp0
pushd %~dp0
SET base_dir=.
SET debug_win_dir=%base_dir%\bin\Debug\net10.0-windows
SET debug_lib_dir=%base_dir%\bin\Debug\netstandard2.1

REM use xcopy instead of copy. xcopy creates directories if necessary and outputs the copied file.
REM /Y Suppresses prompting to confirm that you want to overwrite an existing destination file.
REM /D xcopy copies all Source files that are newer than existing Destination files.

call :copy_native "%debug_win_dir%"
call :copy_native "%debug_lib_dir%"

popd
goto :eof

:copy_native
SET target_dir=%~1

REM ThirdParty
call :copy_file "%base_dir%\ThirdParty\ICSharpCode.SharpZipLib.dll" "%target_dir%\plugins\"

REM ThirdParty\Win32 plugins
call :copy_file "%base_dir%\ThirdParty\Win32\hdcd.dll" "%target_dir%\plugins\win32\"
call :copy_file "%base_dir%\ThirdPartyDebug\Win32\libFLAC_dynamic.dll" "%target_dir%\plugins\win32\"
call :copy_file "%base_dir%\ThirdPartyDebug\Win32\libFLAC_dynamic.pdb" "%target_dir%\plugins\win32\"
call :copy_file "%base_dir%\ThirdParty\Win32\libmp3lame.dll" "%target_dir%\plugins\win32\"
call :copy_file "%base_dir%\ThirdPartyDebug\Win32\MACLibDll.dll" "%target_dir%\plugins\win32\"
call :copy_file "%base_dir%\ThirdPartyDebug\Win32\MACLibDll.pdb" "%target_dir%\plugins\win32\"
call :copy_file "%base_dir%\ThirdParty\Win32\unrar.dll" "%target_dir%\plugins\win32\"
call :copy_file "%base_dir%\ThirdPartyDebug\Win32\wavpackdll.dll" "%target_dir%\plugins\win32\"
call :copy_file "%base_dir%\ThirdPartyDebug\Win32\wavpackdll.pdb" "%target_dir%\plugins\win32\"

REM ThirdParty\x64 plugins
call :copy_file "%base_dir%\ThirdParty\x64\hdcd.dll" "%target_dir%\plugins\x64\"
call :copy_file "%base_dir%\ThirdPartyDebug\x64\libFLAC_dynamic.dll" "%target_dir%\plugins\x64\"
call :copy_file "%base_dir%\ThirdPartyDebug\x64\libFLAC_dynamic.pdb" "%target_dir%\plugins\x64\"
call :copy_file "%base_dir%\ThirdParty\x64\libmp3lame.dll" "%target_dir%\plugins\x64\"
call :copy_file "%base_dir%\ThirdPartyDebug\x64\MACLibDll.dll" "%target_dir%\plugins\x64\"
call :copy_file "%base_dir%\ThirdPartyDebug\x64\MACLibDll.pdb" "%target_dir%\plugins\x64\"
call :copy_file "%base_dir%\ThirdParty\x64\Unrar.dll" "%target_dir%\plugins\x64\"
call :copy_file "%base_dir%\ThirdPartyDebug\x64\wavpackdll.dll" "%target_dir%\plugins\x64\"
call :copy_file "%base_dir%\ThirdPartyDebug\x64\wavpackdll.pdb" "%target_dir%\plugins\x64\"
exit /b 0

:copy_file
if exist "%~1" (
    xcopy /Y /D /I "%~1" "%~2"
) else (
    echo WARNING: Missing %~1
)
exit /b 0
