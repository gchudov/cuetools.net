#define CUETOOLS_VERSION GetEnv('CUETOOLS_VERSION')
#if CUETOOLS_VERSION == ''
  #define CUETOOLS_VERSION '0.0.0'
#endif

#define APPNAME 'CUETools'

[Setup]
AppName={#APPNAME}
AppVersion={#CUETOOLS_VERSION}
WizardStyle=modern
DefaultDirName={autopf}\{#APPNAME}
DefaultGroupName={#APPNAME}
OutputDir=.
OutputBaseFilename={#APPNAME}_Setup_{#CUETOOLS_VERSION}
SetupIconFile=.\CUERipper.Avalonia\Assets\cue2.ico
UninstallDisplayIcon={app}\{#APPNAME}.exe
Compression=lzma2/ultra
SolidCompression=yes
LicenseFile=License.txt
PrivilegesRequiredOverridesAllowed=dialog

[Files]
Source: "bin\Release\CUETools_{#CUETOOLS_VERSION}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs
Source: "License.txt"; DestDir: "{app}"; Flags: onlyifdoesntexist

[Icons]
Name: "{group}\CUETools"; Filename: "{app}\CUETools.exe"; Tasks: starticon
Name: "{group}\CUERipper"; Filename: "{app}\CUERipper.Avalonia.exe"; Tasks: starticon
Name: "{group}\CUERipper (Classic)"; Filename: "{app}\CUERipper.exe"; Tasks: starticon
Name: "{autodesktop}\CUETools"; Filename: "{app}\CUETools.exe"; Tasks: desktopicon
Name: "{autodesktop}\CUERipper"; Filename: "{app}\CUERipper.Avalonia.exe"; Tasks: desktopicon
Name: "{group}\Uninstall CUETools"; Filename: "{uninstallexe}"

[Tasks]
Name: "starticon"; Description: "Create start menu shortcut"; GroupDescription: "Additional shortcuts:"
Name: "desktopicon"; Description: "Create desktop shortcut"; GroupDescription: "Additional shortcuts:"

[UninstallDelete]
Type: filesandordirs; Name: "{app}"