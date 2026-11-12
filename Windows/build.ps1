param([Parameter(Mandatory = $true)][string]$DestinationDirectory)

$ErrorActionPreference = 'Stop'
$source = Split-Path -Parent $PSScriptRoot
$destination = [IO.Path]::GetFullPath($DestinationDirectory)
if (Test-Path -LiteralPath $destination) { throw 'Choose a new output directory.' }
if (-not (Test-Path -LiteralPath (Join-Path $source 'ThirdParty/taglib-sharp/src/TagLib/File.cs'))) {
    throw 'Initialize the pinned dependency: git submodule update --init ThirdParty/taglib-sharp'
}
# Keep the existing Windows .NET Framework host and native SCSI transport.
& dotnet build (Join-Path $source 'CUETools.Ripper.Console/CUETools.ConsoleRipper.csproj') -c Release -f net47 --nologo -p:DebugType=None "-p:PathMap=$source=/_/cuetools"
if ($LASTEXITCODE) { throw 'Windows CUE Ripper build failed.' }
& dotnet run --project (Join-Path $source 'Tests/OffsetCalibration/OffsetCalibration.Tests.csproj') -c Release
if ($LASTEXITCODE) { throw 'Windows offset calibration checks failed.' }
New-Item -ItemType Directory -Path $destination | Out-Null
# Only these two components changed. Retain the previously audited dependencies
# in Encore's runtime distribution; do not copy build/test/framework artifacts.
Copy-Item -LiteralPath (Join-Path $source 'bin/Release/net47/CUETools.Ripper.Console.exe') -Destination $destination
Copy-Item -LiteralPath (Join-Path $source 'bin/Release/plugins/net47/CUETools.Ripper.SCSI.dll') -Destination $destination
