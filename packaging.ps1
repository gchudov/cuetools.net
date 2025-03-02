$ErrorActionPreference = "Stop"

$cueToolsDirectories = Get-ChildItem -Path "bin/Release/CUETools_*" -Directory | 
    Where-Object { $_.Name -match "CUETools_(\d+\.\d+\.\d+)" } |
    Sort-Object { [version]($matches[1]) } -Descending

if (-not $cueToolsDirectories) {
    Write-Error "No CUETools_* directories found!"
}

$cueToolsDir = $cueToolsDirectories[0]
$version = $cueToolsDir.Name -replace "CUETools_", ""

Write-Output "Detected CUETools version: $version"

$eacZip = "CUETools.CTDB.EACPlugin.$version.zip"
$cuetoolsZip = "CUETools_$version.zip"

Write-Output "Creating EAC Zip"
Compress-Archive -Path bin/Release/$($cueToolsDir.Name)/interop/ -DestinationPath $eacZip -Force

Write-Output "Creating CUETools Zip"
Compress-Archive -Path bin/Release/$($cueToolsDir.Name)/ -DestinationPath $cuetoolsZip -Force

# Generate a hashfile using the same format as previous releases
Write-Output "Generating SHA256 hashes..."
$eacHash = (Get-FileHash $eacZip -Algorithm SHA256).Hash.ToLower()
"$eacHash *$eacZip" | Out-File -Encoding ASCII "$eacZip.sha256"

$cuetoolsHash = (Get-FileHash $cuetoolsZip -Algorithm SHA256).Hash.ToLower()
"$cuetoolsHash *$cuetoolsZip" | Out-File -Encoding ASCII "$cuetoolsZip.sha256"

Write-Output "Packaging complete."