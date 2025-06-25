$ErrorActionPreference = "Stop"

$cueToolsDirectories = Get-ChildItem -Path "bin/Release/CUETools_*" -Directory | 
    Where-Object { $_.Name -match "CUETools_(\d+\.\d+\.\d+[a-zA-Z]?)" } |
    Sort-Object -Descending	

if (-not $cueToolsDirectories) {
    Write-Error "No CUETools_* directories found!"
}

$cueToolsDir = $cueToolsDirectories[0]
$version = $cueToolsDir.Name -replace "CUETools_", ""

Write-Output "Detected CUETools version: $version"

$eacZip = "CUETools.CTDB.EACPlugin.$version.zip"
$cuetoolsZip = "CUETools_$version.zip"
$cuetoolsLiteZip = "CUETools.Lite_$version.zip"
$cueRipperLinuxZip = "CUERipper.Linux64_$version.zip"

Write-Output "Creating EAC Zip"
Compress-Archive -Path bin/Release/$($cueToolsDir.Name)/interop/ -DestinationPath $eacZip -Force

Write-Output "Creating CUETools Zip"
Compress-Archive -Path bin/Release/$($cueToolsDir.Name)/ -DestinationPath $cuetoolsZip -Force

$cuetoolsLiteFolder = "bin/Release/CUETools.Lite_$version/"
if (Test-Path $cuetoolsLiteFolder) {
    Write-Output "Creating CUETools Lite Zip"
    Compress-Archive -Path $cuetoolsLiteFolder -DestinationPath $cuetoolsLiteZip -Force
} else {
    Write-Output "CUETools.Lite_$version not found, skip packaging."
}

Write-Output "Creating CUERipper for Linux Zip"
Compress-Archive -Path bin/Publish/linux-x64/CUERipper.Avalonia/ -DestinationPath $cueRipperLinuxZip -Force

# Generate a hashfile using the same format as previous releases
Write-Output "Generating SHA256 hashes..."
$eacHash = (Get-FileHash $eacZip -Algorithm SHA256).Hash.ToLower()
"$eacHash *$eacZip" | Out-File -Encoding ASCII "$eacZip.sha256"

$cuetoolsHash = (Get-FileHash $cuetoolsZip -Algorithm SHA256).Hash.ToLower()
"$cuetoolsHash *$cuetoolsZip" | Out-File -Encoding ASCII "$cuetoolsZip.sha256"

if (Test-Path $cuetoolsLiteFolder) {
    $cuetoolsLiteHash = (Get-FileHash $cuetoolsLiteZip -Algorithm SHA256).Hash.ToLower()
    "$cuetoolsLiteHash *$cuetoolsLiteZip" | Out-File -Encoding ASCII "$cuetoolsLiteZip.sha256"
}

$cueripperLinuxHash = (Get-FileHash $cueRipperLinuxZip -Algorithm SHA256).Hash.ToLower()
"$cueripperLinuxHash *$cueRipperLinuxZip" | Out-File -Encoding ASCII "$cueRipperLinuxZip.sha256"

Write-Output "Packaging complete."
