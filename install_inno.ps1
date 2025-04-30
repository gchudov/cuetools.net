$ErrorActionPreference = "Stop"

$innoUri = "https://files.jrsoftware.org/is/6/innosetup-6.4.2.exe"
$executable = "innosetup-6.4.2.exe"
$expectedHash = "238e2cf82c212a3879a050e02d787283c54bcb72d5cb6070830942de56627d5b"

Invoke-WebRequest -Uri $innoUri -OutFile $executable

$actualHash = (Get-FileHash $executable -Algorithm SHA256).Hash.ToLower()
if ($actualHash -ne $expectedHash) {
	Write-Host "Hash mismatch! Executable is modified."
	exit 1
}

Start-Process -FilePath $executable -ArgumentList "/SILENT" -Wait
