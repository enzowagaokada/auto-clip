[CmdletBinding()]
param(
    [string]$Destination,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path (Split-Path $ScriptDirectory -Parent) "runtime"
}

# Keep this version aligned with github.com/yalue/onnxruntime_go v1.31.0,
# whose checked-in C API headers target ONNX Runtime 1.26.0.
$Version = "1.26.0"
$ArchiveName = "onnxruntime-win-x64-$Version.zip"
$Uri = "https://github.com/microsoft/onnxruntime/releases/download/v$Version/$ArchiveName"
$ExpectedSha256 = "6ebe99b5564bf4d029b6e93eac9ff423682b6212eade769e9ca3f685eaf500b4"
$DllPath = Join-Path $Destination "onnxruntime.dll"

if ((Test-Path $DllPath) -and -not $Force) {
    Write-Host "ONNX Runtime already exists at $DllPath (use -Force to replace it)."
    exit 0
}

$TemporaryDirectory = Join-Path ([System.IO.Path]::GetTempPath()) "auto-clip-onnxruntime-$([guid]::NewGuid())"
$ArchivePath = Join-Path $TemporaryDirectory $ArchiveName
$ExtractPath = Join-Path $TemporaryDirectory "extract"

try {
    New-Item -ItemType Directory -Path $TemporaryDirectory -Force | Out-Null
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Invoke-WebRequest -Uri $Uri -OutFile $ArchivePath

    $ActualSha256 = (Get-FileHash -Path $ArchivePath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualSha256 -ne $ExpectedSha256) {
        throw "SHA-256 mismatch for $ArchiveName. Expected $ExpectedSha256, got $ActualSha256."
    }

    Expand-Archive -Path $ArchivePath -DestinationPath $ExtractPath -Force
    $SourceDll = Join-Path $ExtractPath "onnxruntime-win-x64-$Version\lib\onnxruntime.dll"
    if (-not (Test-Path $SourceDll)) {
        throw "The verified archive did not contain the expected DLL: $SourceDll"
    }

    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    Copy-Item -Path $SourceDll -Destination $DllPath -Force
    Set-Content -Path (Join-Path $Destination "VERSION") -Value $Version -NoNewline
    Set-Content -Path (Join-Path $Destination "SHA256") -Value $ExpectedSha256 -NoNewline
    Write-Host "Installed ONNX Runtime $Version to $DllPath"
} finally {
    if (Test-Path $TemporaryDirectory) {
        Remove-Item -Path $TemporaryDirectory -Recurse -Force
    }
}
