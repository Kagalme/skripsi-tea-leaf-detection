Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$zipPath = "d:\SKRIPSI\phase3_scripts.zip"
if (Test-Path $zipPath) {
    Remove-Item $zipPath -Force
}

$zip = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
$baseDir = "d:\SKRIPSI"
$targetDir = "d:\SKRIPSI\phase3_scripts"

$files = [System.IO.Directory]::GetFiles($targetDir, "*", [System.IO.SearchOption]::AllDirectories)
foreach ($file in $files) {
    $rel = $file.Substring($baseDir.Length).TrimStart('\', '/').Replace('\', '/')
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($zip, $file, $rel)
    Write-Host "Added: $rel"
}
$zip.Dispose()
Write-Host "Zip created with POSIX forward slashes! Size: $((Get-Item $zipPath).Length) bytes"
