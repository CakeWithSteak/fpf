# Build script for Windows
# Feeds kernel.cu to CL for preprocessing, then splits the result into 2000 char raw string literals,
# because MSVC can't handle a single literal being longer than that, but it can concatenate smaller
# literals together without issue.

param (
    [string]$compiler,
    [string]$builddir,
    [string]$srcdir,
    [string]$includes
)

$kerneli = Join-Path -Path $builddir -ChildPath "kernel.i"
$kernelii = Join-Path -Path $builddir -ChildPath "kernel.ii"
$kernelcu = Join-Path -Path $srcdir -ChildPath "Computation/kernel.cu"

Start-Process -FilePath $compiler -ArgumentList ("/P", "/EP", "/Fi`"$kerneli`"", "/I`"$includes`"", "/std:c++latest", "/DBUILD_FOR_NVRTC", "/nologo" , "`"$kernelcu`"") -NoNewWindow -Wait
$preprocessed = Get-Content -Raw -Path $kerneli
$fragments = New-Object System.Collections.Generic.List[System.String]
$tempstr = ""
for($i=0; $i -lt $preprocessed.Length; $i++)
{
    $tempstr += $preprocessed[$i]
    if($i % 2000 -eq 0)
    {
        $fragments.Add("R`"(" + $tempstr + ")`" ")
        $tempstr = ""
    }
}
$fragments.Add("R`"(" + $tempstr + ")`"")
Set-Content -Path $kernelii -Value ([string]::Join("", $fragments))