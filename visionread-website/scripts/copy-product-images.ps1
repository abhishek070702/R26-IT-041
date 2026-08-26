$ErrorActionPreference = "Stop"

$dest = Join-Path $PSScriptRoot "..\public\images"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

$candidates = @(
    (Join-Path $PSScriptRoot "..\public\images\product-headset.png"),
    (Join-Path $env:USERPROFILE "Downloads\*removebg*"),
    (Join-Path $env:USERPROFILE "Downloads\*01_05_07*"),
    (Join-Path $env:USERPROFILE "Desktop\*removebg*"),
    (Join-Path $env:USERPROFILE ".cursor\projects\c-Users-Abhishek-Desktop-Document-Type-Project-Final\assets\*removebg*")
)

$found = Get-ChildItem -Path $candidates -ErrorAction SilentlyContinue |
    Where-Object { -not $_.PSIsContainer }

foreach ($file in $found) {
    if ($file.Name -match "images-removebg" -and $file.Name -notmatch "11ages") {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $dest "product-headset.png") -Force
        Write-Host "Copied headset:" $file.FullName
    }
    if ($file.Name -match "11ages-removebg") {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $dest "product-worn.png") -Force
        Write-Host "Copied worn:" $file.FullName
    }
    if ($file.Name -match "12_43_45.*removebg" -or $file.Name -match "12_43_45_PM-removebg") {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $dest "logo-icon.png") -Force
        Write-Host "Copied logo icon:" $file.FullName
    }
    if ($file.Name -match "12_38_41") {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $dest "logo-wordmark.png") -Force
        Write-Host "Copied logo wordmark:" $file.FullName
    }
    if ($file.Name -match "01_05_07") {
        Copy-Item -LiteralPath $file.FullName -Destination (Join-Path $dest "demo-forbes-magazine.png") -Force
        Write-Host "Copied Forbes demo:" $file.FullName
    }
}

Get-ChildItem $dest
