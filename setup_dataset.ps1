# ---------------------------------------------
#  Dance Style Dataset Setup (364 images, 8 classes)
# ---------------------------------------------
Set-Location -Path (Split-Path -Parent $MyInvocation.MyCommand.Path)

# Clean previous data
Remove-Item -Recurse -Force dataset,dataset_tmp,dataset_zip -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force dataset | Out-Null
New-Item -ItemType Directory -Force dataset_zip | Out-Null
New-Item -ItemType Directory -Force dataset_tmp | Out-Null

# Download proper dataset
Write-Host ">> Downloading 'indian-dance-form-classification' dataset..."
kaggle datasets download -d aditya48/indian-dance-form-classification -p dataset_zip

# Unzip
Write-Host ">> Unzipping..."
Get-ChildItem dataset_zip -Filter *.zip | ForEach-Object {
    Expand-Archive -Path $_.FullName -DestinationPath dataset_tmp -Force
}

# Copy subdirectories into dataset/
Write-Host ">> Organizing class folders..."
Get-ChildItem dataset_tmp -Recurse -Directory | ForEach-Object {
    if ($_.Name -match '^(Bharatanatyam|Kathak|Kathakali|Kuchipudi|Manipuri|Mohiniyattam|Odissi|Sattriya)$') {
        Copy-Item -Path $_.FullName -Destination (Join-Path "dataset" $_.Name) -Recurse -Force
        Write-Host "  Copied:" $_.Name
    }
}

# Show final counts
Write-Host "`n>> Final dataset structure:"
Get-ChildItem dataset -Recurse -File -Include *.jpg,*.jpeg,*.png,*.bmp |
  Group-Object Directory | Select-Object Count, Name | Format-Table -AutoSize
