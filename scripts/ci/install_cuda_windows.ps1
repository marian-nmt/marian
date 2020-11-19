## ------------------------------------------------------------------
## Install CUDA version on Windows via a network CUDA installer checking
## Visual Studio dependencies.
##
## Example usage:
##   .\install_cuda_windows.ps1 "10.2"
##
## Based on: https://github.com/ptheywood/cuda-cmake-github-actions
## Changes:
##   - Changed format of input arguments
##   - Do not check compatibility with Visual Studio
## ------------------------------------------------------------------

Param(
    [Parameter(Mandatory=$true)]
    $cuda
)

## ------------------------------------------------------------------
## Constants
## ------------------------------------------------------------------

# Dictionary of known cuda versions and thier download URLS, which do not follow a consistent pattern :(
$CUDA_KNOWN_URLS = @{
    "8.0" = "http://developer.nvidia.com/compute/cuda/8.0/Prod2/network_installers/cuda_8.0.61_win10_network-exe";
    "9.0" = "http://developer.nvidia.com/compute/cuda/9.0/Prod/network_installers/cuda_9.0.176_win10_network-exe";
    # CUDA 9.1 is removed from supported CUDA versions because its nvcc package is named differently
    "9.2" = "http://developer.nvidia.com/compute/cuda/9.2/Prod2/network_installers2/cuda_9.2.148_win10_network";
    "10.0" = "http://developer.nvidia.com/compute/cuda/10.0/Prod/network_installers/cuda_10.0.130_win10_network";
    "10.1" = "http://developer.download.nvidia.com/compute/cuda/10.1/Prod/network_installers/cuda_10.1.243_win10_network.exe";
    "10.2" = "http://developer.download.nvidia.com/compute/cuda/10.2/Prod/network_installers/cuda_10.2.89_win10_network.exe";
    "11.0" = "http://developer.download.nvidia.com/compute/cuda/11.0.1/network_installers/cuda_11.0.1_win10_network.exe"
}

## ------------------------------------------------------------------
## Select CUDA version
## ------------------------------------------------------------------

# Get the cuda version from the argument
$CUDA_VERSION_FULL = $cuda

# Validate CUDA version, extracting components via regex
$cuda_ver_matched = $CUDA_VERSION_FULL -match "^(?<major>[1-9][0-9]*)\.(?<minor>[0-9]+)$"
if(-not $cuda_ver_matched){
    Write-Output "Invalid CUDA version specified, <major>.<minor> required. Got '$CUDA_VERSION_FULL'."
    exit 1
}
$CUDA_MAJOR=$Matches.major
$CUDA_MINOR=$Matches.minor

echo "CUDA $($CUDA_MAJOR).$($CUDA_MINOR)"

## ------------------------------------------------------------------
## Select CUDA packages to install
## ------------------------------------------------------------------

# cuda_runtime.h is in nvcc <= 10.2, but cudart >= 11.0
# List of subpackages: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#install-cuda-software
$CUDA_PACKAGES_IN = @(
    "nvcc";
    "visual_studio_integration";
    "cublas_dev";
    "curand_dev";
    "cusparse_dev";
    "nvrtc_dev";
    "cudart";
)

$CUDA_PACKAGES = ""
Foreach ($package in $CUDA_PACKAGES_IN) {
    $CUDA_PACKAGES += " $($package)_$($CUDA_MAJOR).$($CUDA_MINOR)"
}
echo "CUDA packages: $($CUDA_PACKAGES)"

## ------------------------------------------------------------------
## Prepare download
## ------------------------------------------------------------------

# Select the download link if known, otherwise have a guess.
$CUDA_REPO_PKG_REMOTE=""
if($CUDA_KNOWN_URLS.containsKey($CUDA_VERSION_FULL)){
    $CUDA_REPO_PKG_REMOTE=$CUDA_KNOWN_URLS[$CUDA_VERSION_FULL]
} else{
    Write-Output "Error: URL for CUDA ${$CUDA_VERSION_FULL} not known"
    exit 1
}
$CUDA_REPO_PKG_LOCAL="cuda_$($CUDA_VERSION_FULL)_win10_network.exe"

## ------------------------------------------------------------------
## Install CUDA
## ------------------------------------------------------------------

# Get CUDA network installer
Write-Output "Downloading CUDA Network Installer for $($CUDA_VERSION_FULL) from: $($CUDA_REPO_PKG_REMOTE)"
Invoke-WebRequest $CUDA_REPO_PKG_REMOTE -OutFile $CUDA_REPO_PKG_LOCAL | Out-Null
if(Test-Path -Path $CUDA_REPO_PKG_LOCAL){
    Write-Output "Downloading Complete"
} else {
    Write-Output "Error: Failed to download $($CUDA_REPO_PKG_LOCAL) from $($CUDA_REPO_PKG_REMOTE)"
    exit 1
}

# Invoke silent install of CUDA (via network installer)
Write-Output "Installing CUDA $($CUDA_VERSION_FULL). Subpackages $($CUDA_PACKAGES)"
Start-Process -Wait -FilePath .\"$($CUDA_REPO_PKG_LOCAL)" -ArgumentList "-s $($CUDA_PACKAGES)"

# Check the return status of the CUDA installer.
if (!$?) {
    Write-Output "Error: CUDA installer reported error. $($LASTEXITCODE)"
    exit 1
}

## ------------------------------------------------------------------
## Set CUDA_PATH
## ------------------------------------------------------------------

# Store the CUDA_PATH in the environment for the current session, to be forwarded in the action.
$CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$($CUDA_MAJOR).$($CUDA_MINOR)"
# Set environmental variables in this session
$env:CUDA_PATH = "$($CUDA_PATH)"
Write-Output "CUDA_PATH= $($CUDA_PATH)"
