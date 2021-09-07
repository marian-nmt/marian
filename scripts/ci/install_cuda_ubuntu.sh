#!/usr/bin/env bash

## ------------------------------------------------------------------
## Install CUDA version on Ubuntu via a network CUDA installer.
## The first and only argument is CUDA MAJOR.MINOR version.
## Ubuntu version is automatically detected from lsb_release.
##
## Example usage:
##    bash install_cuda_ubuntu.sh 10.2
##
## Based on: https://github.com/ptheywood/cuda-cmake-github-actions
## ------------------------------------------------------------------

if [[ $# -lt 1 ]]; then
    echo "Illegal number of parameters."
    echo "Usage: $0 <CUDA_VERSION>"
    exit 2
fi

## ------------------------------------------------------------------
## Find CUDA and OS versions
## ------------------------------------------------------------------

# Get CUDA version.
CUDA_VERSION_MAJOR_MINOR=$1
CUDA_MAJOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f1)
CUDA_MINOR=$(echo "${CUDA_VERSION_MAJOR_MINOR}" | cut -d. -f2)
echo "CUDA_MAJOR: ${CUDA_MAJOR}"
echo "CUDA_MINOR: ${CUDA_MINOR}"

# If we don't know the CUDA_MAJOR or MINOR, error.
if [ -z "${CUDA_MAJOR}" ] ; then
    echo "Error: Unknown CUDA Major version. Aborting."
    exit 1
fi
if [ -z "${CUDA_MINOR}" ] ; then
    echo "Error: Unknown CUDA Minor version. Aborting."
    exit 1
fi

# Find the OS.
UBUNTU_VERSION=$(lsb_release -sr)
UBUNTU_VERSION="${UBUNTU_VERSION//.}"

echo "UBUNTU_VERSION: ${UBUNTU_VERSION}"
# If we don't know the Ubuntu version, error.
if [ -z ${UBUNTU_VERSION} ]; then
    echo "Error: Unknown Ubuntu version. Aborting."
    exit 1
fi

## ------------------------------------------------------------------
## Select CUDA packages to install
## ------------------------------------------------------------------

CUDA_PACKAGES_IN=(
    "command-line-tools"
    "libraries-dev"
)

CUDA_PACKAGES=""
for package in "${CUDA_PACKAGES_IN[@]}"; do
    # Build the full package name and append to the string.
    CUDA_PACKAGES+=" cuda-${package}-${CUDA_MAJOR}-${CUDA_MINOR}"
done

echo "CUDA_PACKAGES ${CUDA_PACKAGES}"

## ------------------------------------------------------------------
## Prepare to install
## ------------------------------------------------------------------

PIN_FILENAME="cuda-ubuntu${UBUNTU_VERSION}.pin"
PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/${PIN_FILENAME}"
APT_KEY_URL="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/7fa2af80.pub"
REPO_URL="http://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/x86_64/"

echo "PIN_FILENAME ${PIN_FILENAME}"
echo "PIN_URL ${PIN_URL}"
echo "APT_KEY_URL ${APT_KEY_URL}"

## ------------------------------------------------------------------
## Install CUDA
## ------------------------------------------------------------------

echo "Adding CUDA Repository"
wget ${PIN_URL}
sudo mv ${PIN_FILENAME} /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys ${APT_KEY_URL}
sudo add-apt-repository "deb ${REPO_URL} /"
sudo apt-get update

echo "Installing CUDA packages ${CUDA_PACKAGES}"
sudo apt-get -y --allow-unauthenticated install ${CUDA_PACKAGES}

if [[ $? -ne 0 ]]; then
    echo "CUDA Installation Error."
    exit 1
fi
