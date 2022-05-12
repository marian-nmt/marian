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
## Bash functions
## ------------------------------------------------------------------

# returns 0 (true) if a >= b
function version_ge() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" == "$2" ]
}
# returns 0 (true) if a > b
function version_gt() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$1" = "$2" ] && return 1 || version_ge $1 $2
}
# returns 0 (true) if a <= b
function version_le() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$(printf '%s\n' "$@" | sort -V | head -n 1)" == "$1" ]
}
# returns 0 (true) if a < b
function version_lt() {
    [ "$#" != "2" ] && echo "${FUNCNAME[0]} requires exactly 2 arguments." && exit 1
    [ "$1" = "$2" ] && return 1 || version_le $1 $2
}


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

# Ideally choose from the list of meta-packages to minimise variance between cuda versions (although it does change too). Some of these packages may not be availble pre cuda 10.
CUDA_PACKAGES_IN=(
    "cuda-compiler"
    "cuda-cudart-dev"
    "cuda-nvtx"
    "cuda-nvrtc-dev"
    "libcublas-dev"
    "libcurand-dev" # 11-0+
    "libcusparse-dev" # 11-0+
    "cuda-cccl" # 11.4+, provides cub and thrust. On 11.3 knwon as cuda-thrust-11-3
)

CUDA_PACKAGES=""
for package in "${CUDA_PACKAGES_IN[@]}"; do
    # @todo This is not perfect. Should probably provide a separate list for diff versions
    # cuda-compiler-X-Y if CUDA >= 9.1 else cuda-nvcc-X-Y
    if [[ "${package}" == "cuda-nvcc" ]] && version_ge "$CUDA_VERSION_MAJOR_MINOR" "9.1" ; then
        package="cuda-compiler"
    elif [[ "${package}" == "cuda-compiler" ]] && version_lt "$CUDA_VERSION_MAJOR_MINOR" "9.1" ; then
        package="cuda-nvcc"
        # CUB/Thrust  are packages in cuda-thrust in 11.3, but cuda-cccl in 11.4+
    elif [[ "${package}" == "cuda-thrust" || "${package}" == "cuda-cccl" ]]; then
        # CUDA cuda-thrust >= 11.4
        if version_ge "$CUDA_VERSION_MAJOR_MINOR" "11.4" ; then
            package="cuda-cccl"
            # Use cuda-thrust > 11.2
        elif version_ge "$CUDA_VERSION_MAJOR_MINOR" "11.3" ; then
            package="cuda-thrust"
            # Do not include this pacakge < 11.3
        else
            continue
        fi
    fi
    # CUDA 11+ includes lib* / lib*-dev packages, which if they existed previously where cuda-cu*- / cuda-cu*-dev-
    if [[ ${package} == libcu* ]] && version_lt "$CUDA_VERSION_MAJOR_MINOR" "11.0" ; then
        if [[ ${package} != libcublas* ]]; then
            package="${package/libcu/cuda-cu}"
        fi
    fi

    if [[ ${package} == libcublas* ]] && version_lt "$CUDA_VERSION_MAJOR_MINOR" "11.0" ; then
        CUDA_PACKAGES+=" ${package}"
    else
        # Build the full package name and append to the string.
        CUDA_PACKAGES+=" ${package}-${CUDA_MAJOR}-${CUDA_MINOR}"
    fi
done

echo "CUDA_PACKAGES ${CUDA_PACKAGES}"

## ------------------------------------------------------------------
## Prepare to install
## ------------------------------------------------------------------

CPU_ARCH="x86_64"
PIN_FILENAME="cuda-ubuntu${UBUNTU_VERSION}.pin"
PIN_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/${CPU_ARCH}/${PIN_FILENAME}"
# apt keyring package now available https://developer.nvidia.com/blog/updating-the-cuda-linux-gpg-repository-key/
KERYRING_PACKAGE_FILENAME="cuda-keyring_1.0-1_all.deb"
KEYRING_PACKAGE_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/${CPU_ARCH}/${KERYRING_PACKAGE_FILENAME}"
REPO_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${UBUNTU_VERSION}/${CPU_ARCH}/"

echo "PIN_FILENAME ${PIN_FILENAME}"
echo "PIN_URL ${PIN_URL}"
echo "KEYRING_PACKAGE_URL ${KEYRING_PACKAGE_URL}"
echo "REPO_URL ${REPO_URL}"

## ------------------------------------------------------------------
## Install CUDA
## ------------------------------------------------------------------

echo "Adding CUDA Repository"
wget ${PIN_URL}
sudo mv ${PIN_FILENAME} /etc/apt/preferences.d/cuda-repository-pin-600
wget ${KEYRING_PACKAGE_URL} && sudo dpkg -i ${KERYRING_PACKAGE_FILENAME} && rm ${KERYRING_PACKAGE_FILENAME}
sudo add-apt-repository "deb ${REPO_URL} /"
sudo apt-get update

echo "Installing CUDA packages ${CUDA_PACKAGES}"
sudo apt-get -y --allow-unauthenticated install ${CUDA_PACKAGES}

if [[ $? -ne 0 ]]; then
    echo "CUDA Installation Error."
    exit 1
fi
