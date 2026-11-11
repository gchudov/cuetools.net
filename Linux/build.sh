#!/usr/bin/env bash
set -euo pipefail
source_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_root="${CUERIPPER_BUILD_ROOT:-$source_dir/build}"
test -f "$source_dir/../ThirdParty/taglib-sharp/src/TagLib/File.cs" || {
    echo 'Initialize the pinned TagLib dependency: git submodule update --init ThirdParty/taglib-sharp' >&2
    exit 1
}
dotnet build "$source_dir/CUERipper.csproj" -c Release -p:CueRipperBuildRoot="$build_root"
if [[ "${1:-}" == --test ]]; then
    cc "$source_dir/Tests/abi.c" -o "$build_root/abi"
    "$build_root/abi"
    dotnet run --project "$source_dir/Tests/CUERipper.Tests.csproj" -c Release -p:CueRipperBuildRoot="$build_root"
    helper="$build_root/bin/CUERipper/Release/net10.0/CUERipper.dll"
    dotnet "$helper" --help >/dev/null
    dotnet "$helper" --list-drives
    set +e
    dotnet "$helper" --drive /dev/null --info >"$build_root/invalid-device.log" 2>&1
    result=$?
    set -e
    [[ "$result" == 1 ]] || { cat "$build_root/invalid-device.log"; exit 1; }
elif [[ "${1:-}" == --publish ]]; then
    destination="${2:?Provide a publish directory}"
    dotnet publish "$source_dir/CUERipper.csproj" -c Release -r linux-x64 --self-contained true \
        -p:CueRipperBuildRoot="$build_root" -p:PublishSingleFile=true -p:IncludeNativeLibrariesForSelfExtract=true \
        -p:DebugType=None -o "$destination"
fi
