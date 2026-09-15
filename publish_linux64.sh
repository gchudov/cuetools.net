#!/bin/bash
set -euo pipefail

PUBLISH_BASE="./bin/Publish/linux-x64/CUERipper.Avalonia"
NATIVE_PLUGIN_BASE="$PUBLISH_BASE/plugins/x64"

mkdir -p "$NATIVE_PLUGIN_BASE"

# Compile libFLAC
pushd ./ThirdParty/flac
./autogen.sh
./configure CFLAGS="-O2 -march=x86-64 -mtune=generic" CXXFLAGS="-O2 -march=x86-64 -mtune=generic"
make
popd
cp -L ./ThirdParty/flac/src/libFLAC/.libs/libFLAC.so "$NATIVE_PLUGIN_BASE"
mv "$NATIVE_PLUGIN_BASE/libFLAC.so" "$NATIVE_PLUGIN_BASE/libFLAC_dynamic.so"

# Compile LAME
mkdir -p ./ThirdParty/lame
tar -xz --file=./ThirdParty/lame-3.100.tar.gz --directory=./ThirdParty/lame --strip-components=1
pushd ./ThirdParty/lame
./configure CFLAGS="-O2 -march=x86-64 -mtune=generic" CXXFLAGS="-O2 -march=x86-64 -mtune=generic"
make
popd
cp -L ./ThirdParty/lame/libmp3lame/.libs/libmp3lame.so "$NATIVE_PLUGIN_BASE"

find . -name "*.csproj" | while read -r csproj; do
    if [[ "$csproj" == *"CUETools.Codecs.lame_enc"* ]]; then
        echo "Skipping $csproj, might be a leftover project?" 
        continue
    fi

    if grep -q '<TargetFramework.*netstandard2\.0' "$csproj"; then
        echo "Checking $csproj"

        if grep -q '<OutputPath>.*plugins' "$csproj"; then
            echo "Plugin"
            output="$PUBLISH_BASE/plugins"
        else
            echo "Non plugin"
            output="$PUBLISH_BASE"
        fi

        echo "Publishing $csproj to $output"
        dotnet publish "$csproj" -f netstandard2.0 -c Release -r linux-x64 -o "$output" /p:DebugType=None /p:DebugSymbols=false
    fi
done

dotnet publish ./CUERipper.Avalonia/CUERipper.Avalonia.csproj -f net8.0 -c Release -r linux-x64 -o "$PUBLISH_BASE" /p:DebugType=None /p:DebugSymbols=false --self-contained true

