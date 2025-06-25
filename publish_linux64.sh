#!/bin/bash
PUBLISH_BASE="./bin/Publish/linux-x64/CUERipper.Avalonia"

find . -name "*.csproj" | while read -r csproj; do
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
        dotnet publish "$csproj" -f netstandard2.0 -c Release -r linux-x64 -o "$output"
    fi
done

dotnet publish ./CUERipper.Avalonia/CUERipper.Avalonia.csproj -f net8.0 -c Release -r linux-x64 -o "$PUBLISH_BASE"
