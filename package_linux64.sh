#!/bin/bash
set -euo pipefail

if [[ -z "${CUETOOLS_VERSION:-}" ]]; then
  echo "CUETOOLS_VERSION hasn't been set."
  exit 1
fi

if [[ ! -d ./bin/Publish/linux-x64 ]]; then
  echo "Where's the published directory?"
  exit 1
fi

OUTPUT_NAME="CUERipper.Linux64_${CUETOOLS_VERSION}"
TAR_FILE="${OUTPUT_NAME}.tar.gz"
SHA_FILE="${OUTPUT_NAME}.tar.gz.sha256"

tar -czf "$TAR_FILE" -C ./bin/Publish/linux-x64 .

HASH=$(sha256sum "$TAR_FILE" | awk '{ print $1 }')
echo "${HASH} *${TAR_FILE}" > "$SHA_FILE"


