#!/bin/bash
set -euo pipefail

# -- Configuration --

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$SCRIPT_DIR/../core"
DEST_DIR="$SCRIPT_DIR/../../backend"
THRIFT_NAMESPACE="thrift_gen"  # Match this to your namespace prefix in .thrift files
THRIFT_GEN_DIR="$DEST_DIR/$THRIFT_NAMESPACE"

# -- Safety checks --

# Confirm directories exist
if [[ ! -d "$CORE_DIR" ]]; then
  echo "Error: core directory not found at: $CORE_DIR"
  exit 1
fi

if [[ ! -d "$DEST_DIR" ]]; then
  echo "Warning: backend directory does not exist at: $DEST_DIR"
  echo "Creating backend directory..."
  mkdir -p "$DEST_DIR"
fi

# Confirm before deleting thrift_gen folder
if [[ -d "$THRIFT_GEN_DIR" ]]; then
  echo "⚠️  About to remove existing generated folder: $THRIFT_GEN_DIR"
  read -p "Are you sure you want to delete this folder? (y/N) " confirm
  if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborting script. No files were changed."
    exit 0
  fi
  rm -rf "$THRIFT_GEN_DIR"
fi

# -- Thrift generation --

cd "$CORE_DIR"

echo "Generating Python code from .thrift files..."

for file in *.thrift; do
  echo "  Processing $file ..."
  # Generate directly into backend folder (will create thrift_gen inside)
  thrift --gen py -out "$DEST_DIR" "$file"
done

echo "✅ Thrift Python code successfully generated into $THRIFT_GEN_DIR"

# -- Done --
