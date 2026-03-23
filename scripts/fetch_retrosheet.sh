#!/bin/bash
# scripts/fetch_retrosheet.sh
# Fetch Retrosheet event files for a given year.
# Usage: ./scripts/fetch_retrosheet.sh [year]

YEAR=${1:-2023}
BASE_URL="https://www.retrosheet.org/events/"
ZIP_FILE="${YEAR}eve.zip"
TARGET_DIR="data/raw/retrosheet/${YEAR}"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "Fetching Retrosheet data for $YEAR..."
curl -L -o "$TARGET_DIR/$ZIP_FILE" "$BASE_URL$ZIP_FILE"

if [ $? -eq 0 ]; then
    echo "Successfully downloaded $ZIP_FILE to $TARGET_DIR. Unzipping..."
    unzip -o "$TARGET_DIR/$ZIP_FILE" -d "$TARGET_DIR"
    # rm "$TARGET_DIR/$ZIP_FILE" # Optional: keep zip or remove it
    echo "Done."
else
    echo "Error: Failed to download $ZIP_FILE from $BASE_URL."
    exit 1
fi
