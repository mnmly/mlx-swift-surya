#!/usr/bin/env bash
# Build a static DocC site for MLXSurya into ./docs (GitHub Pages-ready).
#
# Uses `xcodebuild docbuild` + `docc process-archive transform-for-static-hosting` rather than
# the SwiftPM docc plugin: this package depends on mlx-swift, whose Metal shaders fail to compile
# under `swift package generate-documentation` in some toolchains. xcodebuild compiles them
# correctly.
#
# Env:
#   TARGETS            target to document (first word used). Default: MLXSurya.
#   HOSTING_BASE_PATH  repo name on GitHub Pages. Default: mlx-swift-surya.
#   OUTPUT_DIR         Default: docs.
#   DERIVED_DATA       Default: .xcdd-docs.
set -euo pipefail
cd "$(dirname "$0")/.."

# `Scripts/build_docs.sh`          → build the static site into ./docs
# `Scripts/build_docs.sh publish`  → build, then force-push it to the gh-pages branch (Pages source)
MODE="${1:-build}"
TARGET="${TARGETS:-MLXSurya}"; TARGET="${TARGET%% *}"
HOSTING_BASE_PATH="${HOSTING_BASE_PATH:-mlx-swift-surya}"
OUTPUT_DIR="${OUTPUT_DIR:-docs}"
DD="${DERIVED_DATA:-.xcdd-docs}"
slug="$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')"
REMOTE="$(git remote get-url origin 2>/dev/null || true)"

# Resolve packages first. On a fresh checkout (e.g. CI) `docbuild` otherwise fails at
# ComputeTargetDependencyGraph with "Supported platforms … is empty" because the package graph
# isn't resolved yet.
echo ">> resolve package dependencies"
xcodebuild -resolvePackageDependencies -scheme "$TARGET" -derivedDataPath "$DD" 2>&1 | tail -2

echo ">> docbuild $TARGET"
xcodebuild docbuild \
  -scheme "$TARGET" \
  -destination 'generic/platform=macOS' \
  -derivedDataPath "$DD" \
  | tail -2

ARCHIVE="$(find "$DD/Build/Products" -name "$TARGET.doccarchive" -maxdepth 3 2>/dev/null | head -1)"
[ -n "$ARCHIVE" ] || { echo "error: no $TARGET.doccarchive produced under $DD"; exit 1; }

rm -rf "$OUTPUT_DIR"; mkdir -p "$OUTPUT_DIR"
echo ">> transform-for-static-hosting → $OUTPUT_DIR/$TARGET"
xcrun docc process-archive transform-for-static-hosting "$ARCHIVE" \
  --output-path "$OUTPUT_DIR/$TARGET" \
  --hosting-base-path "$HOSTING_BASE_PATH/$TARGET"

# Redirect the Pages root to the target's documentation so the root URL doesn't 404.
cat > "$OUTPUT_DIR/index.html" <<HTML
<!doctype html>
<meta charset="utf-8">
<title>${HOSTING_BASE_PATH}</title>
<meta http-equiv="refresh" content="0; url=/${HOSTING_BASE_PATH}/${TARGET}/documentation/${slug}/">
<link rel="canonical" href="/${HOSTING_BASE_PATH}/${TARGET}/documentation/${slug}/">
<p>Redirecting to <a href="/${HOSTING_BASE_PATH}/${TARGET}/documentation/${slug}/">documentation</a>.</p>
HTML

echo "Docs written to $OUTPUT_DIR/$TARGET/. Root redirect: $OUTPUT_DIR/index.html"

# Optional: publish the static site to the gh-pages branch (GitHub Pages source). DocC emits
# files/dirs with leading underscores, so `.nojekyll` is required or Pages (Jekyll) drops them.
if [ "$MODE" = "publish" ]; then
    [ -n "$REMOTE" ] || { echo "error: no git remote 'origin' to publish to"; exit 1; }
    echo ">> publishing $OUTPUT_DIR → gh-pages on $REMOTE"
    tmp="$(mktemp -d)"
    cp -R "$OUTPUT_DIR/." "$tmp"/
    touch "$tmp/.nojekyll"
    (
        cd "$tmp"
        git init -q
        git checkout -q -b gh-pages
        git add -A
        git commit -qm "Publish DocC site"
        git remote add origin "$REMOTE"
        git -c http.postBuffer=524288000 push -fq origin gh-pages
    )
    rm -rf "$tmp"
    echo "Published to gh-pages → https://<owner>.github.io/${HOSTING_BASE_PATH}/"
fi
