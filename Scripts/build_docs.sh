#!/usr/bin/env bash
# Build static DocC site(s) for one or more SPM targets into ./docs
# (GitHub Pages-ready). Modeled after ml-explore/mlx-swift's
# tools/build-documentation.sh, with optional LLM-Markdown export.
#
# Usage:
#   Scripts/build_docs.sh                # build all $TARGETS into ./docs
#   Scripts/build_docs.sh preview        # local preview (first target only)
#   Scripts/build_docs.sh -f             # bypass gh-pages branch guard
#
# Required-ish env (edit defaults below or pass at call site):
#   TARGETS              Space-separated target names. Single-target packages
#                        can just set one. Default: MLXSurya.
#   HOSTING_BASE_PATH    Repo name on GitHub Pages (e.g. "my-lib"). Each
#                        target is hosted at <BASE>/<Target>/ so source URLs
#                        and asset paths resolve correctly.
#   REPO_URL             https URL to the GitHub repo (no trailing slash).
#                        Enables "View on GitHub" source links per symbol.
#   REPO_BRANCH          Branch the source links point at. Default: main.
#
# Optional env:
#   OUTPUT_DIR           Default: docs
#   REQUIRE_GH_PAGES=1   Refuse to build off the gh-pages branch unless -f.
#   EMIT_MARKDOWN=1      Pass the experimental Markdown-output flags
#                        (--enable-experimental-markdown-output and
#                         --enable-experimental-markdown-output-manifest) so docc
#                        emits per-symbol .md files (+ a manifest) under
#                        <out>/<target>/. Requires a recent swift-docc — the
#                        Xcode-bundled docc lacks these and they are skipped with
#                        a warning.
#   EMIT_LLMS_TXT=1      Above + concatenate the .md into <OUTPUT_DIR>/llms.txt.
#
# Toolchain note: this uses `swift package generate-documentation` (the
# SwiftDocC plugin), which needs a Swift toolchain that matches the active SDK.
# CI builds on a `macos-*` runner whose default Xcode supplies that. Locally, a
# mismatched bare `swift` (e.g. a swiftly toolchain older than the Xcode SDK)
# fails to resolve — select a matching one with `swiftly use` / `TOOLCHAIN=…`.
set -euo pipefail

cd "$(dirname "$0")/.."

TARGETS="${TARGETS:-MLXSurya}"
HOSTING_BASE_PATH="${HOSTING_BASE_PATH:-mlx-swift-surya}"
REPO_URL="${REPO_URL:-https://github.com/mnmly/mlx-swift-surya}"
REPO_BRANCH="${REPO_BRANCH:-main}"
OUTPUT_DIR="${OUTPUT_DIR:-docs}"

# Convenience: build with a specific toolchain (e.g. a swift.org snapshot that has
# the experimental Markdown flags). Accepts a toolchain identifier or the alias
# "swift" (latest installed swift.org toolchain). `TOOLCHAINS` is honored too.
#   TOOLCHAIN=swift Scripts/build_docs.sh
# If swiftly is installed, its env (and selected toolchain on PATH) is sourced
# automatically so plain `Scripts/build_docs.sh` uses your `swiftly use` choice.
if [[ -z "${TOOLCHAINS:-}" && -n "${TOOLCHAIN:-}" ]]; then
    export TOOLCHAINS="$TOOLCHAIN"
fi
if [[ -z "${TOOLCHAINS:-}" && -f "$HOME/.swiftly/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/.swiftly/env.sh"
fi

FORCE=0
MODE="build"
for arg in "$@"; do
    case "$arg" in
        -f|--force) FORCE=1 ;;
        preview)    MODE="preview" ;;
    esac
done

if [[ "${REQUIRE_GH_PAGES:-0}" == "1" && "$MODE" == "build" && $FORCE -eq 0 ]]; then
    branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo)"
    if [[ "$branch" != "gh-pages" ]]; then
        echo "Refusing to build off branch '$branch'. Use -f to override."
        exit 1
    fi
fi

export DOCC_JSON_PRETTYPRINT=YES

# Preview: first target only — `swift package preview-documentation` is
# single-target and interactive.
if [[ "$MODE" == "preview" ]]; then
    first_target="${TARGETS%% *}"
    exec swift package --disable-sandbox \
        preview-documentation --target "$first_target"
fi

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# The experimental Markdown-output flags only exist in recent swift-docc. The
# Xcode-bundled docc may not have them, so probe `docc convert --help` and skip
# (with a warning) rather than hard-failing on an unknown option.
# Resolve the same docc the plugin will use: prefer one on PATH (swiftly puts the
# selected toolchain's binaries there), otherwise fall back to xcrun (which honors
# an exported TOOLCHAINS).
DOCC_BIN="$(command -v docc 2>/dev/null || xcrun --find docc 2>/dev/null || true)"
docc_supports() {
    [[ -n "$DOCC_BIN" ]] && "$DOCC_BIN" convert --help 2>&1 | grep -q -- "$1"
}

EXTRA_FLAGS=()
if [[ "${EMIT_MARKDOWN:-0}" == "1" || "${EMIT_LLMS_TXT:-0}" == "1" ]]; then
    for flag in --enable-experimental-markdown-output \
                --enable-experimental-markdown-output-manifest; do
        if docc_supports "$flag"; then
            EXTRA_FLAGS+=("$flag")
        else
            echo "warning: active docc does not support '$flag' — skipping." >&2
            echo "         (needs a recent swift.org toolchain; select it with TOOLCHAINS=…" >&2
            echo "          or 'xcrun --toolchain <id>'. The Xcode-bundled docc lacks it.)" >&2
        fi
    done
fi

SOURCE_FLAGS=()
if [[ -n "$REPO_URL" ]]; then
    SOURCE_FLAGS+=(
        --source-service github
        --source-service-base-url "${REPO_URL%/}/blob/${REPO_BRANCH}"
        --checkout-path "$(pwd)"
    )
fi

for TARGET in $TARGETS; do
    slug="$(echo "$TARGET" | tr '[:upper:]' '[:lower:]')"
    out="$OUTPUT_DIR/$TARGET"
    mkdir -p "$out"

    echo ">> Building DocC for $TARGET → $out"
    swift package --allow-writing-to-directory "$out" \
        generate-documentation \
        --target "$TARGET" \
        --fallback-bundle-identifier "${HOSTING_BASE_PATH}.${slug}" \
        --output-path "$out" \
        --emit-digest \
        --disable-indexing \
        --transform-for-static-hosting \
        --hosting-base-path "${HOSTING_BASE_PATH}/${TARGET}" \
        ${SOURCE_FLAGS[@]+"${SOURCE_FLAGS[@]}"} \
        ${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}
done

if [[ "${EMIT_LLMS_TXT:-0}" == "1" ]]; then
    LLMS="$OUTPUT_DIR/llms.txt"
    {
        echo "# ${HOSTING_BASE_PATH} — DocC export for LLM consumption"
        echo
        echo "Generated $(date -u +%FT%TZ) from swift-docc."
        echo "Targets: $TARGETS"
        echo
        for TARGET in $TARGETS; do
            find "$OUTPUT_DIR/$TARGET/data" -name '*.md' -type f 2>/dev/null \
                | sort \
                | while IFS= read -r f; do
                    rel="${f#$OUTPUT_DIR/}"
                    echo
                    echo "---"
                    echo "## $rel"
                    echo
                    cat "$f"
                done
        done
    } > "$LLMS"
    echo "Wrote $LLMS ($(wc -l < "$LLMS" | tr -d ' ') lines)."
fi

# Write a top-level redirect index.html so the Pages root URL lands on the
# first target's documentation instead of returning 404. The DocC build emits
# everything under $OUTPUT_DIR/$TARGET/; without this redirect,
# https://<user>.github.io/<HOSTING_BASE_PATH>/ has no index and 404s even on
# successful deploys.
first_target="${TARGETS%% *}"
first_slug="$(echo "$first_target" | tr '[:upper:]' '[:lower:]')"
redirect_url="/${HOSTING_BASE_PATH}/${first_target}/documentation/${first_slug}/"
cat > "$OUTPUT_DIR/index.html" <<HTML
<!doctype html>
<meta charset="utf-8">
<title>${HOSTING_BASE_PATH}</title>
<meta http-equiv="refresh" content="0; url=${redirect_url}">
<link rel="canonical" href="${redirect_url}">
<p>Redirecting to <a href="${redirect_url}">${redirect_url}</a>.</p>
HTML

echo
echo "Docs written to $OUTPUT_DIR/. Open $OUTPUT_DIR/<Target>/index.html"
echo "or push to gh-pages."
