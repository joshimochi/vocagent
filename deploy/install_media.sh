#!/usr/bin/env bash
# Run INSIDE the Ubuntu hs inference container. No dpkg installation, package
# configuration, alternatives update, or GPU-driver change is performed.
set -euo pipefail

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  cat <<'HELP'
Usage: bash deploy/install_media.sh [PREFIX]

Download FFmpeg, ffprobe, fonts and all dependencies from the container's APT
sources; extract them privately and verify software H.264/AAC plus ASS subtitles.
Default PREFIX: $VOCAGENT_DEPLOY_ROOT/run-a406/media
Use PREFIX/current/bin in PATH. The system dpkg database and libraries are not
changed. APT state and downloaded archives remain in PREFIX/apt for inspection.

VOCAGENT_MEDIA_OFFLINE=1 reuses previously downloaded indexes/packages.
VOCAGENT_FFMPEG_VERSION pins an available distribution package version.
HELP
  exit 0
fi
if (($# > 1)); then echo 'Expected at most one installation prefix.' >&2; exit 2; fi
for executable in apt-get dpkg-deb find sort mktemp sha256sum flock; do
  command -v "$executable" >/dev/null || { echo "Missing required tool: $executable" >&2; exit 1; }
done

prefix=${1:-${VOCAGENT_DEPLOY_ROOT:-/data/group6/vocagent}/run-a406/media}
mkdir -p "$prefix"
prefix=$(cd "$prefix" && pwd -P)
if [[ $prefix == / || $prefix == *$'\n'* || $prefix == *:* || $prefix == *\"* || $prefix == *\\* ]]; then
  echo 'Use a private prefix without colons, quotes, backslashes or newlines.' >&2; exit 2
fi
exec 9>"$prefix/install.lock"
flock -n 9 || { echo 'Another media installation is using this prefix.' >&2; exit 1; }
state=$prefix/apt
mkdir -p "$state/lists/partial" "$state/archives" "$state/log" "$prefix/releases"
if [[ ${VOCAGENT_MEDIA_OFFLINE:-0} == 1 ]]; then
  if [[ ! -f $prefix/current/archives-path ]]; then
    echo 'Offline mode requires an earlier successful installation.' >&2; exit 1
  fi
  IFS= read -r archive_dir < "$prefix/current/archives-path"
else
  # Do not mix cached versions: extracting old and new .debs into the same tree
  # could silently overwrite newer libraries with an older archive on reruns.
  archive_dir=$(mktemp -d "$state/archives/download-XXXXXXXX")
fi
mkdir -p "$archive_dir/partial"
: > "$state/status"
# APT_CONFIG is read before the normal apt.conf.d files. Disable those files so
# container-specific post-invoke hooks cannot run outside our private prefix.
cat > "$state/apt.conf" <<EOF
Dir::Etc::parts "-";
Dir::Etc::main "-";
Dir::Etc::sourcelist "/etc/apt/sources.list";
Dir::Etc::sourceparts "/etc/apt/sources.list.d";
Dir::Etc::trusted "/etc/apt/trusted.gpg";
Dir::Etc::trustedparts "/etc/apt/trusted.gpg.d";
Dir::State "$state";
Dir::State::status "$state/status";
Dir::State::lists "$state/lists";
Dir::State::extended_states "$state/extended_states";
Dir::Cache "$state";
Dir::Cache::archives "$archive_dir";
Dir::Cache::pkgcache "$state/pkgcache.bin";
Dir::Cache::srcpkgcache "$state/srcpkgcache.bin";
Dir::Log "$state/log";
APT::Sandbox::User "$(id -un)";
APT::Install-Recommends "false";
APT::Install-Suggests "false";
APT::Get::List-Cleanup "false";
EOF
apt=(env "APT_CONFIG=$state/apt.conf" apt-get)
download_options=()
if [[ ${VOCAGENT_MEDIA_OFFLINE:-0} == 1 ]]; then
  download_options+=(--no-download)
else
  "${apt[@]}" update
fi
package=ffmpeg
if [[ -n ${VOCAGENT_FFMPEG_VERSION:-} ]]; then package+="=$VOCAGENT_FFMPEG_VERSION"; fi
# An empty PRIVATE status file requests the complete dependency closure and
# ignores the container's partially configured packages/vendor OpenCL conflicts.
"${apt[@]}" --yes --download-only --no-install-recommends "${download_options[@]}" \
  install "$package" fontconfig fonts-dejavu-core

stage=$(mktemp -d "$prefix/releases/.build-XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
mkdir -p "$stage/root" "$stage/bin"
shopt -s nullglob
archives=("$archive_dir/"*.deb)
if ((${#archives[@]} == 0)); then echo 'APT returned no package archives.' >&2; exit 1; fi
for archive in "${archives[@]}"; do
  dpkg-deb --extract "$archive" "$stage/root"
  dpkg-deb --field "$archive" Package Version Architecture >> "$stage/packages.txt"
  sha256sum "$archive" >> "$stage/archive-sha256.txt"
done
printf '%s\n' "$archive_dir" > "$stage/archives-path"
for binary in ffmpeg ffprobe; do
  [[ -x $stage/root/usr/bin/$binary ]] || { echo "Missing extracted $binary." >&2; exit 1; }
done
# Use the extracted loader with its matching libc. Prepending a newly downloaded
# libc to the system loader's LD_LIBRARY_PATH can break even basic process startup.
mapfile -t loaders < <(find "$stage/root" -type f -name 'ld-linux*.so*' | sort)
if ((${#loaders[@]} != 1)); then echo 'Expected one extracted Linux ELF loader.' >&2; exit 1; fi
printf '%s\n' "${loaders[0]#"$stage/"}" > "$stage/loader-path"
while IFS= read -r directory; do
  printf '%s\n' "${directory#"$stage/"}"
done < <(find "$stage/root" -type f -name '*.so*' -printf '%h\n' | sort -u) > "$stage/library-paths"
mkdir -p "$stage/root/var/cache/fontconfig"
for binary in ffmpeg ffprobe; do
  cat > "$stage/bin/$binary" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
release=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
binary=${BASH_SOURCE[0]##*/}
IFS= read -r loader < "$release/loader-path"
library_path=
while IFS= read -r directory; do
  library_path+="${library_path:+:}$release/$directory"
done < "$release/library-paths"
# These variables apply only to the media process, never the parent GPU runtime.
export LD_LIBRARY_PATH="$library_path"
export FONTCONFIG_FILE="$release/root/etc/fonts/fonts.conf"
export FONTCONFIG_PATH="$release/root/etc/fonts"
export FONTCONFIG_SYSROOT="$release/root"
exec "$release/$loader" --library-path "$library_path" "$release/root/usr/bin/$binary" "$@"
WRAPPER
  chmod +x "$stage/bin/$binary"
done

# Exercise the exact software codecs/filter required by the vocabulary workflow.
cat > "$stage/smoke.ass" <<'ASS'
[Script Info]
ScriptType: v4.00+
PlayResX: 128
PlayResY: 96
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,12,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,1,0,2,4,4,4,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:00.20,Default,,0,0,0,,media check
ASS
(
  cd "$stage"
  ./bin/ffmpeg -hide_banner -loglevel error -nostdin -y \
    -f lavfi -i color=c=blue:s=128x96:d=0.2 \
    -f lavfi -i anullsrc=r=16000:cl=mono -vf ass=smoke.ass \
    -t 0.2 -c:v libx264 -pix_fmt yuv420p -c:a aac -f mp4 smoke.mp4
  ./bin/ffprobe -v error -show_entries stream=codec_type -of csv=p=0 smoke.mp4 > smoke-streams.txt
  streams=$(cat smoke-streams.txt)
  [[ $streams == *video* && $streams == *audio* ]] || { echo 'Media validation lacked video/audio.' >&2; exit 1; }
  ./bin/ffmpeg -version > ffmpeg-version.txt
)
release_name=${stage##*/}
release_name=${release_name#.build-}
mv "$stage" "$prefix/releases/$release_name"
stage=$prefix/releases/.removed-build-$release_name
link=$prefix/.current-$release_name
ln -s "releases/$release_name" "$link"
mv -Tf "$link" "$prefix/current"
trap - EXIT
printf 'Validated private FFmpeg/ffprobe: %s/current/bin\n' "$prefix"
