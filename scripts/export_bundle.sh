#!/bin/sh
# Package commits for a machine that cannot reach github.com (e.g. the company laptop).
#
#   scripts/export_bundle.sh          # first time: dist/document-chunking-full.bundle (whole repo)
#                                     # afterwards: dist/document-chunking-<from>-<to>.bundle (new commits only)
#   scripts/export_bundle.sh --full   # force a full bundle
#
# On the other machine:
#   first time:  git clone document-chunking-full.bundle document-chunking
#   updates:     git pull /path/to/document-chunking-<from>-<to>.bundle main
#
# The tag `company-sync` remembers the last exported commit.
set -eu
cd "$(dirname "$0")/.."
mkdir -p dist
to=$(git rev-parse --short main)
if [ "${1:-}" = "--full" ] || ! git rev-parse -q --verify company-sync >/dev/null; then
  out=dist/document-chunking-full.bundle
  git bundle create "$out" main
else
  from=$(git rev-parse --short company-sync)
  if [ "$from" = "$to" ]; then echo "nothing new since $from"; exit 0; fi
  out=dist/document-chunking-$from-$to.bundle
  git bundle create "$out" company-sync..main
fi
git bundle verify -q "$out"
git tag -f company-sync main >/dev/null
echo "wrote $out ($(du -h "$out" | cut -f1)), commits up to $to"
