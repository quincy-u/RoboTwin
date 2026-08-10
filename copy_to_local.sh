#!/bin/bash
# Parallel copy of the merged dataset from NFS /shared to local disk (same disk as Lingbot-VA).
# Copies per chunk/cam leaf dir with 16 workers (single-threaded rsync over 110k NFS files crawls).
set -u
SRC=/shared/perception/datasets/robotwin_eef_obb_27500
DST=/home/qinxiyu2/lerobot_local/quincyu/robotwin_eef_obb_27500

echo "[copy] $(date) $SRC -> $DST"
mkdir -p "$DST"
# meta is small — copy directly
rsync -a "$SRC/meta" "$DST/"

cd "$SRC" || { echo "[copy] cannot cd $SRC"; exit 1; }
# leaf dirs: data/chunk-XXX (parquets) + videos/chunk-XXX/<cam> (mp4s); -R recreates the path under DST
{ find data -mindepth 1 -maxdepth 1 -type d
  find videos -mindepth 2 -maxdepth 2 -type d
} | xargs -P 16 -I{} rsync -aR "{}/" "$DST/"

echo "[copy] DONE $(date)"
