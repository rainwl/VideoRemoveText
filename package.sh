#!/usr/bin/env bash
# 打包成可发给朋友的 zip
set -e
cd "$(dirname "$0")"

OUT="VideoRemoveText.zip"
rm -f "$OUT"

zip -r "$OUT" . \
  -x "$OUT" \
  -x ".venv/*" \
  -x ".venv.bak/*" \
  -x "work/*" \
  -x "*.mp4" \
  -x ".git/*" \
  -x ".gitignore" \
  -x "__pycache__/*" \
  -x "**/__pycache__/*" \
  -x "*.pyc" \
  -x ".DS_Store" \
  -x "**/.DS_Store"

echo ""
echo "✅ 打包完成: $(pwd)/$OUT"
echo "   大小: $(du -h "$OUT" | cut -f1)"
echo ""
echo "把这个 zip 文件 + INSTALL_FRIEND.md 一起发给朋友即可。"
