#!/usr/bin/env bash
# 双击启动器：自动安装 Python (uv)、依赖、模型权重，然后启动网页 UI
set -e

# 切换到脚本所在目录（双击运行时 cwd 是用户家目录）
cd "$(dirname "$0")"

echo "================================================"
echo "  视频字幕去除器 v0.2.1 — 启动中"
echo "================================================"
echo ""

# ---- 1. 确保 uv 可用 ----
if ! command -v uv >/dev/null 2>&1; then
  # 检查 uv 是否在标准位置但不在 PATH
  if [ -x "$HOME/.local/bin/uv" ]; then
    export PATH="$HOME/.local/bin:$PATH"
  else
    echo "[1/3] 正在安装 uv (Python 包管理器，一次性)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    if ! command -v uv >/dev/null 2>&1; then
      echo ""
      echo "❌ uv 安装失败。请检查网络连接后重试。"
      echo "按任意键退出..."
      read -n 1
      exit 1
    fi
    echo "✅ uv 安装完成"
    echo ""
  fi
fi

# ---- 2. 创建虚拟环境 + 装依赖（仅首次）----
if [ ! -d ".venv" ] || [ ! -f ".venv/.setup_done" ]; then
  echo "[2/3] 首次启动初始化 (大约需要 3-5 分钟，下载约 1 GB)..."
  echo "      这一步只在第一次运行时执行，之后启动只要几秒。"
  echo ""

  uv venv --python 3.11 --seed
  uv pip install -r requirements.txt

  echo ""
  echo "      正在预下载 ffmpeg 和 LaMa 模型权重..."
  uv run python -c "
from static_ffmpeg.run import get_or_fetch_platform_executables_else_raise
ff, fp = get_or_fetch_platform_executables_else_raise()
print(f'  ffmpeg : {ff}')
print(f'  ffprobe: {fp}')
from simple_lama_inpainting import SimpleLama
print('  Loading LaMa weights (~200 MB on first run)...')
SimpleLama()
print('  Done.')
"

  touch .venv/.setup_done
  echo ""
  echo "✅ 初始化完成"
  echo ""
fi

# ---- 3. 启动网页 UI ----
echo "[3/3] 启动网页界面..."
echo "      浏览器会自动打开。如果 7860 被占用，会自动切换到别的端口。"
echo "      关闭这个终端窗口即可退出程序。"
echo ""

uv run python -m app.web
