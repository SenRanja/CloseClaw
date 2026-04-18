#!/bin/bash
# ============================================================
# 清理缓存释放磁盘空间
# ============================================================

echo ">>> 当前磁盘使用情况："
df -h / 2>/dev/null || df -h . 2>/dev/null
echo ""

# ---- pip 缓存 ----
PIP_CACHE=$(pip cache dir 2>/dev/null)
if [ -n "$PIP_CACHE" ] && [ -d "$PIP_CACHE" ]; then
    echo ">>> pip 缓存: $(du -sh "$PIP_CACHE" 2>/dev/null | cut -f1)"
    pip cache purge
    echo "    已清理"
else
    echo ">>> pip 缓存: 无"
fi

# ---- HuggingFace 缓存 ----
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -d "$HF_CACHE" ]; then
    echo ">>> HuggingFace 缓存: $(du -sh "$HF_CACHE" 2>/dev/null | cut -f1)"
    read -p "    是否清理？(y/N) " ans
    if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
        rm -rf "$HF_CACHE"
        echo "    已清理"
    else
        echo "    跳过"
    fi
else
    echo ">>> HuggingFace 缓存: 无"
fi

# ---- torch 缓存 ----
TORCH_CACHE="$HOME/.cache/torch"
if [ -d "$TORCH_CACHE" ]; then
    echo ">>> torch 缓存: $(du -sh "$TORCH_CACHE" 2>/dev/null | cut -f1)"
    rm -rf "$TORCH_CACHE"
    echo "    已清理"
else
    echo ">>> torch 缓存: 无"
fi

# ---- conda 缓存 ----
if command -v conda &>/dev/null; then
    echo ">>> conda 缓存:"
    conda clean --all -y 2>/dev/null
    echo "    已清理"
fi

# ---- __pycache__ ----
PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYCACHE_SIZE=$(find "$PROJ_DIR" -type d -name "__pycache__" -exec du -sh {} + 2>/dev/null | tail -1 | cut -f1)
if [ -n "$PYCACHE_SIZE" ]; then
    echo ">>> 项目 __pycache__: $PYCACHE_SIZE"
    find "$PROJ_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
    echo "    已清理"
fi

echo ""
echo ">>> 清理后磁盘使用情况："
df -h / 2>/dev/null || df -h . 2>/dev/null
