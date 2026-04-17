#!/bin/bash
# 命令行打飞机游戏启动脚本

echo "=== 命令行打飞机游戏 Linux版 ==="
echo "正在检查Python环境..."

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

echo "Python版本: $(python3 --version)"

# 检查终端是否支持颜色
if [ -z "$TERM" ] || [ "$TERM" = "dumb" ]; then
    echo "警告: 终端可能不支持颜色显示，游戏体验可能受影响"
fi

echo "终端类型: $TERM"
echo "终端大小: $(tput cols)x$(tput lines)"

# 检查终端大小
COLS=$(tput cols)
LINES=$(tput lines)

if [ "$COLS" -lt 60 ] || [ "$LINES" -lt 20 ]; then
    echo "警告: 终端窗口太小，建议至少60x20字符"
    echo "当前大小: ${COLS}x${LINES}"
    echo "请调整终端窗口大小后重试"
    read -p "是否继续? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "环境检查完成！"
echo "启动游戏..."
echo ""

# 启动游戏
python3 terminal_plane_game.py