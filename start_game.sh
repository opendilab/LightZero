#!/bin/bash
# 打飞机游戏启动脚本

echo "=== 打飞机游戏 Linux版 ==="
echo "正在检查Python环境..."

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python3"
    exit 1
fi

echo "Python版本: $(python3 --version)"

# 检查pip是否安装
if ! command -v pip3 &> /dev/null; then
    echo "错误: 未找到pip3，请先安装pip3"
    exit 1
fi

# 安装依赖
echo "正在安装游戏依赖..."
pip3 install -r game_requirements.txt --user

if [ $? -ne 0 ]; then
    echo "依赖安装失败，尝试使用系统包管理器安装pygame..."

    # 检测系统类型并安装pygame
    if command -v apt-get &> /dev/null; then
        echo "检测到Ubuntu/Debian系统，使用apt安装..."
        sudo apt-get update
        sudo apt-get install -y python3-pygame
    elif command -v yum &> /dev/null; then
        echo "检测到CentOS/RHEL系统，使用yum安装..."
        sudo yum install -y python3-pygame
    elif command -v dnf &> /dev/null; then
        echo "检测到Fedora系统，使用dnf安装..."
        sudo dnf install -y python3-pygame
    elif command -v pacman &> /dev/null; then
        echo "检测到Arch Linux系统，使用pacman安装..."
        sudo pacman -S python-pygame
    else
        echo "无法自动安装pygame，请手动安装: pip3 install pygame"
        exit 1
    fi
fi

echo "依赖安装完成！"
echo "启动游戏..."
echo ""

# 启动游戏
python3 plane_game.py