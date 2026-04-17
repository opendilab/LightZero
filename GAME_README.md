# 打飞机游戏 - Linux Python版

一个使用Python和Pygame开发的经典打飞机游戏，适用于Linux系统。

## 游戏特色

- 🚀 经典的打飞机玩法
- 🎯 流畅的操作体验
- 💥 爆炸特效
- 🏆 分数系统
- ❤️ 血量系统
- ⏸️ 暂停功能

## 系统要求

- Linux操作系统
- Python 3.6+
- Pygame库

## 安装和运行

### 方法1: 使用启动脚本（推荐）

```bash
# 给启动脚本执行权限
chmod +x start_game.sh

# 运行游戏
./start_game.sh
```

启动脚本会自动：
- 检查Python环境
- 安装所需依赖
- 启动游戏

### 方法2: 手动安装

```bash
# 安装依赖
pip3 install pygame --user

# 或者使用系统包管理器
# Ubuntu/Debian:
sudo apt-get install python3-pygame

# CentOS/RHEL:
sudo yum install python3-pygame

# Fedora:
sudo dnf install python3-pygame

# Arch Linux:
sudo pacman -S python-pygame

# 运行游戏
python3 plane_game.py
```

## 游戏操作

| 按键 | 功能 |
|------|------|
| ↑↓←→ | 移动飞机 |
| 空格 | 发射子弹 |
| P | 暂停/继续游戏 |
| R | 游戏结束后重新开始 |
| 关闭窗口 | 退出游戏 |

## 游戏玩法

1. 使用方向键控制蓝色飞机移动
2. 按空格键发射黄色子弹攻击红色敌机
3. 避免被敌机的红色子弹击中
4. 避免与敌机相撞
5. 击毁敌机获得分数
6. 血量归零游戏结束

## 游戏截图

游戏界面包含：
- 玩家飞机（蓝色三角形）
- 敌机（红色三角形）
- 子弹（黄色/红色矩形）
- 爆炸效果（彩色圆圈）
- 血量条（红色进度条）
- 分数显示
- 控制说明

## 技术特性

- 面向对象设计
- 碰撞检测系统
- 粒子效果系统
- 游戏状态管理
- 60FPS流畅运行

## 故障排除

### 常见问题

1. **ImportError: No module named 'pygame'**
   ```bash
   pip3 install pygame --user
   ```

2. **权限问题**
   ```bash
   chmod +x start_game.sh
   ```

3. **显示问题**
   确保系统支持图形界面，如果是SSH连接需要X11转发：
   ```bash
   ssh -X username@hostname
   ```

4. **音频问题**
   如果遇到音频相关错误，可以禁用音频：
   ```bash
   export SDL_AUDIODRIVER=dummy
   python3 plane_game.py
   ```

## 开发信息

- 语言: Python 3
- 图形库: Pygame
- 开发环境: Linux
- 代码风格: PEP 8

## 许可证

本项目仅供学习和娱乐使用。

## 更新日志

### v1.0.0
- 基础游戏功能
- 玩家飞机控制
- 敌机生成和AI
- 子弹系统
- 碰撞检测
- 爆炸效果
- 分数系统
- 血量系统
- 暂停功能

---

享受游戏吧！🎮