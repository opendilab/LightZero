#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令行打飞机游戏 - 纯文本版本
使用ASCII字符在终端中运行的打飞机游戏
"""

import curses
import random
import time
import threading
from collections import deque

class GameObject:
    """游戏对象基类"""
    def __init__(self, x, y, char, color=1):
        self.x = x
        self.y = y
        self.char = char
        self.color = color
        self.active = True

class Player(GameObject):
    """玩家飞机"""
    def __init__(self, x, y):
        super().__init__(x, y, '^', 2)  # 绿色
        self.health = 100
        self.max_health = 100

    def move(self, dx, dy, max_x, max_y):
        """移动玩家"""
        new_x = max(0, min(max_x - 1, self.x + dx))
        new_y = max(0, min(max_y - 1, self.y + dy))
        self.x, self.y = new_x, new_y

class Enemy(GameObject):
    """敌机"""
    def __init__(self, x, y):
        super().__init__(x, y, 'v', 1)  # 红色
        self.speed = random.uniform(0.3, 0.8)
        self.last_move = time.time()

    def update(self):
        """更新敌机位置"""
        current_time = time.time()
        if current_time - self.last_move >= self.speed:
            self.y += 1
            self.last_move = current_time

class Bullet(GameObject):
    """子弹"""
    def __init__(self, x, y, direction=1, char='|'):
        color = 3 if direction == 1 else 1  # 玩家子弹黄色，敌机子弹红色
        super().__init__(x, y, char, color)
        self.direction = direction  # 1向上，-1向下
        self.speed = 0.1
        self.last_move = time.time()

    def update(self):
        """更新子弹位置"""
        current_time = time.time()
        if current_time - self.last_move >= self.speed:
            self.y -= self.direction
            self.last_move = current_time

class Explosion(GameObject):
    """爆炸效果"""
    def __init__(self, x, y):
        super().__init__(x, y, '*', 3)  # 黄色
        self.lifetime = 0.5
        self.created_time = time.time()

    def update(self):
        """更新爆炸效果"""
        if time.time() - self.created_time >= self.lifetime:
            self.active = False

class Game:
    """游戏主类"""
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.height, self.width = stdscr.getmaxyx()
        self.height -= 5  # 留出状态栏空间

        # 初始化颜色
        curses.start_color()
        curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)      # 敌机/敌弹
        curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)    # 玩家
        curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)   # 玩家子弹/爆炸
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)     # UI文字
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)    # 普通文字

        # 设置无阻塞输入
        self.stdscr.nodelay(True)
        curses.curs_set(0)  # 隐藏光标

        # 游戏对象
        self.player = Player(self.width // 2, self.height - 2)
        self.enemies = []
        self.bullets = []
        self.enemy_bullets = []
        self.explosions = []

        # 游戏状态
        self.score = 0
        self.game_over = False
        self.paused = False
        self.running = True

        # 计时器
        self.last_enemy_spawn = time.time()
        self.last_enemy_shoot = time.time()
        self.last_frame = time.time()

    def spawn_enemy(self):
        """生成敌机"""
        current_time = time.time()
        if current_time - self.last_enemy_spawn >= random.uniform(1.0, 3.0):
            x = random.randint(0, self.width - 1)
            enemy = Enemy(x, 0)
            self.enemies.append(enemy)
            self.last_enemy_spawn = current_time

    def enemy_shoot(self):
        """敌机射击"""
        current_time = time.time()
        if current_time - self.last_enemy_shoot >= 2.0:
            for enemy in self.enemies:
                if random.random() < 0.3:  # 30%概率射击
                    bullet = Bullet(enemy.x, enemy.y + 1, -1, 'o')
                    self.enemy_bullets.append(bullet)
            self.last_enemy_shoot = current_time

    def handle_input(self):
        """处理用户输入"""
        try:
            key = self.stdscr.getch()
            if key != -1:  # 有按键输入
                if key == ord('q') or key == 27:  # q或ESC退出
                    self.running = False
                elif key == ord(' ') and not self.game_over:  # 空格射击
                    bullet = Bullet(self.player.x, self.player.y - 1, 1)
                    self.bullets.append(bullet)
                elif key == ord('r') and self.game_over:  # r重新开始
                    self.restart_game()
                elif key == ord('p'):  # p暂停
                    self.paused = not self.paused
                elif not self.game_over and not self.paused:
                    # 移动控制
                    if key == curses.KEY_LEFT or key == ord('a'):
                        self.player.move(-1, 0, self.width, self.height)
                    elif key == curses.KEY_RIGHT or key == ord('d'):
                        self.player.move(1, 0, self.width, self.height)
                    elif key == curses.KEY_UP or key == ord('w'):
                        self.player.move(0, -1, self.width, self.height)
                    elif key == curses.KEY_DOWN or key == ord('s'):
                        self.player.move(0, 1, self.width, self.height)
        except:
            pass

    def update_game_objects(self):
        """更新游戏对象"""
        if self.game_over or self.paused:
            return

        # 生成敌机
        self.spawn_enemy()

        # 敌机射击
        self.enemy_shoot()

        # 更新敌机
        for enemy in self.enemies[:]:
            enemy.update()
            if enemy.y >= self.height:
                self.enemies.remove(enemy)

        # 更新子弹
        for bullet in self.bullets[:]:
            bullet.update()
            if bullet.y < 0:
                self.bullets.remove(bullet)

        for bullet in self.enemy_bullets[:]:
            bullet.update()
            if bullet.y >= self.height:
                self.enemy_bullets.remove(bullet)

        # 更新爆炸效果
        for explosion in self.explosions[:]:
            explosion.update()
            if not explosion.active:
                self.explosions.remove(explosion)

        # 碰撞检测
        self.check_collisions()

        # 检查游戏结束
        if self.player.health <= 0:
            self.game_over = True

    def check_collisions(self):
        """碰撞检测"""
        # 玩家子弹击中敌机
        for bullet in self.bullets[:]:
            for enemy in self.enemies[:]:
                if bullet.x == enemy.x and bullet.y == enemy.y:
                    self.bullets.remove(bullet)
                    self.enemies.remove(enemy)
                    self.explosions.append(Explosion(enemy.x, enemy.y))
                    self.score += 10
                    break

        # 敌机子弹击中玩家
        for bullet in self.enemy_bullets[:]:
            if bullet.x == self.player.x and bullet.y == self.player.y:
                self.enemy_bullets.remove(bullet)
                self.player.health -= 10
                self.explosions.append(Explosion(self.player.x, self.player.y))

        # 敌机撞击玩家
        for enemy in self.enemies[:]:
            if enemy.x == self.player.x and enemy.y == self.player.y:
                self.enemies.remove(enemy)
                self.player.health -= 20
                self.explosions.append(Explosion(enemy.x, enemy.y))

    def draw(self):
        """绘制游戏画面"""
        self.stdscr.clear()

        # 绘制边框
        for x in range(self.width):
            self.stdscr.addch(0, x, '-', curses.color_pair(5))
            self.stdscr.addch(self.height, x, '-', curses.color_pair(5))

        # 绘制游戏对象
        if not self.game_over:
            # 玩家
            if 0 <= self.player.y < self.height and 0 <= self.player.x < self.width:
                self.stdscr.addch(self.player.y + 1, self.player.x,
                                self.player.char, curses.color_pair(self.player.color))

        # 敌机
        for enemy in self.enemies:
            if 0 <= enemy.y < self.height and 0 <= enemy.x < self.width:
                self.stdscr.addch(enemy.y + 1, enemy.x,
                                enemy.char, curses.color_pair(enemy.color))

        # 子弹
        for bullet in self.bullets:
            if 0 <= bullet.y < self.height and 0 <= bullet.x < self.width:
                self.stdscr.addch(bullet.y + 1, bullet.x,
                                bullet.char, curses.color_pair(bullet.color))

        for bullet in self.enemy_bullets:
            if 0 <= bullet.y < self.height and 0 <= bullet.x < self.width:
                self.stdscr.addch(bullet.y + 1, bullet.x,
                                bullet.char, curses.color_pair(bullet.color))

        # 爆炸效果
        for explosion in self.explosions:
            if 0 <= explosion.y < self.height and 0 <= explosion.x < self.width:
                self.stdscr.addch(explosion.y + 1, explosion.x,
                                explosion.char, curses.color_pair(explosion.color))

        # 绘制UI
        self.draw_ui()

        self.stdscr.refresh()

    def draw_ui(self):
        """绘制用户界面"""
        ui_y = self.height + 1

        # 分数和血量
        score_text = f"分数: {self.score}"
        health_bar = "血量: " + "█" * (self.player.health // 5) + "░" * ((100 - self.player.health) // 5)

        self.stdscr.addstr(ui_y, 0, score_text, curses.color_pair(4))
        self.stdscr.addstr(ui_y + 1, 0, health_bar, curses.color_pair(2))

        # 控制说明
        controls = "WASD/方向键:移动 空格:射击 P:暂停 Q:退出"
        if len(controls) < self.width:
            self.stdscr.addstr(ui_y + 2, 0, controls, curses.color_pair(5))

        # 游戏状态
        if self.paused:
            pause_text = "游戏暂停 - 按P继续"
            x = max(0, (self.width - len(pause_text)) // 2)
            self.stdscr.addstr(self.height // 2, x, pause_text, curses.color_pair(3))

        if self.game_over:
            game_over_text = "游戏结束!"
            final_score = f"最终分数: {self.score}"
            restart_text = "按R重新开始，按Q退出"

            x1 = max(0, (self.width - len(game_over_text)) // 2)
            x2 = max(0, (self.width - len(final_score)) // 2)
            x3 = max(0, (self.width - len(restart_text)) // 2)

            self.stdscr.addstr(self.height // 2 - 1, x1, game_over_text, curses.color_pair(1))
            self.stdscr.addstr(self.height // 2, x2, final_score, curses.color_pair(4))
            self.stdscr.addstr(self.height // 2 + 1, x3, restart_text, curses.color_pair(5))

    def restart_game(self):
        """重新开始游戏"""
        self.player = Player(self.width // 2, self.height - 2)
        self.enemies = []
        self.bullets = []
        self.enemy_bullets = []
        self.explosions = []
        self.score = 0
        self.game_over = False
        self.paused = False
        self.last_enemy_spawn = time.time()
        self.last_enemy_shoot = time.time()

    def run(self):
        """运行游戏主循环"""
        while self.running:
            current_time = time.time()

            # 控制帧率 (约30 FPS)
            if current_time - self.last_frame >= 1.0 / 30:
                self.handle_input()
                self.update_game_objects()
                self.draw()
                self.last_frame = current_time

            time.sleep(0.01)  # 避免CPU占用过高

def show_welcome():
    """显示欢迎信息"""
    print("=" * 50)
    print("🚀 命令行打飞机游戏 🚀")
    print("=" * 50)
    print()
    print("游戏说明:")
    print("• 你是绿色的 ^ 飞机")
    print("• 红色的 v 是敌机")
    print("• | 是你的子弹，o 是敌机子弹")
    print("• * 是爆炸效果")
    print()
    print("操作方法:")
    print("• WASD 或 方向键: 移动飞机")
    print("• 空格键: 发射子弹")
    print("• P键: 暂停/继续游戏")
    print("• R键: 游戏结束后重新开始")
    print("• Q键或ESC: 退出游戏")
    print()
    print("游戏目标:")
    print("• 击毁敌机获得分数")
    print("• 避免被敌机子弹击中")
    print("• 避免与敌机相撞")
    print("• 血量归零游戏结束")
    print()
    print("按任意键开始游戏...")
    input()

def main():
    """主函数"""
    try:
        show_welcome()
        curses.wrapper(lambda stdscr: Game(stdscr).run())
    except KeyboardInterrupt:
        print("\n游戏被用户中断")
    except Exception as e:
        print(f"\n游戏运行出错: {e}")
        print("请确保终端支持颜色显示和方向键输入")

if __name__ == "__main__":
    main()