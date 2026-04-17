#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
打飞机游戏 - Python版本
使用pygame库开发的经典打飞机游戏
"""

import pygame
import random
import sys
import os

# 初始化pygame
pygame.init()

# 游戏常量
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

class Player:
    """玩家飞机类"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 60
        self.height = 40
        self.speed = 5
        self.health = 100
        self.max_health = 100

    def move(self, keys):
        """根据按键移动玩家飞机"""
        if keys[pygame.K_LEFT] and self.x > 0:
            self.x -= self.speed
        if keys[pygame.K_RIGHT] and self.x < SCREEN_WIDTH - self.width:
            self.x += self.speed
        if keys[pygame.K_UP] and self.y > 0:
            self.y -= self.speed
        if keys[pygame.K_DOWN] and self.y < SCREEN_HEIGHT - self.height:
            self.y += self.speed

    def draw(self, screen):
        """绘制玩家飞机"""
        # 飞机主体
        pygame.draw.polygon(screen, BLUE, [
            (self.x + self.width//2, self.y),
            (self.x, self.y + self.height),
            (self.x + self.width//4, self.y + self.height - 10),
            (self.x + 3*self.width//4, self.y + self.height - 10),
            (self.x + self.width, self.y + self.height)
        ])

        # 血条
        bar_width = self.width
        bar_height = 6
        fill = (self.health / self.max_health) * bar_width
        outline_rect = pygame.Rect(self.x, self.y - 15, bar_width, bar_height)
        fill_rect = pygame.Rect(self.x, self.y - 15, fill, bar_height)
        pygame.draw.rect(screen, RED, fill_rect)
        pygame.draw.rect(screen, WHITE, outline_rect, 2)

    def get_rect(self):
        """获取碰撞矩形"""
        return pygame.Rect(self.x, self.y, self.width, self.height)

class Enemy:
    """敌机类"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.width = 40
        self.height = 30
        self.speed = random.randint(2, 4)
        self.health = 30

    def move(self):
        """敌机向下移动"""
        self.y += self.speed

    def draw(self, screen):
        """绘制敌机"""
        pygame.draw.polygon(screen, RED, [
            (self.x + self.width//2, self.y + self.height),
            (self.x, self.y),
            (self.x + self.width//4, self.y + 10),
            (self.x + 3*self.width//4, self.y + 10),
            (self.x + self.width, self.y)
        ])

    def get_rect(self):
        """获取碰撞矩形"""
        return pygame.Rect(self.x, self.y, self.width, self.height)

class Bullet:
    """子弹类"""
    def __init__(self, x, y, direction=1):
        self.x = x
        self.y = y
        self.width = 4
        self.height = 10
        self.speed = 8
        self.direction = direction  # 1为向上，-1为向下

    def move(self):
        """子弹移动"""
        self.y -= self.speed * self.direction

    def draw(self, screen):
        """绘制子弹"""
        color = YELLOW if self.direction == 1 else RED
        pygame.draw.rect(screen, color, (self.x, self.y, self.width, self.height))

    def get_rect(self):
        """获取碰撞矩形"""
        return pygame.Rect(self.x, self.y, self.width, self.height)

class Explosion:
    """爆炸效果类"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = 5
        self.max_radius = 30
        self.speed = 2

    def update(self):
        """更新爆炸效果"""
        self.radius += self.speed
        return self.radius < self.max_radius

    def draw(self, screen):
        """绘制爆炸效果"""
        colors = [YELLOW, RED, (255, 165, 0)]  # 黄色、红色、橙色
        for i, color in enumerate(colors):
            radius = max(1, self.radius - i * 5)
            if radius > 0:
                pygame.draw.circle(screen, color, (int(self.x), int(self.y)), radius)

class Game:
    """游戏主类"""
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("打飞机游戏")
        self.clock = pygame.time.Clock()

        # 游戏对象
        self.player = Player(SCREEN_WIDTH // 2 - 30, SCREEN_HEIGHT - 80)
        self.enemies = []
        self.bullets = []
        self.enemy_bullets = []
        self.explosions = []

        # 游戏状态
        self.score = 0
        self.game_over = False
        self.paused = False

        # 计时器
        self.enemy_spawn_timer = 0
        self.enemy_shoot_timer = 0

        # 字体
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)

    def spawn_enemy(self):
        """生成敌机"""
        if self.enemy_spawn_timer <= 0:
            x = random.randint(0, SCREEN_WIDTH - 40)
            enemy = Enemy(x, -30)
            self.enemies.append(enemy)
            self.enemy_spawn_timer = random.randint(60, 120)  # 1-2秒
        else:
            self.enemy_spawn_timer -= 1

    def enemy_shoot(self):
        """敌机射击"""
        if self.enemy_shoot_timer <= 0:
            for enemy in self.enemies:
                if random.randint(1, 100) < 2:  # 2%概率射击
                    bullet = Bullet(enemy.x + enemy.width//2, enemy.y + enemy.height, -1)
                    self.enemy_bullets.append(bullet)
            self.enemy_shoot_timer = 30  # 0.5秒
        else:
            self.enemy_shoot_timer -= 1

    def handle_events(self):
        """处理事件"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not self.game_over:
                    # 发射子弹
                    bullet = Bullet(self.player.x + self.player.width//2, self.player.y)
                    self.bullets.append(bullet)
                elif event.key == pygame.K_r and self.game_over:
                    # 重新开始游戏
                    self.restart_game()
                elif event.key == pygame.K_p:
                    # 暂停/继续
                    self.paused = not self.paused
        return True

    def update(self):
        """更新游戏状态"""
        if self.game_over or self.paused:
            return

        # 移动玩家
        keys = pygame.key.get_pressed()
        self.player.move(keys)

        # 生成敌机
        self.spawn_enemy()

        # 敌机射击
        self.enemy_shoot()

        # 移动敌机
        for enemy in self.enemies[:]:
            enemy.move()
            if enemy.y > SCREEN_HEIGHT:
                self.enemies.remove(enemy)

        # 移动子弹
        for bullet in self.bullets[:]:
            bullet.move()
            if bullet.y < 0:
                self.bullets.remove(bullet)

        for bullet in self.enemy_bullets[:]:
            bullet.move()
            if bullet.y > SCREEN_HEIGHT:
                self.enemy_bullets.remove(bullet)

        # 更新爆炸效果
        for explosion in self.explosions[:]:
            if not explosion.update():
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
                if bullet.get_rect().colliderect(enemy.get_rect()):
                    self.bullets.remove(bullet)
                    enemy.health -= 20
                    if enemy.health <= 0:
                        self.enemies.remove(enemy)
                        self.explosions.append(Explosion(enemy.x + enemy.width//2, enemy.y + enemy.height//2))
                        self.score += 10
                    break

        # 敌机子弹击中玩家
        for bullet in self.enemy_bullets[:]:
            if bullet.get_rect().colliderect(self.player.get_rect()):
                self.enemy_bullets.remove(bullet)
                self.player.health -= 10
                self.explosions.append(Explosion(self.player.x + self.player.width//2, self.player.y + self.player.height//2))

        # 敌机撞击玩家
        for enemy in self.enemies[:]:
            if enemy.get_rect().colliderect(self.player.get_rect()):
                self.enemies.remove(enemy)
                self.player.health -= 20
                self.explosions.append(Explosion(enemy.x + enemy.width//2, enemy.y + enemy.height//2))

    def draw(self):
        """绘制游戏画面"""
        self.screen.fill(BLACK)

        # 绘制星空背景
        for i in range(50):
            x = random.randint(0, SCREEN_WIDTH)
            y = random.randint(0, SCREEN_HEIGHT)
            pygame.draw.circle(self.screen, WHITE, (x, y), 1)

        if not self.game_over:
            # 绘制游戏对象
            self.player.draw(self.screen)

            for enemy in self.enemies:
                enemy.draw(self.screen)

            for bullet in self.bullets:
                bullet.draw(self.screen)

            for bullet in self.enemy_bullets:
                bullet.draw(self.screen)

        # 绘制爆炸效果
        for explosion in self.explosions:
            explosion.draw(self.screen)

        # 绘制UI
        self.draw_ui()

        pygame.display.flip()

    def draw_ui(self):
        """绘制用户界面"""
        # 分数
        score_text = self.font.render(f"分数: {self.score}", True, WHITE)
        self.screen.blit(score_text, (10, 10))

        # 暂停提示
        if self.paused:
            pause_text = self.font.render("游戏暂停 - 按P继续", True, YELLOW)
            text_rect = pause_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            self.screen.blit(pause_text, text_rect)

        # 游戏结束界面
        if self.game_over:
            game_over_text = self.font.render("游戏结束!", True, RED)
            score_text = self.font.render(f"最终分数: {self.score}", True, WHITE)
            restart_text = self.small_font.render("按R重新开始", True, GREEN)

            game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50))
            score_rect = score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50))

            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(score_text, score_rect)
            self.screen.blit(restart_text, restart_rect)

        # 控制说明
        if not self.game_over:
            controls = [
                "方向键: 移动",
                "空格: 射击",
                "P: 暂停"
            ]
            for i, control in enumerate(controls):
                text = self.small_font.render(control, True, WHITE)
                self.screen.blit(text, (SCREEN_WIDTH - 150, 10 + i * 25))

    def restart_game(self):
        """重新开始游戏"""
        self.player = Player(SCREEN_WIDTH // 2 - 30, SCREEN_HEIGHT - 80)
        self.enemies = []
        self.bullets = []
        self.enemy_bullets = []
        self.explosions = []
        self.score = 0
        self.game_over = False
        self.paused = False
        self.enemy_spawn_timer = 0
        self.enemy_shoot_timer = 0

    def run(self):
        """运行游戏主循环"""
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()
        sys.exit()

def main():
    """主函数"""
    print("打飞机游戏启动中...")
    print("控制说明:")
    print("- 方向键: 移动飞机")
    print("- 空格键: 发射子弹")
    print("- P键: 暂停/继续游戏")
    print("- R键: 游戏结束后重新开始")
    print("- 关闭窗口或Ctrl+C: 退出游戏")

    try:
        game = Game()
        game.run()
    except KeyboardInterrupt:
        print("\n游戏被用户中断")
        pygame.quit()
        sys.exit()
    except Exception as e:
        print(f"游戏运行出错: {e}")
        pygame.quit()
        sys.exit(1)

if __name__ == "__main__":
    main()