import pygame
import random
from enum import Enum
from typing import NamedTuple
import numpy as np

pygame.init()
font = pygame.font.Font(size=60)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


class Point(NamedTuple):
    x: int
    y: int


WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 20
SPEED = 40


class SnakeGame:
    def __init__(self, w=640, h=480, n_snakes=1):
        self.w = w
        self.h = h
        self.n_snakes = n_snakes
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake")
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = [Direction.RIGHT] * self.n_snakes
        self.head = [
            Point(self.w / 2 - i * BLOCK_SIZE * 5, self.h / 2)
            for i in range(self.n_snakes)
        ]
        self.snake = [
            [
                self.head[i],
                Point(self.head[i].x - BLOCK_SIZE, self.head[i].y),
                Point(self.head[i].x - (2 * BLOCK_SIZE), self.head[i].y),
            ]
            for i in range(self.n_snakes)
        ]

        self.score = [0] * self.n_snakes
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        for snake in self.snake:
            if self.food in snake:
                self._place_food()
                break

    def play_step(self, actions):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        rewards = [0] * self.n_snakes
        game_overs = [False] * self.n_snakes
        for i in range(self.n_snakes):
            self._move(i, actions[i])
            self.snake[i].insert(0, self.head[i])

            if self.is_collision(i) or self.frame_iteration > 100 * len(self.snake[i]):
                game_overs[i] = True
                rewards[i] = -10
            elif self.head[i] == self.food:
                self.score[i] += 1
                rewards[i] = 10
                self._place_food()
            else:
                self.snake[i].pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return rewards, game_overs, self.score

    def is_collision(self, i, pt=None):
        if pt is None:
            pt = self.head[i]
        if (
            pt.x > self.w - BLOCK_SIZE
            or pt.x < 0
            or pt.y > self.h - BLOCK_SIZE
            or pt.y < 0
        ):
            return True
        if pt in self.snake[i][1:]:
            return True
        for j in range(self.n_snakes):
            if i == j:
                continue
            if pt in self.snake[j]:
                return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for i in range(len(self.snake)):
            snake = self.snake[i]
            colors = [(0, 0, 255 - i * 100), (0, 0, 255 - i * 100 - 20)]
            for pt in snake:
                pygame.draw.rect(
                    self.display,
                    colors[0],
                    pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE),
                )
                pygame.draw.rect(
                    self.display, colors[1], pygame.Rect(pt.x + 4, pt.y + 4, 12, 12)
                )

        pygame.draw.rect(
            self.display,
            RED,
            pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE),
        )

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, i, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction[i])

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

        self.direction[i] = new_dir

        x = self.head[i].x
        y = self.head[i].y
        if self.direction[i] == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction[i] == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction[i] == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction[i] == Direction.UP:
            y -= BLOCK_SIZE

        self.head[i] = Point(x, y)
