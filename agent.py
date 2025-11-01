import torch
import random
import numpy as np
from collections import deque
from game import SnakeGame, Direction, Point
from model import QNet, ModelTrainer
from helper import plot


class Agent:
    def __init__(self, agent_id, state_size, action_size):
        self.id = agent_id
        self.n_games = 0
        self.epsilon = 0.0
        self.gamma = 0.9
        self.lr = 0.001
        self.max_mem = 100_000
        self.batch_size = 1_000

        self.memory = deque(maxlen=self.max_mem)
        self.model = QNet(state_size, 256, action_size)
        self.trainer = ModelTrainer(self.model, lr=self.lr, gamma=self.gamma)

    def get_state(self, game, snake_id):
        head = game.snake[snake_id][0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        dir_l = game.direction[snake_id] == Direction.LEFT
        dir_r = game.direction[snake_id] == Direction.RIGHT
        dir_u = game.direction[snake_id] == Direction.UP
        dir_d = game.direction[snake_id] == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(snake_id, point_r))
            or (dir_l and game.is_collision(snake_id, point_l))
            or (dir_u and game.is_collision(snake_id, point_u))
            or (dir_d and game.is_collision(snake_id, point_d)),
            # Danger right
            (dir_u and game.is_collision(snake_id, point_r))
            or (dir_d and game.is_collision(snake_id, point_l))
            or (dir_l and game.is_collision(snake_id, point_u))
            or (dir_r and game.is_collision(snake_id, point_d)),
            # Danger left
            (dir_d and game.is_collision(snake_id, point_r))
            or (dir_u and game.is_collision(snake_id, point_l))
            or (dir_r and game.is_collision(snake_id, point_u))
            or (dir_l and game.is_collision(snake_id, point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food location
            game.food.x < game.head[snake_id].x,  # food left
            game.food.x > game.head[snake_id].x,  # food right
            game.food.y < game.head[snake_id].y,  # food up
            game.food.y > game.head[snake_id].y,  # food down
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > self.batch_size:
            mini_sample = random.sample(self.memory, self.batch_size)
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, training=True):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if training and random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            pred = self.model(state0)
            move = torch.argmax(pred).item()
            final_move[move] = 1
        return final_move


def train(n_agents=1, model_file=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agents = [Agent(i, 11, 3) for i in range(n_agents)]
    game = SnakeGame(n_snakes=n_agents)
    if model_file is not None:
        state_dict = torch.load(model_file)
        for agent in agents:
            agent.model.load_state_dict(state_dict)

    while True:
        states_old = [
            agents[i].get_state(game, i) for i in range(n_agents)
        ]
        final_moves = [
            agents[i].get_action(states_old[i]) for i in range(n_agents)
        ]

        rewards, dones, scores = game.play_step(final_moves)
        states_new = [
            agents[i].get_state(game, i) for i in range(n_agents)
        ]

        for i in range(n_agents):
            agents[i].train_short_memory(
                states_old[i], final_moves[i], rewards[i], states_new[i], dones[i]
            )
            agents[i].remember(
                states_old[i], final_moves[i], rewards[i], states_new[i], dones[i]
            )

        if any(dones):
            for i in range(n_agents):
                if dones[i]:
                    if scores[i] > record:
                        record = scores[i]
                        agents[i].model.save(file_name=f"model_{i}.pth")
                        print(f"Saved model for agent {i}")

                    game.reset()
                    agents[i].n_games += 1
                    agents[i].train_long_memory()

                    print(
                        "Game",
                        agents[i].n_games,
                        "Score",
                        scores[i],
                        "Record:",
                        record,
                    )

                    plot_scores.append(scores[i])
                    total_score += scores[i]
                    mean_score = total_score / agents[i].n_games
                    plot_mean_scores.append(mean_score)
                    plot(plot_scores, plot_mean_scores)


def test(n_agents=1, model_file="models/best.pth"):
    agents = [Agent(i, 11, 3) for i in range(n_agents)]
    game = SnakeGame(n_snakes=n_agents)
    state_dict = torch.load(model_file)
    for agent in agents:
        agent.model.load_state_dict(state_dict)
    while True:
        states_old = [
            agents[i].get_state(game, i) for i in range(n_agents)
        ]
        final_moves = [
            agents[i].get_action(states_old[i], training=False)
            for i in range(n_agents)
        ]
        rewards, dones, scores = game.play_step(final_moves)
        if any(dones):
            game.reset()


if __name__ == "__main__":
    n_agents = int(input("Enter number of agents: "))
    mode = input("1.Train\n2.Test\nEnter choice:")
    if mode == "1":
        train(n_agents, "models/model.pth")
    elif mode == "2":
        test(n_agents, "models/best.pth")
    else:
        print("Invalid choice!")
