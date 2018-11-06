import numpy as np
from typing import List

from God.Bird import Bird


class Sky:
    birds: List[Bird]

    def __init__(self, L: float, gridstep: float):
        self.L = L
        self.gridstep = gridstep

        # Generate grid
        self.gridL = int(np.ceil(L / gridstep))
        self.grid = None
        self.init_grid()

        self.birds = []

    def truedistance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        diff = v1 - v2
        return np.sqrt(min(diff[0], self.L - diff[0]) ** 2 + min(diff[1], self.L - diff[1]) ** 2)

    def get_avg_speed(self) -> float:
        return np.linalg.norm(np.mean([bird.speedV for bird in self.birds], axis=0))

    def get_avg_angle(self) -> float:
        median_cos = 0
        median_sin = 0
        for bird in self.birds:
            median_cos += np.cos(bird.angle)
            median_sin += np.sin(bird.angle)
        return np.arctan2(median_sin, median_cos)

    def get_angles_correlations(self, n: int = np.inf) -> tuple:
        pos = np.array([bird.pos for bird in self.birds])
        speedV = np.array([bird.speedV for bird in self.birds])

        n_birds = len(self.birds)
        n = int(min(n, n_birds * (n_birds - 1) / 2))
        indices_i, indices_j = np.triu_indices(n_birds, k=1)
        chosen_indices = list(range(n))
        np.random.shuffle(chosen_indices)
        chosen_indices = chosen_indices[:n]
        indices_i, indices_j = [indices_i[chosen_indices[i]] for i in range(n)], [indices_j[chosen_indices[i]] for i in
                                                                                  range(n)]

        correlations = []
        distances = []

        for i, j in zip(indices_i, indices_j):
            correlations.append(np.dot(speedV[i], speedV[j]))

        for i, j in zip(indices_i, indices_j):
            distances.append(self.truedistance(pos[i], pos[j]))

        return np.array(sorted(distances)), np.array([x for _, x in sorted(zip(distances, correlations))])

    def add_bird(self, bird: Bird) -> None:
        self.birds.append(bird)

    def init_grid(self) -> None:
        self.grid = np.empty((self.gridL, self.gridL), dtype=object)

    def update_grid(self) -> None:
        self.init_grid()
        for bird in self.birds:
            gridpos = bird.pos // self.gridstep
            i, j = int(gridpos[0]), int(gridpos[1])
            if self.grid[i, j] is None:
                self.grid[i, j] = []
            self.grid[i, j].append(bird)

    def add_n_random_birds(self, n: int, vel: float, ang_vel: float) -> None:
        for _ in range(n):
            theta = np.random.rand() * np.pi * 2
            bird = Bird(np.random.rand(2) * self.L, vel, ang_vel, theta)
            self.add_bird(bird)

    def vel_max(self) -> float:
        vel_max = 0
        for bird in self.birds:
            if (vel_max < bird.vel):
                vel_max = bird.vel
        return vel_max
