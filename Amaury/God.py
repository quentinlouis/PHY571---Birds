import numpy as np
import itertools
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
from matplotlib.gridspec import GridSpec
import cmocean


class Bird:
    def __init__(self, pos: np.ndarray, vel: float, angle: float):
        self.vel = vel
        self.angle = angle
        self.pos = pos
        self.lookV = None
        self.speedV = None
        self.update_calculated_props()

    def __repr__(self):
        return "Bird(%.2f, %.2f)" % (self.pos[0], self.pos[1])

    def update_calculated_props(self):
        self.lookV = np.array([np.cos(self.angle), np.sin(self.angle)])
        self.speedV = self.lookV * self.vel


class Sky:
    def __init__(self, L: float, gridstep: float):
        self.L = L
        self.gridstep = gridstep

        # Generate grid
        self.gridL = int(np.ceil(L/gridstep))
        self.grid = None
        self.init_grid()

        self.birds = []

    def add_bird(self, bird: Bird):
        self.birds.append(bird)

    def init_grid(self):
        self.grid = np.zeros((self.gridL, self.gridL), dtype=object)
        for i, j in itertools.product(range(self.gridL), range(self.gridL)):
            self.grid[i, j] = []

    def update_grid(self):
        self.init_grid()
        for bird in self.birds:
            gridpos = bird.pos // self.gridstep
            self.grid[int(gridpos[0]), int(gridpos[1])].append(bird)

    def add_n_random_birds(self, n: int, vel: float):
        for _ in range(n):
            theta = np.random.rand() * np.pi * 2
            bird = Bird(np.random.rand(2) * self.L, vel, theta)
            self.add_bird(bird)

    def plot(self):
        pos = np.array([bird.pos for bird in self.birds])
        angles = np.array([bird.angle for bird in self.birds])
        plt.quiver(pos[:, 0], pos[:, 1], np.cos(angles), np.sin(angles))
        plt.show()

class Physics:
    def __init__(self, sky: Sky, interaction_radius: float, eta: float):
        self.sky = sky; sky.update_grid()
        self.interaction_radius = interaction_radius
        self.eta = eta

    def advance(self, dt: float):
        all_final_interactions = []
        for bird in self.sky.birds:
            # Collect all bird in interaction range
            interact_with = []
            gridpos = bird.pos // self.sky.gridstep
            grid_interaction_radius = self.interaction_radius / self.sky.gridstep

            grid_xmin = int(gridpos[0]-grid_interaction_radius)
            grid_xmax = int(np.ceil(gridpos[0]+grid_interaction_radius))
            grid_ymin = int(gridpos[1] - grid_interaction_radius)
            grid_ymax = int(np.ceil(gridpos[1] + grid_interaction_radius))

            for i, j in itertools.product(range(grid_xmin, grid_xmax + 1), range(grid_ymin, grid_ymax + 1)):
                interact_with += self.sky.grid[i % self.sky.gridL, j % self.sky.gridL]


            # Apply interaction
            # Enforce radius
            interact_with_close = []
            for other_bird in interact_with:
                distance = np.linalg.norm(other_bird.pos - bird.pos)
                if distance <= self.interaction_radius:
                    interact_with_close.append(other_bird)

            all_final_interactions.append([bird, interact_with_close])
            median_angle = 0
            for other_bird in interact_with_close:
                median_angle += other_bird.angle
            median_angle = median_angle / len(interact_with_close)
            delta_angle = self.eta * (np.random.rand()-.5) * 2
            bird.angle = (median_angle + delta_angle) % (2 * np.pi)

        # Verlet movement *after* updating directions
        for bird in self.sky.birds:
            bird.update_calculated_props()
            bird.pos = (bird.pos + bird.speedV * dt) % L

        sky.update_grid()
        return all_final_interactions

    def animate(self, total_time: float, dt: float):
        Writer = animation.writers['ffmpeg']  # to save video
        writer = Writer(fps=15, bitrate=-1)  # to save video

        gs = GridSpec(2, 2)
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        fig.set_size_inches(10, 10, True)

        time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
        timestamps = np.arange(0, total_time, dt)

        norm = matplotlib.colors.Normalize(vmin=0, vmax=2 * np.pi)
        cm = cmocean.cm.phase

        pos = np.array([bird.pos for bird in self.sky.birds])
        angles = np.array([bird.angle for bird in self.sky.birds])
        Q = ax1.quiver(pos[:, 0], pos[:, 1], np.cos(angles), np.sin(angles), cmap=cm)
        ax1.set_xlim(0, self.sky.L)
        ax1.set_ylim(0, self.sky.L)

        interaction_lines = []
        def update_quiver(num, Q, time_text):
            """updates the horizontal and vertical vector components by a
            fixed increment on each frame
            """
            interactions = self.advance(dt)

            pos = np.array([bird.pos for bird in self.sky.birds])
            angles = np.array([bird.angle for bird in self.sky.birds])

            X, Y = pos[:, 0], pos[:, 1]
            N = len(self.sky.birds)
            new_offsets = np.zeros((N, 2))
            for i in range(N):
                new_offsets[i][0] = X[i]
                new_offsets[i][1] = Y[i]
            U = np.cos(angles)
            V = np.sin(angles)
            Q.set_offsets(new_offsets)
            Q.set_UVC(U, V, norm(angles))

            time_text.set_text('time = %.1f s' % round(dt * num, 3))

            for line in interaction_lines:
                line.remove()
                del line
            interaction_lines[:] = []
            for interaction in interactions:
                bird, other_birds = interaction[0], interaction[1]
                for other_bird in other_birds:
                    if bird != other_bird:
                        interaction_lines.append(ax1.plot([bird.pos[0],other_bird.pos[0]], [bird.pos[1],other_bird.pos[1]], color="red")[0])

            return (Q, time_text) + tuple(interaction_lines)

        anim = animation.FuncAnimation(fig, update_quiver, frames=int(total_time / dt), fargs=(Q, time_text),
                                       interval=200, blit=False)
        anim.save('lines.mp4', writer=writer)
        plt.show()



L = 10
gridstep = 1
sky = Sky(L, gridstep)
sky.add_n_random_birds(20, 1)

physics = Physics(sky, 1, .1)

physics.animate(4, .05)



























































