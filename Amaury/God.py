import numpy as np
import itertools
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors
from matplotlib.gridspec import GridSpec
import cmocean
import scipy.optimize
import time
import datetime
import json


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def short_angle_dist(a0: float, a1: float):
    da = (a1 - a0)
    return (da + np.pi) % (2 * np.pi) - np.pi


class Bird:
    def __init__(self, pos: np.ndarray, vel: float, ang_vel: float, angle: float):
        self.vel = vel
        self.angle = angle
        self.ang_vel = ang_vel
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

    def get_avg_speed(self):
        angles = np.array([bird.angle for bird in self.birds])
        velocities = np.array([bird.vel for bird in self.birds])
        vel_x_mean, vel_y_mean = np.mean(np.cos(angles)* velocities), np.mean(np.sin(angles)* velocities)
        return np.sqrt(np.dot(vel_x_mean, vel_x_mean) ** 2 + np.dot(vel_y_mean, vel_y_mean) ** 2)

    def get_avg_angle(self):
        median_cos = 0
        median_sin = 0
        for bird in self.birds:
            median_cos += np.cos(bird.angle)
            median_sin += np.sin(bird.angle)
        return np.arctan2(median_sin, median_cos)

    def get_angles_correlations(self, n: int=np.inf):
        pos = np.array([bird.pos for bird in self.birds])
        speedV = np.array([bird.speedV for bird in self.birds])

        n_birds = len(self.birds)
        n = int(min(n, n_birds * (n_birds-1) / 2))
        indices_i, indices_j = np.triu_indices(n_birds, k=1)
        chosen_indices = list(range(n))
        np.random.shuffle(chosen_indices)
        chosen_indices = chosen_indices[:n]
        indices_i, indices_j = [indices_i[chosen_indices[i]] for i in range(n)], [indices_j[chosen_indices[i]] for i in range(n)]

        correlations = []
        distances = []

        for i,j in zip(indices_i, indices_j):
            correlations.append(np.dot(speedV[i], speedV[j]))

        for i, j in zip(indices_i, indices_j):
            distances.append(np.linalg.norm(pos[i] - pos[j]))

        return np.array(sorted(distances)), np.array([x for _, x in sorted(zip(distances, correlations))])

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

    def add_n_random_birds(self, n: int, vel: float, ang_vel: float):
        for _ in range(n):
            theta = np.random.rand() * np.pi * 2
            bird = Bird(np.random.rand(2) * self.L, vel, ang_vel, theta)
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
        sky.update_grid()
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

            median_cos = 0
            median_sin = 0
            for other_bird in interact_with_close:
                median_cos += np.cos(other_bird.angle)
                median_sin += np.sin(other_bird.angle)
            median_angle = np.arctan2(median_sin, median_cos)
            diff = short_angle_dist(bird.angle, median_angle)
            t_move = abs(diff) / bird.ang_vel
            t_move = min(dt, t_move)

            fluctuation_angle = bird.ang_vel * self.eta * (np.random.rand()-.5) * dt
            bird.angle = (bird.angle + np.sign(diff) * bird.ang_vel * t_move + fluctuation_angle) % (2 * np.pi)

        # Verlet movement *after* updating directions
        for bird in self.sky.birds:
            bird.update_calculated_props()
            bird.pos = (bird.pos + bird.speedV * dt) % L


class Life:
    def __init__(self, physics: Physics, dt: float, total_time: float):
        self.dt = dt
        self.total_time = total_time
        self.physics = physics

        self.writer = animation.writers['ffmpeg'](fps=15, bitrate=-1)  # to save video

    def simulate(self, verbose_prop: float=.01, output_file: str="data.json"):
        timestamps = np.arange(0, self.total_time, self.dt)
        total_frames = len(timestamps)
        frames = []

        start_t = time.time()
        print("Simulation start at t=%s" % datetime.datetime.fromtimestamp(start_t).strftime('%Y-%m-%d %H:%M:%S'))
        frame_n = 0
        for _ in timestamps:
            frame_n += 1
            if frame_n % (1+int(total_frames*verbose_prop)) == 0:
                print("Simulating frame %d/%d" % (frame_n, total_frames))

            self.physics.advance(self.dt)

            frames.append([[bird.pos, bird.angle, bird.vel, bird.ang_vel] for bird in self.physics.sky.birds])

        elapsed = time.time() - start_t
        print("Simulation ended at t=%s, elapsed: %dh %dm %ds. N=%d, L=%d, eta=%.2f, v=%.1f, omega=%.1f, T=%d, dt=%.1f" % (
            datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
            elapsed // 3600 % 24, elapsed // 60 % 60, elapsed % 60,
            len(self.physics.sky.birds), self.physics.sky.L, self.physics.eta, self.physics.sky.birds[0].vel, self.physics.sky.birds[0].ang_vel, self.total_time,
            self.dt))

        data_to_save = {"frames": frames, "parameters": [self.dt, self.total_time, self.physics.sky.L, self.physics.eta, self.physics.interaction_radius]}
        with open(output_file, "w") as f:
            json.dump(data_to_save, f, cls=NumpyEncoder)

    def animate(self, verbose_prop: float=.01, to_draw: list=[], input_file: str="data.json", output_file: str="lines.mp4", options: dict={}):
        # load data
        with open(input_file, "r") as f:
            data = json.load(f)
        frames = data["frames"]
        dt, total_time, L, eta, interaction_radius = data["parameters"]
        print("Got %d frames" % len(frames))

        # chose what to draw
        draw_quiver = "quiver" in to_draw
        draw_avg_speed = "avg_speed" in to_draw
        draw_avg_angle = "avg_angle" in to_draw
        draw_correlations = "correlations" in to_draw
        draw_correlations_fit = "correlations_fit" in to_draw

        # setup axes
        gs = GridSpec(2, 2)
        fig = plt.figure()
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax2_2 = ax2.twinx()
        ax3_2 = ax3.twinx()
        fig.set_size_inches(10, 10, True)

        # text and time
        time_text = ax1.text(0.02, 1.05, '', transform=ax1.transAxes)
        correlations_text = ax3.text(0.02, 1.05, '', transform=ax3.transAxes)
        timestamps = np.arange(0, total_time, dt)
        total_frames = len(timestamps)

        # colors
        norm = matplotlib.colors.Normalize(vmin=0, vmax=2 * np.pi)
        cm = cmocean.cm.phase

        # quiver
        ax1.set_xlim(0, L)
        ax1.set_ylim(0, L)

        # avg_speed
        ax2.set_xlim(0, total_time)
        ax2.set_ylim(0, 1)
        if draw_avg_speed:
            avg_speed, = ax2.plot([], [], lw=2, label="speed")
            ax2.set_xlabel("time (s)")
            ax2.set_ylabel("avg. speed (m/s)")
            avg_speeds = []

        # avg_angle
        ax2_2.set_ylim(-np.pi, np.pi)
        if draw_avg_angle:
            avg_angle, = ax2_2.plot([], [], lw=2, color="orange", label="angle")
            ax2_2.set_ylabel("avg. angle (rad)")
            avg_angles = []

        # correlations
        ax3.set_xlim(0, L)
        ax3.set_ylim(0, 1)
        ax3_2.set_ylim(0, L/3)
        if draw_correlations:
            all_corr, = ax3.plot([], [], lw=2)
            if draw_correlations_fit:
                fitted_corr, = ax3.plot([], [], lw=2)
                fitted_corr_length, = ax3_2.plot([], [], lw=2, label="correlation length", color="green")
                corr_lengths = []
                ax3_2.set_ylabel("correlation length (m)")
            ax3.set_xlabel("distance (m)")
            ax3.set_ylabel("correlation (normed)")

        to_unblit = []

        def update_animation(num):
            artists = []
            if num % (1+int(total_frames*verbose_prop)) == 0:
                print("Drawing frame %d/%d" % (num, total_frames))

            # unblit artists created on the spot
            for artist in list(to_unblit):
                to_unblit.remove(artist)
                artist.remove()
                del artist

            frame = np.array(frames[num])
            passed_time = timestamps[:num + 1]

            sky = Sky(L, L)
            for bird in frame:
                sky.add_bird(Bird(bird[0], bird[2], bird[3], bird[1]))

            positions = np.array([bird.pos for bird in sky.birds])
            angles = np.array([bird.angle for bird in sky.birds])

            if draw_quiver:
                x, y = positions[:, 0], positions[:, 1]
                n = len(frame)
                new_offsets = np.zeros((n, 2))
                for i in range(n):
                    new_offsets[i][0] = x[i]
                    new_offsets[i][1] = y[i]
                u = np.cos(angles)
                v = np.sin(angles)
                quiver = ax1.quiver(x, y, u, v, norm(angles), cmap=cm, angles='xy', scale_units='xy', scale=1)
                artists.append(quiver)
                to_unblit.append(quiver)

            if draw_avg_speed:
                avg_speeds.append(sky.get_avg_speed())
                avg_speed.set_data(passed_time, avg_speeds[:num + 1])
                artists.append(avg_speed)

            if draw_correlations:
                correlations_stochastic_points = options.get("correlations_stochastic_points", 2000)
                dists, corrs = sky.get_angles_correlations(n=correlations_stochastic_points)
                space_points = options.get("fit_spatial_points", 100)
                regular_dists = np.linspace(0, L, space_points)
                regular_corrs = []

                for i in range(len(regular_dists) - 1):
                    mask = np.logical_and(dists > regular_dists[i], dists < regular_dists[i + 1])
                    if len(corrs[mask]) == 0:
                        mean = 0
                    else:
                        mean = np.mean(corrs[mask])
                    regular_corrs.append(mean)
                regular_corrs = np.array(regular_corrs)
                regular_dists = np.array(regular_dists[:-1])
                all_corr.set_data(regular_dists, regular_corrs)
                artists.append(all_corr)

                correlations_text_value = "n_points: %d, stochastic correlation on %d points" % (space_points, correlations_stochastic_points)
                if draw_correlations_fit:
                    def func_fit(x, a, zeta):
                        return a * np.exp(- x / zeta)

                    try:
                        popt, _ = scipy.optimize.curve_fit(func_fit, regular_dists, regular_corrs)
                        corr_lengths.append(popt[1])
                        fitted_corr.set_data(regular_dists, func_fit(regular_dists, *popt))
                        artists.append(fitted_corr)
                        fitted_corr_length.set_data(passed_time*L/total_time, corr_lengths[:num + 1])
                        artists.append(fitted_corr_length)

                        residuals = regular_corrs - func_fit(regular_dists, *popt)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((regular_corrs - np.mean(regular_corrs)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        correlations_text_value += "\n$R^2$ = %.5f" % r_squared
                    except Exception as e:
                        corr_lengths.append(0)
                        print("Exception in fit: %s" % e)

                correlations_text.set_text(correlations_text_value)
                artists.append(correlations_text)

            if draw_avg_angle:
                avg_angles.append(sky.get_avg_angle())
                avg_angle.set_data(passed_time, avg_angles[:num + 1])
                artists.append(avg_angle)

            time_text.set_text('time = %.1f s - params: N=%d, L=%d, eta=%.2f, v=%.1f, omega=%.1f, T=%d, dt=%.1f' %
                               (round(dt * num, 3),
                                len(sky.birds), L, eta, sky.birds[0].vel,
                                sky.birds[0].ang_vel, total_time, dt))
            artists.append(time_text)

            if draw_avg_speed:
                ax2.legend(loc=1)
            if draw_avg_angle:
                ax2_2.legend(loc=2)
            if draw_correlations_fit:
                ax3_2.legend(loc=2)
            plt.tight_layout()

            return artists

        start_t = time.time()
        print("Drawing start at t=%s" % datetime.datetime.fromtimestamp(start_t).strftime('%Y-%m-%d %H:%M:%S'))

        anim = animation.FuncAnimation(fig, update_animation, frames=total_frames, interval=200, blit=True, repeat=True)
        anim.save(output_file, writer=self.writer)
        elapsed = time.time() - start_t
        print(
            "Drawing ended at t=%s, elapsed: %dh %dm %ds. N=%d, L=%d, eta=%.2f, v=%.1f, omega=%.1f, T=%d, dt=%.1f" % (
            datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
            elapsed // 3600 % 24, elapsed // 60 % 60, elapsed % 60,
            len(sky.birds), L, eta, sky.birds[0].vel, sky.birds[0].ang_vel, total_time, dt))


L = 70
gridstep = .5
n_birds = 1000
vel = 1
ang_vel = np.pi/2
interaction_radius = 1
eta = .5

dt = 1
total_time = 500

sky = Sky(L, gridstep)
sky.add_n_random_birds(n_birds, vel, ang_vel)
physics = Physics(sky, interaction_radius, eta)
life = Life(physics, dt, total_time)

life.simulate(verbose_prop=.1)
life.animate(verbose_prop=.05, to_draw=["quiver", "avg_speed", "avg_angle", "correlations", "correlations_fit"], options={"fit_spatial_points": 100,
                                                                                                                          "correlations_stochastic_points": 5000})


























































