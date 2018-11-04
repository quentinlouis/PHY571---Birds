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

from Amaury.Pandora import short_angle_dist, NumpyEncoder


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

    def truedistance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        diff = v1-v2
        return np.sqrt(min(diff[0], self.L-diff[0])**2+min(diff[1], self.L-diff[1])**2)

    def get_avg_speed(self) -> float:
        angles = np.array([bird.angle for bird in self.birds])
        velocities = np.array([bird.vel for bird in self.birds])
        vel_x_mean, vel_y_mean = np.mean(np.cos(angles)* velocities), np.mean(np.sin(angles)* velocities)
        return np.sqrt(np.dot(vel_x_mean, vel_x_mean) ** 2 + np.dot(vel_y_mean, vel_y_mean) ** 2)

    def get_avg_angle(self) -> float:
        median_cos = 0
        median_sin = 0
        for bird in self.birds:
            median_cos += np.cos(bird.angle)
            median_sin += np.sin(bird.angle)
        return np.arctan2(median_sin, median_cos)

    def get_angles_correlations(self, n: int=np.inf) -> tuple:
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
            distances.append(self.truedistance(pos[i], pos[j]))

        return np.array(sorted(distances)), np.array([x for _, x in sorted(zip(distances, correlations))])

    def add_bird(self, bird: Bird) -> None:
        self.birds.append(bird)

    def init_grid(self) -> None:
        self.grid = np.zeros((self.gridL, self.gridL), dtype=object)
        for i, j in itertools.product(range(self.gridL), range(self.gridL)):
            self.grid[i, j] = []

    def update_grid(self) -> None:
        self.init_grid()
        for bird in self.birds:
            gridpos = bird.pos // self.gridstep
            self.grid[int(gridpos[0]), int(gridpos[1])].append(bird)

    def add_n_random_birds(self, n: int, vel: float, ang_vel: float) -> None:
        for _ in range(n):
            theta = np.random.rand() * np.pi * 2
            bird = Bird(np.random.rand(2) * self.L, vel, ang_vel, theta)
            self.add_bird(bird)

    def plot(self) -> None:
        pos = np.array([bird.pos for bird in self.birds])
        angles = np.array([bird.angle for bird in self.birds])
        plt.quiver(pos[:, 0], pos[:, 1], np.cos(angles), np.sin(angles))
        plt.show()


class Physics:
    def __init__(self, sky: Sky, interaction_radius: float, eta: float):
        self.sky = sky; sky.update_grid()
        self.interaction_radius = interaction_radius
        self.eta = eta

    def get_groups(self) -> tuple:
        interactions = {}
        for bird in self.sky.birds:
            # Collect all bird in interaction range
            interact_with_close = self.get_interact_with_radius(bird)

            # Check angle difference, and not same one
            for other_bird in list(interact_with_close):
                angle_difference = abs(short_angle_dist(bird.angle, other_bird.angle))
                if angle_difference > self.eta or other_bird == bird:
                    interact_with_close.remove(other_bird)

            interactions[bird] = interact_with_close

        # find groups
        bird_to_group = {}
        groups_to_bird = []
        curr_group = 0

        def connect(bird, group):
            if bird in bird_to_group and bird_to_group[bird] == group:
                return
            bird_to_group[bird] = group
            groups_to_bird[group].append(bird)

        for bird in interactions:
            other_birds = interactions[bird]
            for other_bird in other_birds:
                # if another bird has a group
                if other_bird in bird_to_group:
                    new_group = bird_to_group[other_bird]
                    # make current bird in group
                    connect(bird, new_group)
                    # look at other connected birds
                    for other_other_bird in other_birds:
                        if other_other_bird not in bird_to_group:
                            # if they're not in a group, connect
                            connect(other_other_bird, new_group)
                        else:
                            # otherwise connect all the members of their group to the new group
                            other_new_group = bird_to_group[other_other_bird]
                            if other_new_group != new_group:
                                for bird_in_group in groups_to_bird[other_new_group]:
                                    connect(bird_in_group, new_group)
                                groups_to_bird[other_new_group] = []
                    break

            if bird not in bird_to_group:
                groups_to_bird.append([])
                connect(bird, curr_group)
                curr_group += 1
            group = bird_to_group[bird]
            for other_bird in other_birds:
                connect(other_bird, group)

        return groups_to_bird, bird_to_group

    def get_interact_with(self, bird: Bird) -> list:
        interact_with = []
        gridpos = bird.pos // self.sky.gridstep
        grid_interaction_radius = self.interaction_radius / self.sky.gridstep

        grid_xmin = int(gridpos[0] - grid_interaction_radius)
        grid_xmax = int(np.ceil(gridpos[0] + grid_interaction_radius))
        grid_ymin = int(gridpos[1] - grid_interaction_radius)
        grid_ymax = int(np.ceil(gridpos[1] + grid_interaction_radius))

        for i, j in itertools.product(range(grid_xmin, grid_xmax + 1), range(grid_ymin, grid_ymax + 1)):
            interact_with += self.sky.grid[i % self.sky.gridL, j % self.sky.gridL]

        return interact_with

    def get_interact_with_radius(self, bird: Bird) -> list:
        interact_with = self.get_interact_with(bird)

        interact_with_close = []
        for other_bird in interact_with:
            distance = self.sky.truedistance(other_bird.pos, bird.pos)
            if distance <= self.interaction_radius:
                interact_with_close.append(other_bird)

        return interact_with_close

    def advance(self, dt: float) -> None:
        self.sky.update_grid()
        for bird in self.sky.birds:
            # Collect all bird in interaction range
            interact_with_close = self.get_interact_with_radius(bird)

            # Apply interaction
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
            bird.pos = (bird.pos + bird.speedV * dt) % self.sky.L


class Life:
    def __init__(self):
        self.writer = animation.writers['ffmpeg'](fps=15, bitrate=-1)  # to save video

    def simulate(self, physics: Physics, dt: float, total_time: float, verbose_prop: float=.01, output_file: str="data.json", evolve=lambda sky: None):
        timestamps = np.arange(0, total_time, dt)
        total_frames = len(timestamps)
        frames = []

        start_t = time.time()
        print("Simulation start at t=%s" % datetime.datetime.fromtimestamp(start_t).strftime('%Y-%m-%d %H:%M:%S'))
        frame_n = 0
        for _ in timestamps:
            frame_n += 1
            if frame_n % (1+int(total_frames*verbose_prop)) == 0:
                print("Simulating frame %d/%d" % (frame_n, total_frames))

            if evolve is not None:
                evolve(physics.sky)

            physics.advance(dt)

            frames.append([[bird.pos, bird.angle, bird.vel, bird.ang_vel] for bird in physics.sky.birds])

        elapsed = time.time() - start_t
        print("Simulation ended at t=%s, elapsed: %dh %dm %ds. N=%d, L=%d, eta=%.2f, v=%.1f, omega=%.1f, T=%d, dt=%.1f" % (
            datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
            elapsed // 3600 % 24, elapsed // 60 % 60, elapsed % 60,
            len(physics.sky.birds), physics.sky.L, physics.eta, physics.sky.birds[0].vel, physics.sky.birds[0].ang_vel, total_time,
            dt))

        data_to_save = {"frames": frames, "parameters": [dt, total_time, physics.sky.L, physics.eta, physics.interaction_radius]}
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
        draw_avg_polar = "avg_polar" in to_draw
        draw_correlations = "correlations" in to_draw
        draw_correlations_fit = "correlations_fit" in to_draw
        draw_groups = "groups" in to_draw
        draw_avg_groups = "avg_groups" in to_draw

        # setup axes
        gs = GridSpec(2, 4)
        fig = plt.figure()
        ax_quiver = fig.add_subplot(gs[0:2, 0:2])
        ax_speed = fig.add_subplot(gs[0, 2])
        ax_groups = fig.add_subplot(gs[1, 3])
        ax_correlation = fig.add_subplot(gs[1, 2])
        ax_polar = fig.add_subplot(gs[0, 3], projection='polar')
        ax_angle = ax_speed.twinx()
        ax_correlation_length = ax_correlation.twinx()
        fig.set_size_inches(18, 10, True)

        # text and time
        time_text = ax_quiver.text(0.02, 1.05, '', transform=ax_quiver.transAxes)
        correlations_text = ax_correlation.text(0.02, 1.05, '', transform=ax_correlation.transAxes)
        groups_text = ax_groups.text(0.02, 1.05, '', transform=ax_groups.transAxes)
        timestamps = np.arange(0, total_time, dt)
        total_frames = len(timestamps)

        # colors
        norm = matplotlib.colors.Normalize(vmin=0, vmax=2 * np.pi)
        cm = cmocean.cm.phase

        # quiver
        ax_quiver.set_xlim(0, L)
        ax_quiver.set_ylim(0, L)

        # groups
        max_group_size = options.get("max_group_size", 20)
        max_num_groups = options.get("max_num_groups", 20)
        ax_groups.set_xlim(1, max_group_size)
        ax_groups.set_ylim(.1, max_num_groups)
        ax_groups.set_xscale("log")
        ax_groups.set_yscale("log")
        if draw_groups:
            groups_line, = ax_groups.plot([], [], marker="o", lw=2)
            ax_groups.set_xlabel("group size")
            ax_groups.set_ylabel("# of groups")

        if draw_avg_groups:
            avg_groups_line, = ax_groups.plot([], [], marker="o", lw=2, label="average")
            avg_groups_fit, = ax_groups.plot([], [], lw=2, label="fit")
            avg_groups = np.array([1]+[.1 for _ in range(max_group_size)])

        # avg_speed
        ax_speed.set_xlim(0, total_time)
        ax_speed.set_ylim(0, 1)
        if draw_avg_speed:
            avg_speed, = ax_speed.plot([], [], lw=2, label="speed")
            ax_speed.set_xlabel("time (s)")
            ax_speed.set_ylabel("avg. speed (m/s)")
            avg_speeds = []

        # avg_angle
        ax_angle.set_ylim(-np.pi, np.pi)
        if draw_avg_angle:
            avg_angle, = ax_angle.plot([], [], lw=2, color="orange", label="angle")
            ax_angle.set_ylabel("avg. angle (rad)")
            avg_angles = []

        # correlations
        ax_correlation.set_xlim(0, L)
        ax_correlation.set_ylim(0, 1)
        ax_correlation_length.set_ylim(0, L/3)
        if draw_correlations:
            all_corr, = ax_correlation.plot([], [], lw=2)
            if draw_correlations_fit:
                fitted_corr, = ax_correlation.plot([], [], lw=2)
                fitted_corr_length, = ax_correlation_length.plot([], [], lw=2, label="correlation length", color="green")
                corr_lengths = []
                ax_correlation_length.set_ylabel("correlation length (m)")
            ax_correlation.set_xlabel("distance (m)")
            ax_correlation.set_ylabel("correlation (normed)")

        # avg_polar
        if draw_avg_polar:
            ax_polar.set_title("Avg. angle and speed")
            avg_polar, = ax_polar.plot([], [], lw=2)
            avg_polar_end, = ax_polar.plot([], [], lw=2, marker="o", markersize=6, color="red")
            ax_polar.set_rmax(1)

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
                sky.add_bird(Bird(np.array(bird[0]), bird[2], bird[3], bird[1]))

            if draw_groups:
                sky.gridstep = interaction_radius/2
                sky.gridL = int(np.ceil(L / sky.gridstep))
                physics = Physics(sky, interaction_radius, eta)

            positions = np.array([bird.pos for bird in sky.birds])
            angles = np.array([bird.angle for bird in sky.birds])
            quiver_colors = norm(angles)

            if draw_groups or draw_avg_groups:
                groups, bird_to_group = physics.get_groups()
                np.random.seed(11121997)
                group_colors = np.random.rand(len(groups))
                quiver_colors = []
                for bird in sky.birds:
                    quiver_colors.append(group_colors[bird_to_group[bird]])

                size_groups = [len(group) for group in groups if len(group) > 0]
                max_size_group = np.max(size_groups)
                size_x = range(1, max_size_group+1)
                if draw_groups:
                    groups_line.set_data(size_x, [size_groups.count(i) for i in size_x])
                    artists.append(groups_line)
                if draw_avg_groups:
                    avg_groups[0] += 1  # stock number of times computed here
                    for i in range(len(avg_groups)):
                        avg_groups[i] += size_groups.count(i)
                    size_x = np.array(range(1, len(avg_groups)))
                    size_y = avg_groups[1:] / avg_groups[0]
                    avg_groups_line.set_data(size_x, size_y)
                    artists.append(avg_groups_line)

                    # linear fit:
                    try:
                        fit = lambda x,a,b: b*x**a
                        popt, _ = scipy.optimize.curve_fit(fit, size_x, size_y)
                        a, b = popt
                        avg_groups_fit.set_data(size_x, fit(size_x, a, b))
                        artists.append(avg_groups_fit)

                        residuals = size_y - fit(size_x, a, b)
                        ss_res = np.sum(residuals ** 2)
                        ss_tot = np.sum((size_y - np.mean(size_y)) ** 2)
                        r_squared = 1 - (ss_res / ss_tot)
                        groups_text.set_text("$R^2$ = %.5f\n $bx^a$: a=%.2f, b=%.2f" % (r_squared, a, b))
                        artists.append(groups_text)
                    except Exception as e:
                        print("Exception in fit: %s" % e)


            if draw_quiver:
                x, y = positions[:, 0], positions[:, 1]
                n = len(frame)
                new_offsets = np.zeros((n, 2))
                for i in range(n):
                    new_offsets[i][0] = x[i]
                    new_offsets[i][1] = y[i]
                u = np.cos(angles)
                v = np.sin(angles)
                quiver = ax_quiver.quiver(x, y, u, v, quiver_colors, cmap=cm, angles='xy', scale_units='xy', scale=1)
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

                correlations_text_value = "draw pts: %d\ncorr pts: %d" % (space_points, correlations_stochastic_points)
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

            if draw_avg_polar:
                avg_polar.set_data(avg_angles, avg_speeds)
                artists.append(avg_polar)
                avg_polar_end.set_data([avg_angles[-1]], [avg_speeds[-1]])
                artists.append(avg_polar_end)

            time_text.set_text('time = %.1f s - params: N=%d, L=%d, eta=%.2f, v=%.1f, omega=%.1f, T=%d, dt=%.1f' %
                               (round(dt * num, 3),
                                len(sky.birds), L, eta, sky.birds[0].vel,
                                sky.birds[0].ang_vel, total_time, dt))
            artists.append(time_text)

            if draw_avg_speed:
                ax_speed.legend(loc=1)
            if draw_avg_angle:
                ax_angle.legend(loc=2)
            if draw_avg_groups:
                ax_groups.legend()
            if draw_correlations_fit:
                ax_correlation_length.legend(loc=2)
            plt.tight_layout()

            return artists

        start_t = time.time()
        print("Drawing start at t=%s" % datetime.datetime.fromtimestamp(start_t).strftime('%Y-%m-%d %H:%M:%S'))

        anim = animation.FuncAnimation(fig, update_animation, init_func=lambda: (), frames=total_frames, interval=200, blit=True, repeat=True)
        anim.save(output_file, writer=self.writer)
        elapsed = time.time() - start_t
        print(
            "Drawing ended at t=%s, elapsed: %dh %dm %ds. N=%d, L=%d, eta=%.2f, v=%.1f, omega=%.1f, T=%d, dt=%.1f" % (
            datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
            elapsed // 3600 % 24, elapsed // 60 % 60, elapsed % 60,
            len(frames[0]), L, eta, frames[0][0][2], frames[0][0][3], total_time, dt))


def launch_simulation(output_file: str, L: float, n_birds: int, vel: float=1, ang_vel: float=np.pi/2, interaction_radius: float=1,
                      eta: float=.5, dt: float=1, total_time: float=100):
    gridstep = interaction_radius / 2
    sky = Sky(L, gridstep)
    sky.add_n_random_birds(n_birds, vel, ang_vel)
    physics = Physics(sky, interaction_radius, eta)
    life = Life()

    life.simulate(physics, dt, total_time, verbose_prop=.1, output_file=output_file)


launch_simulation("test.json", 10, 10)

#Life().animate(input_file="test.json", verbose_prop=.001, to_draw=["quiver", "avg_speed", "avg_angle", "avg_polar", "groups", "avg_groups"], options={"fit_spatial_points": 100,
#                                                                                                                     "max_num_groups": 100/5})






















































