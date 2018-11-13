import datetime
import logging
import time
from typing import Any, Tuple, List, Union

import cmocean
import matplotlib.animation as animation
import matplotlib.colors
import numpy as np
import scipy.spatial
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

import God.SaveAndLoad as SaveAndLoad

log = logging.getLogger('DataVisualisation')
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


def get_layout(option: str = "default") -> Tuple[Any, dict, dict]:
    subplots = {}
    fig = plt.figure()
    layout_artists = {}

    if option == "default_old":
        gs = GridSpec(2, 4)
        subplots["quiver"] = fig.add_subplot(gs[0:2, 0:2])
        subplots["speed"] = fig.add_subplot(gs[0, 2])
        subplots["angle"] = subplots["speed"].twinx()
        subplots["polar"] = fig.add_subplot(gs[0, 3], projection='polar')
        subplots["correlations"] = fig.add_subplot(gs[1, 2])
        subplots["correlation_length"] = subplots["correlations"].twinx()
        subplots["groups"] = fig.add_subplot(gs[1, 3])
        fig.set_size_inches(18, 10, True)

        # text and time
        layout_artists["time_text"] = subplots["quiver"].text(0.02, 1.05, '', transform=subplots["quiver"].transAxes)
    elif option == "default":
        gs = GridSpec(3, 6)
        subplots["quiver"] = fig.add_subplot(gs[0:3, 0:3])
        subplots["speed"] = fig.add_subplot(gs[0, 3])
        subplots["angle"] = subplots["speed"].twinx()
        subplots["polar"] = fig.add_subplot(gs[0, 4], projection='polar')
        subplots["correlations"] = fig.add_subplot(gs[1, 3])
        subplots["correlation_length"] = subplots["correlations"].twinx()
        subplots["groups"] = fig.add_subplot(gs[2, 3])
        subplots["evolution_group_size"] = fig.add_subplot(gs[1, 4])

        fig.set_size_inches(18, 10, True)

        # text and time
        layout_artists["time_text"] = subplots["quiver"].text(0.02, 1.05, '', transform=subplots["quiver"].transAxes)
    else:
        raise Exception("Wrong layout provided")

    return fig, subplots, layout_artists


drawable_to_data = {
    "quiver": ["_simulation"],
    "avg_speed": ["avg_speed"],
    "avg_angle": ["avg_angle"],
    "avg_polar": ["avg_speed", "avg_angle"],
    "correlations": ["correlations"],
    "correlations_fit": ["correlations_fit"],
    "correlation_length": ["correlations_fit"],
    "group_size": ["group_size"],
    "group_size_avg": ["group_size_avg"],
    "group_size_avg_fit": ["group_size_avg_fit"],
    "evolution_group_size": ["group_size", "groups"],
}


class Visualiser:
    processing_dir: str
    output_file: str
    to_draw: dict
    simulation_data_file: str
    verbose_prop: float
    layout: str
    options: dict
    t_start: float
    t_end: float

    processed_data: dict
    to_unblit: list
    fig: Any
    subplots: dict
    layout_artists: dict
    timestamps: np.ndarray
    simulation_parameters: Union[list, dict]
    norm: Any
    cm: Any
    grou_colors: np.ndarray
    start_frame: int

    def __init__(self, processing_dir: str, output_file: str, to_draw: List[str], simulation_data_file: str = "",
                 verbose_prop: float = .1, layout: str = "default", options: dict = None, t_start: float = None,
                 t_end: float = None):
        self.processing_dir = processing_dir
        self.output_file = output_file
        self.to_draw = dict()
        self.chose_what_to_draw(to_draw)
        self.simulation_data_file = simulation_data_file
        self.verbose_prop = verbose_prop
        self.layout = layout
        self.options = dict() if options is None else options
        if "max_group_size" not in self.options:
            self.options["max_group_size"] = 20
        if "max_num_groups" not in self.options:
            self.options["max_num_groups"] = 20
        if "animation" not in self.options:
            self.options["animation"] = True
        if "quiver_color_by_group" not in self.options:
            self.options["quiver_color_by_group"] = False
        if "quiver_color_single" not in self.options:
            self.options["quiver_color_single"] = False
        if "quiver_draw_by_group" not in self.options:
            self.options["quiver_draw_by_group"] = False
        if "groups_num_ymin" not in self.options:
            self.options["groups_num_ymin"] = .01
        if "vel_scaling" not in self.options:
            self.options["vel_scaling"] = 1

        self.group_colors = np.random.rand(100)

        self.processed_data = dict()
        self.to_unblit = []
        self.fig = None
        self.subplots = dict()
        self.layout_artists = dict()
        self.timestamps = None
        self.writer = animation.writers['ffmpeg'](fps=15, bitrate=-1)
        self.norm = None
        self.cm = None
        self.t_start = t_start
        self.t_end = t_end
        self.start_frame = 0

    def chose_what_to_draw(self, to_draw: List[str]) -> None:
        self.to_draw["quiver"] = "quiver" in to_draw
        self.to_draw["avg_speed"] = "avg_speed" in to_draw
        self.to_draw["avg_angle"] = "avg_angle" in to_draw
        self.to_draw["avg_polar"] = "avg_polar" in to_draw
        self.to_draw["correlations"] = "correlations" in to_draw
        self.to_draw["correlations_fit"] = "correlations_fit" in to_draw
        self.to_draw["correlation_length"] = "correlation_length" in to_draw
        self.to_draw["group_size"] = "group_size" in to_draw
        self.to_draw["group_size_avg"] = "group_size_avg" in to_draw
        self.to_draw["group_size_avg_fit"] = "group_size_avg_fit" in to_draw
        self.to_draw["evolution_group_size"] = "evolution_group_size" in to_draw

        keys_to_draw = [k for k, v in self.to_draw.items() if v]
        log.info("The following properties will be drawn: %s" % keys_to_draw)

    def load_data(self, simulation_data_file: str, processing_dir: str) -> None:
        # decide which data is necessary
        data_to_fetch = set()
        for drawable in self.to_draw:
            required_data = drawable_to_data[drawable]
            for data in required_data:
                data_to_fetch.add(data)
        if self.options["quiver_color_by_group"] or self.options["quiver_draw_by_group"]:
            data_to_fetch.add("groups")

        # load data
        self.processed_data = {}
        if "_simulation" in data_to_fetch:
            self.processed_data["_simulation"] = SaveAndLoad.load_data(simulation_data_file)
            data_to_fetch.remove("_simulation")
        for property_name in data_to_fetch:
            self.processed_data[property_name] = SaveAndLoad.load_data_dirname(processing_dir,
                                                                               "%s.json" % property_name)
        self.simulation_parameters = SaveAndLoad.load_data_dirname(processing_dir, "processing_parameters.json")

    def init_layout(self) -> None:
        total_time = self.timestamps[-1]
        L = self.simulation_parameters["L"]

        if self.to_draw["quiver"]:
            ax_quiver = self.subplots["quiver"]
            ax_quiver.set_xlim(0, L)
            ax_quiver.set_ylim(0, L)

        if self.to_draw["group_size"] or self.to_draw["group_size_avg"]:
            ax_groups = self.subplots["groups"]
            max_group_size = self.options["max_group_size"]
            max_num_groups = self.options["max_num_groups"]
            groups_num_ymin = self.options["groups_num_ymin"]
            ax_groups.set_xlim(1, max_group_size)
            ax_groups.set_ylim(groups_num_ymin, max_num_groups)
            ax_groups.set_xscale("log")
            ax_groups.set_yscale("log")
            self.layout_artists["groups_text"] = self.subplots["groups"].text(
                0.02, 1.05, '', transform=self.subplots["groups"].transAxes)
            if self.to_draw["group_size"]:
                self.layout_artists["group_size"], = ax_groups.plot([], [], linestyle="None", marker="o", lw=2)
                ax_groups.set_xlabel("group size")
                ax_groups.set_ylabel("# of groups")
            if self.to_draw["group_size_avg"]:
                self.layout_artists["group_size_avg"], = ax_groups.plot([], [], marker="o", lw=2, label="average")
                self.layout_artists["group_size_avg_fit"], = ax_groups.plot([], [], lw=2, label="fit")

        if self.to_draw["avg_speed"]:
            vel_scaling = self.options["vel_scaling"]
            ax_speed = self.subplots["speed"]
            ax_speed.set_xlim(self.t_start, self.t_end)
            ax_speed.set_ylim(0, vel_scaling)
            self.layout_artists["avg_speed"], = ax_speed.plot([], [], lw=2, label="speed")
            ax_speed.set_xlabel("time (s)")
            ax_speed.set_ylabel("avg. speed (m/s)")

        if self.to_draw["avg_angle"]:
            ax_angle = self.subplots["angle"]
            ax_angle.set_ylim(-np.pi, np.pi)
            self.layout_artists["avg_angle"], = ax_angle.plot([], [], lw=2, color="orange", label="angle")
            ax_angle.set_ylabel("avg. angle (rad)")

        if self.to_draw["correlations"]:
            ax_correlation = self.subplots["correlations"]
            ax_correlation_length = self.subplots["correlation_length"]
            ax_correlation.set_xlim(0, L)
            ax_correlation.set_ylim(0, 1)
            ax_correlation_length.set_ylim(0, L / 3)
            self.layout_artists["correlations"], = ax_correlation.plot([], [], lw=2)
            self.layout_artists["correlations_text"] = self.subplots["correlations"].text(0.02, 1.05, '',
                                                                                          transform=self.subplots[
                                                                                              "correlations"].transAxes)
            if self.to_draw["correlations_fit"]:
                self.layout_artists["correlations_fit"], = ax_correlation.plot([], [], lw=2)
            if self.to_draw["correlation_length"]:
                self.layout_artists["correlation_length"], = ax_correlation_length.plot([], [], lw=2,
                                                                                        label="correlation length",
                                                                                        color="green")
                ax_correlation_length.set_ylabel("correlation length (m)")
            ax_correlation.set_xlabel("distance (m)")
            ax_correlation.set_ylabel("correlation (normed)")

        if self.to_draw["avg_polar"]:
            vel_scaling = self.options["vel_scaling"]
            ax_polar = self.subplots["polar"]
            ax_polar.set_title("Avg. angle and speed")
            self.layout_artists["avg_polar"], = ax_polar.plot([], [], lw=2)
            self.layout_artists["avg_polar_endpoint"], = ax_polar.plot([], [], lw=2, marker="o", markersize=6,
                                                                       color="red")
            ax_polar.set_rmax(vel_scaling)

        if self.to_draw["evolution_group_size"]:
            ax_speed = self.subplots["evolution_group_size"]
            ax_speed.set_xlim(self.t_start, self.t_end)
            max_group_size = self.options["max_group_size"]
            ax_speed.set_ylim(0, max_group_size)
            self.layout_artists["evolution_group_size"], = ax_speed.plot([], [], lw=2, label="evolution_group_size")
            ax_speed.set_xlabel("time (s)")
            ax_speed.set_ylabel("group size")

    def initialise_visualisation(self) -> None:
        # load data
        self.load_data(self.simulation_data_file, self.processing_dir)
        total_time = self.simulation_parameters["total_time"]
        dt = self.simulation_parameters["dt"]
        L = self.simulation_parameters["L"]

        # get layout
        self.fig, self.subplots, self.layout_artists = get_layout(self.layout)

        # time variables - rescale time, remove unnecessary data
        if self.t_start is None:
            self.t_start = 0
        if self.t_end is None:
            self.t_end = total_time
        self.t_start = max(0, self.t_start)
        self.t_end = min(total_time, self.t_end)
        self.timestamps = np.arange(self.t_start, self.t_end, dt)
        self.start_frame = int(self.t_start / (self.t_end - self.t_start) * len(self.timestamps))
        for key in self.processed_data:
            if key[0] != "_":
                self.processed_data[key] = self.processed_data[key][self.start_frame:]
            else:
                self.processed_data[key]["frames"] = self.processed_data[key]["frames"][self.start_frame:]

        self.init_layout()

    def plot_avg_speed(self, frame_num: int) -> None:
        times = self.timestamps[:frame_num + 1]
        speeds = self.processed_data["avg_speed"][:frame_num + 1]
        self.layout_artists["avg_speed"].set_data(times, speeds)
        self.subplots["speed"].legend(loc=1)

    def plot_avg_angle(self, frame_num: int) -> None:
        times = self.timestamps[:frame_num + 1]
        angles = self.processed_data["avg_angle"][:frame_num + 1]
        self.layout_artists["avg_angle"].set_data(times, angles)
        self.subplots["angle"].legend(loc=2)

    def plot_avg_polar(self, frame_num: int) -> None:
        speeds = self.processed_data["avg_speed"][:frame_num + 1]
        angles = self.processed_data["avg_angle"][:frame_num + 1]
        self.layout_artists["avg_polar"].set_data(angles, speeds)
        self.layout_artists["avg_polar_endpoint"].set_data([angles[-1]], [speeds[-1]])

    def plot_correlations(self, frame_num: int) -> None:
        spacial_points = self.simulation_parameters["processing_options"]["fit_spatial_points"]
        L = self.simulation_parameters["L"]

        spatial_x, step = np.linspace(0, L, spacial_points, endpoint=False, retstep=True)
        spatial_x = spatial_x[:-1] + step / 2
        correlations = self.processed_data["correlations"][frame_num]
        self.layout_artists["correlations"].set_data(spatial_x, correlations)

    def plot_correlations_fit(self, frame_num: int) -> None:
        def func_fit(x, a1, zeta1):
            return a1 * np.exp(- x / zeta1)

        spacial_points = self.simulation_parameters["processing_options"]["fit_spatial_points"]
        correlation_stochastic_points = self.simulation_parameters["processing_options"][
            "correlations_stochastic_points"]
        L = self.simulation_parameters["L"]

        spatial_x, step = np.linspace(0, L, spacial_points + 1, retstep=True)
        spatial_x -= step / 2
        data = self.processed_data["correlations_fit"][frame_num]
        if data is None:
            return
        a, zeta, r_squared = data
        self.layout_artists["correlations_fit"].set_data(spatial_x, func_fit(spatial_x, a, zeta))
        self.layout_artists["correlations_text"].set_text(
            "draw pts: %d\ncorr pts: %d\n$R^2$ = %.5f" % (spacial_points, correlation_stochastic_points, r_squared))

    def plot_correlations_length(self, frame_num: int) -> None:
        L = self.simulation_parameters["L"]

        times = self.timestamps[:frame_num + 1]
        zetas = []
        for data in self.processed_data["correlations_fit"][:frame_num + 1]:
            if data is not None:
                zetas.append(data[1])
            else:
                zetas.append(zetas[-1] if len(zetas) > 0 else 0)
        self.layout_artists["correlation_length"].set_data(times * L / self.timestamps[-1], zetas)
        self.subplots["correlation_length"].legend(loc=2)

    def plot_group_size(self, frame_num: int) -> None:
        correlations = self.processed_data["group_size"][frame_num]
        max_group_size = self.options["max_group_size"]
        size_diff = max_group_size - len(correlations)
        if size_diff > 0:
            correlations += [0] * size_diff
        self.layout_artists["group_size"].set_data(range(1, max_group_size + 1), correlations[:max_group_size])

    def plot_group_size_avg(self, frame_num: int) -> None:
        correlations = self.processed_data["group_size_avg"][frame_num]
        max_group_size = self.options["max_group_size"]
        size_diff = max_group_size - len(correlations)
        if size_diff > 0:
            correlations += [0] * size_diff
        self.layout_artists["group_size_avg"].set_data(range(1, max_group_size + 1), correlations[:max_group_size])

    def plot_group_size_avg_fit(self, frame_num: int) -> None:
        def fit(x, a1, b1):
            return np.exp(b1) * x**a1

        data = self.processed_data["group_size_avg_fit"][frame_num]
        if data is None:
            return
        a, b, r_squared = data
        max_group_size = self.options["max_group_size"]
        x_data = np.array(range(1, max_group_size + 1))
        self.layout_artists["group_size_avg_fit"].set_data(x_data, fit(x_data, a, b))
        self.layout_artists["groups_text"].set_text("$R^2$ = %.5f\n $ax+b$: a=%.2f, b=%.2f" % (r_squared, a, b))

    def plot_evolution_group_size(self, frame_num: int) -> None:
        times = self.timestamps[:frame_num + 1]
        ydata = [self.processed_data["group_size"][frame][(self.processed_data["groups"][frame][0])] for frame in range(frame_num)]
        self.layout_artists["evolution_group_size"].set_data(times, ydata )

    def plot_quiver(self, frame_num: int) -> None:
        frame = self.processed_data["_simulation"]["frames"][frame_num]
        positions = np.array([np.array(bird[0]) for bird in frame])
        angles = np.array([np.array(bird[1]) for bird in frame])
        if self.options["quiver_color_by_group"]:
            groups = self.processed_data["groups"][frame_num]
            quiver_colors = []
            for bird_group in groups:
                quiver_colors.append(self.group_colors[bird_group % len(self.group_colors)])
        elif self.options["quiver_color_single"]:
            quiver_colors = np.zeros(positions.shape[0]) + 1  # grey
            quiver_colors[0] = 0  # Red
            self.cm = plt.get_cmap("Set1")
        else:
            quiver_colors = self.norm(angles)

        if self.options["quiver_draw_by_group"]:
            L = self.simulation_parameters["L"]
            groups = self.processed_data["groups"][frame_num]
            indexes_of_bird_by_group = [[i for i in range(len(groups)) if groups[i]==j] for j in range(len(groups))]
            for group in range(len(groups)):
                birds_indexes = indexes_of_bird_by_group[group]
                if len(birds_indexes) == 0:
                    continue
                elif len(birds_indexes) == 1:
                    pos = positions[birds_indexes[0]]
                    hull_plot_points, = self.subplots["quiver"].plot([pos[0]], [pos[1]], 'ro', markersize=3, color="blue")
                    self.to_unblit.append(hull_plot_points)
                    self.layout_artists["quiver_hull_%s_points" % group] = hull_plot_points
                    continue

                positions_hull = positions[birds_indexes]
                angles_hull = angles[birds_indexes]
                if len(birds_indexes) == 2:
                    hull = list(range(len(birds_indexes)))
                else:
                    hull = scipy.spatial.ConvexHull(positions_hull).vertices
                x = list(positions_hull[hull, 0])+[positions_hull[hull[0], 0]]
                y = list(positions_hull[hull, 1])+[positions_hull[hull[0], 1]]
                angles_hull = list(angles_hull[hull]) + [angles_hull[hull[0]]]
                for point in range(len(list(x))-1):
                    distance_x = abs(x[point + 1] - x[point])
                    distance_y = abs(y[point + 1] - y[point])
                    if self.options["quiver_color_by_group"]:
                        color = self.cm(self.group_colors[group % len(self.group_colors)])
                    else:
                        color = self.cm(self.norm((angles_hull[point]+angles_hull[point+1])/2))
                    if distance_x < L/2 and distance_y < L/2:
                        hull_plot_segment, = self.subplots["quiver"].plot([x[point], x[point+1]], [y[point], y[point+1]], linewidth=2, color=color)
                        self.to_unblit.append(hull_plot_segment)
                        self.layout_artists["quiver_hull_%s_%s" % (group, point)] = hull_plot_segment
                        # hull_plot_points, = self.subplots["quiver"].plot([x[point], x[point+1]], [y[point], y[point+1]], 'ro', markersize=3, color="blue")
                        # self.to_unblit.append(hull_plot_points)
                        # self.layout_artists["quiver_hull_%s_%s_points" % (group, point)] = hull_plot_points
        else:
            x, y = positions[:, 0], positions[:, 1]
            n = len(frame)
            new_offsets = np.zeros((n, 2))
            for i in range(n):
                new_offsets[i][0] = x[i]
                new_offsets[i][1] = y[i]
            u = np.cos(angles)
            v = np.sin(angles)
            quiver = self.subplots["quiver"].quiver(x, y, u, v, color=self.cm(quiver_colors), angles='xy', scale_units='xy',
                                                    scale=1)
            self.layout_artists["quiver"] = quiver
            self.to_unblit.append(quiver)
        self.layout_artists["time_text"].set_text("t=%.1f, N=%d" % (self.timestamps[frame_num], len(positions)))

    def vizualize(self) -> None:
        self.initialise_visualisation()

        animation_mode = self.options["animation"]
        if animation_mode:
            self.draw_animation()

    def update_animation(self, frame_number: int, start_t: float) -> list:
        total_frames = len(self.timestamps)
        if frame_number % (1 + int(total_frames * self.verbose_prop)) == 0:
            time_per_frame = (time.time() - start_t) / (frame_number + 1)
            remaining_time = time_per_frame * (total_frames - frame_number)
            log.info("Drawing frame %d/%d - remaining est. %dh %dm %ds" % (frame_number, total_frames,
                                                                              remaining_time // 3600 % 24,
                                                                              remaining_time // 60 % 60,
                                                                              remaining_time % 60,))

        # unblit artists created on the spot
        for artist in list(self.to_unblit):
            self.to_unblit.remove(artist)
            artist.remove()
            del artist

        if self.to_draw["avg_speed"]:
            self.plot_avg_speed(frame_number)

        if self.to_draw["avg_angle"]:
            self.plot_avg_angle(frame_number)

        if self.to_draw["avg_polar"]:
            self.plot_avg_polar(frame_number)

        if self.to_draw["correlations"]:
            self.plot_correlations(frame_number)

        if self.to_draw["correlations_fit"]:
            self.plot_correlations_fit(frame_number)

        if self.to_draw["correlation_length"]:
            self.plot_correlations_length(frame_number)

        if self.to_draw["group_size"]:
            self.plot_group_size(frame_number)

        if self.to_draw["group_size_avg"]:
            self.plot_group_size_avg(frame_number)

        if self.to_draw["group_size_avg_fit"]:
            self.plot_group_size_avg_fit(frame_number)

        if self.to_draw["quiver"]:
            self.plot_quiver(frame_number)

        if self.to_draw["evolution_group_size"]:
            self.plot_evolution_group_size(frame_number)

        plt.tight_layout()
        return list(self.layout_artists.values())

    def draw_animation(self) -> None:
        # colors
        self.norm = matplotlib.colors.Normalize(vmin=0, vmax=2 * np.pi)
        self.cm = cmocean.cm.phase

        # time
        total_frames = len(self.timestamps)

        start_t = time.time()
        log.info("Drawing start at t=%s" % datetime.datetime.fromtimestamp(start_t).strftime('%Y-%m-%d %H:%M:%S'))

        anim = animation.FuncAnimation(self.fig, self.update_animation, init_func=lambda: (), frames=total_frames,
                                       interval=200, blit=True, repeat=True, fargs=(start_t, ))
        SaveAndLoad.make_path_available(self.output_file)
        anim.save(self.output_file, writer=self.writer)
        elapsed = time.time() - start_t

        log.info(
            "Drawing ended at t=%s, elapsed: %dh %dm %ds" % (
                datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                elapsed // 3600 % 24, elapsed // 60 % 60, elapsed % 60))
        log.info("Visualisation results saved")
