import numpy as np

import Amaury.God.Life as Life
from Amaury.God.DataProcessing import Processor
from Amaury.God.DataVisualisation import Visualiser
from Amaury.God.Physics import Physics
from Amaury.God.Sky import Sky


def launch_simulation(output_file: str, L: float, n_birds: int, vel: float = 1, ang_vel: float = np.pi / 2,
                      interaction_radius: float = 1,
                      eta: float = .5, dt: float = 1, total_time: float = 100):
    gridstep = interaction_radius / 2
    sky = Sky(L, gridstep)
    sky.add_n_random_birds(n_birds, vel, ang_vel)
    physics = Physics(sky, interaction_radius, eta)
    Life.simulate(physics, dt, total_time, verbose_prop=.1, output_file=output_file)


launch_simulation("simulation_data/test.json", L=100, n_birds=1000)

to_process = ["avg_speed", "avg_angle", "group_size", "group_size_avg", "group_size_avg_fit", "groups", "correlations",
              "correlations_fit"]
Processor().process("simulation_data/test.json", "processing_data/test", verbose_prop=.1, to_process=to_process)

to_draw = ["avg_speed", "avg_angle", "avg_polar", "correlations", "correlations_fit", "correlation_length",
           "group_size", "group_size_avg", "group_size_avg_fit", "quiver"]
Visualiser("processing_data/test", "visualisations/test.mp4", simulation_data_file="simulation_data/test.json",
           verbose_prop=.1,
           to_draw=to_draw, options={"quiver_color_by_group": True}).vizualize()
