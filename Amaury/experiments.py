from typing import Callable

import numpy as np

import God.Life as Life
from God.Bird import Bird
from God.DataProcessing import Processor
from God.DataVisualisation import Visualiser
from God.Physics import Physics
from God.Sky import Sky


def launch_simulation_random(output_file: str, L: float, n_birds: int, vel: float = 1, ang_vel: float = np.pi / 2,
                             interaction_radius: float = 1,
                             eta: float = .5, dt: float = 1, total_time: float = 100, evolve: Callable = None) -> None:
    gridstep = interaction_radius
    sky = Sky(L, gridstep)
    sky.add_n_random_birds(n_birds, vel, ang_vel)
    physics = Physics(sky, interaction_radius, eta)
    Life.simulate(physics, dt, total_time, verbose_prop=.1, output_file=output_file, evolve=evolve)


def launch_two_groups(output_file: str, L: float, n_birds_1: int, n_birds_2: int, radius_1: int, radius_2: int,
                      center_1: list, center_2: list, angle_1: float, angle_2: float, vel: float = 1,
                      ang_vel: float = np.pi / 2,
                      interaction_radius: float = 1,
                      eta: float = .1, dt: float = 1, total_time: float = 100) -> None:
    gridstep = interaction_radius
    sky = Sky(L, gridstep)
    group_1_positions = (np.random.rand(n_birds_1, 2) - .5) * radius_1 + np.array(center_1)
    for i in range(n_birds_1):
        sky.add_bird(Bird(group_1_positions[i], vel, ang_vel, angle_1))
    group_2_positions = (np.random.rand(n_birds_2, 2) - .5) * radius_2 + np.array(center_2)
    for i in range(n_birds_2):
        sky.add_bird(Bird(group_2_positions[i], vel, ang_vel, angle_2))

    physics = Physics(sky, interaction_radius, eta)
    Life.simulate(physics, dt, total_time, verbose_prop=.1, output_file=output_file)


launch_two_groups("simulation_data/test.json", L=100, n_birds_1=200, n_birds_2=100, radius_1=5, radius_2=5,
                  total_time=100, center_1=[20, 50], center_2=[80, 50], angle_1=0, angle_2=np.pi)

to_process = ["avg_speed", "avg_angle", "group_size", "group_size_avg", "group_size_avg_fit", "groups", "correlations",
              "correlations_fit"]
Processor().process("simulation_data/test.json", "processing_data/test", verbose_prop=.1, to_process=to_process,
                    options={"correlations_stochastic_points": 5000})

to_draw = ["avg_speed", "avg_angle", "avg_polar", "correlations", "correlations_fit", "correlation_length",
           "group_size", "group_size_avg", "group_size_avg_fit", "quiver"]

Visualiser("processing_data/test", "visualisations/test.mp4", simulation_data_file="simulation_data/test.json",
           verbose_prop=.1,
           to_draw=to_draw,
           options={"quiver_color_by_group": False, "quiver_draw_by_group": False, "max_group_size": 5000,
                    "max_num_groups": 500}).vizualize()

# N = [10000]
#
# L = [1000]
#
# Eta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
#
# angle_vel = 2 * np.pi
#
# T = 200
#
# for i in range(len(N)):
#
#     for j in range(len(L)):
#
#         for k in range(len(Eta)):
#             name = "N_" + str(N[i]) + "_L_" + str(L[j]) + "_Eta_" + str(Eta[k]) + "_angle_vel_" + "pi" + "_T_" + str(T)
#
#             launch_simulation("simulation_data/04_11_18/" + name + ".json", L=L[j], n_birds=N[i], ang_vel=angle_vel,
#                               eta=Eta[k], total_time=T)  # , evolve=evolve)
#
#             to_process = ["avg_speed", "avg_angle", "group_size", "group_size_avg", "group_size_avg_fit",
#                           "correlations", "correlations_fit"]
#
#             Processor().process("simulation_data/04_11_18/" + name + ".json", "processing_data/04_11_18/" + name,
#                                 verbose_prop=.1, to_process=to_process)
#
#             to_draw = ["avg_speed", "avg_angle", "avg_polar", "correlations", "correlations_fit", "correlation_length",
#
#                        "group_size", "group_size_avg", "group_size_avg_fit", "quiver"]
#
#             Visualiser("processing_data/04_11_18/" + name, "visualisations/04_11_18/" + name + ".mp4",
#                        simulation_data_file="simulation_data/04_11_18/" + name + ".json",
#
#                        verbose_prop=.1,
#
#                        to_draw=to_draw,
#                        options={"quiver_color_by_group": False, "max_group_size": 60, "max_num_groups": 60}).vizualize()
