import numpy as np
from typing import Callable

import God.Life as Life
from God.DataProcessing import Processor
from God.DataVisualisation import Visualiser
from God.Physics import Physics
from God.Sky import Sky


def launch_simulation(output_file: str, L: float, n_birds: int, vel: float = 1, ang_vel: float = np.pi / 2,
                      interaction_radius: float = 1,
                      eta: float = .5, dt: float = 1, total_time: float = 100, evolve: Callable=None) -> None:
    gridstep = interaction_radius / 2
    sky = Sky(L, gridstep)
    sky.add_n_random_birds(n_birds, vel, ang_vel)
    physics = Physics(sky, interaction_radius, eta)
    Life.simulate(physics, dt, total_time, verbose_prop=.1, output_file=output_file)#, evolve=evolve)


def evolve(sky: Sky):
    sky.add_n_random_birds(3, 1, np.pi/2)

#Original simulation code
"""
launch_simulation("simulation_data/test.json", L=100, n_birds=1000, total_time=100)#, evolve=evolve)

to_process = ["avg_speed", "avg_angle", "group_size", "group_size_avg", "group_size_avg_fit", "groups", "correlations",
              "correlations_fit"]
Processor().process("simulation_data/test.json", "processing_data/test", verbose_prop=.1, to_process=to_process)

to_draw = ["avg_speed", "avg_angle", "avg_polar", "correlations", "correlations_fit", "correlation_length",
           "group_size", "group_size_avg", "group_size_avg_fit", "quiver"]
Visualiser("processing_data/test", "visualisations/test.mp4", simulation_data_file="simulation_data/test.json",
           verbose_prop=.1,
           to_draw=to_draw, options={"quiver_color_by_group": True, "max_group_size": 40, "max_num_groups": 40}).vizualize()"""

"""N = [300, 3000, 30000]
L = [10, 100, 1000]
Eta = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
angle_vel = 2*np.pi
T = 200

for i in range(len(N)):
    for j in range(len(L)):
        for k in range(len(Eta)):
            name = "N_" + str(N[i]) + "_L_" + str(L[j]) + "_Eta_" + str(Eta[k]) + "_angle_vel_" + "pi" + "_T_" + str(T)
            launch_simulation("simulation_data/04_11_18/"+ name +".json", L=L[j], n_birds=N[i], ang_vel = angle_vel, eta = Eta[k], total_time=T)#, evolve=evolve)

            to_process = ["avg_speed", "avg_angle", "group_size", "group_size_avg", "group_size_avg_fit", "correlations", "correlations_fit"]
            
            Processor().process("simulation_data/04_11_18/"+ name +".json", "processing_data/04_11_18/"+ name, verbose_prop=.1, to_process=to_process)

            to_draw = ["avg_speed", "avg_angle", "avg_polar", "correlations", "correlations_fit", "correlation_length",
                       "group_size", "group_size_avg", "group_size_avg_fit", "quiver"]
            Visualiser("processing_data/04_11_18/"+ name, "visualisations/04_11_18/"+ name +".mp4", simulation_data_file="simulation_data/04_11_18/"+ name +".json",
                       verbose_prop=.1,
                       to_draw=to_draw, options={"quiver_color_by_group": False, "max_group_size": 60, "max_num_groups": 60}).vizualize()"""

            
            
#Code for the simulation with 100000 points
to_draw = ["avg_speed", "avg_angle", "avg_polar", "group_size", "group_size_avg", "group_size_avg_fit", "quiver"]
Visualiser("processing_data/test", "visualisations/test.mp4", simulation_data_file="simulation_data/test.json",
                       verbose_prop=.1,
                       to_draw=to_draw, options={"quiver_color_by_group": False, "max_group_size": 10000, "max_num_groups": 100000}).vizualize()