import datetime
import logging
import time

import numpy as np

import Amaury.God.SaveAndLoad as SaveAndLoad
from Amaury.God.Physics import Physics

log = logging.getLogger('Life')
log.addHandler(logging.StreamHandler())
log.setLevel(logging.INFO)


def simulate(physics: Physics, dt: float, total_time: float, verbose_prop: float = .01,
             output_file: str = "simulation_data/data.json",
             evolve=lambda sky: None):
    timestamps = np.arange(0, total_time, dt)
    total_frames = len(timestamps)
    frames = []

    start_t = time.time()
    log.info(
        "Simulation start at t=%s. Parameters: total_time=%.1f, dt=%.2f, L=%.1f, N=%d, vel=%.1f, omega=%.1f, r=%.1f, eta=%.2f" %
        (datetime.datetime.fromtimestamp(start_t).strftime('%Y-%m-%d %H:%M:%S'),
         total_time, dt, physics.sky.L, len(physics.sky.birds), physics.sky.birds[0].vel, physics.sky.birds[0].ang_vel,
         physics.interaction_radius, physics.eta))
    frame_n = 0
    for _ in timestamps:
        frame_n += 1
        if frame_n % (1 + int(total_frames * verbose_prop)) == 0:
            log.info("Simulating frame %d/%d" % (frame_n, total_frames))

        if evolve is not None:
            evolve(physics.sky)

        physics.advance(dt)

        frames.append([[bird.pos, bird.angle, bird.vel, bird.ang_vel] for bird in physics.sky.birds])

    elapsed = time.time() - start_t
    log.info(
        "Simulation ended at t=%s, elapsed: %dh %dm %ds. N=%d, L=%d, eta=%.2f, v=%.1f, omega=%.1f, T=%d, dt=%.1f" % (
            datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
            elapsed // 3600 % 24, elapsed // 60 % 60, elapsed % 60,
            len(physics.sky.birds), physics.sky.L, physics.eta, physics.sky.birds[0].vel, physics.sky.birds[0].ang_vel,
            total_time,
            dt))

    data_to_save = {"frames": frames,
                    "parameters": {"dt": dt, "total_time": total_time, "L": physics.sky.L, "eta": physics.eta,
                                   "interaction_radius": physics.interaction_radius}}
    SaveAndLoad.save_data(data_to_save, output_file)
    log.info("Simulation results saved")
