import numpy as np
import itertools
from typing import Tuple, List

from God.Pandora import short_angle_dist
from God.Bird import Bird
from God.Sky import Sky


class Physics:
    def __init__(self, sky: Sky, interaction_radius: float, eta: float):
        self.sky = sky
        sky.update_grid()
        self.interaction_radius = interaction_radius
        self.eta = eta

    def get_groups(self) -> Tuple[List[list], dict]:
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

        def connect(bird_, group_):
            if bird_ in bird_to_group and bird_to_group[bird_] == group_:
                return
            bird_to_group[bird_] = group_
            groups_to_bird[group_].append(bird_)

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
            other_birds = self.sky.grid[i % self.sky.gridL, j % self.sky.gridL]
            if other_birds is not None:
                interact_with += other_birds

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

            fluctuation_angle = bird.ang_vel * self.eta * (np.random.rand() - .5) * dt
            bird.angle = (bird.angle + np.sign(diff) * bird.ang_vel * t_move + fluctuation_angle) % (2 * np.pi)

        # Verlet movement *after* updating directions
        for bird in self.sky.birds:
            bird.update_calculated_props()
            bird.pos = (bird.pos + bird.speedV * dt) % self.sky.L
