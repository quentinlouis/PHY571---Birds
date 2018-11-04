import numpy as np


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
