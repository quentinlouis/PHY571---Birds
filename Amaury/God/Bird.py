import numpy as np


class Bird:
    vel: float
    angle: float
    ang_vel: float
    pos: np.ndarray
    lookV: np.ndarray
    speedV: np.ndarray

    def __init__(self, pos: np.ndarray, vel: float, ang_vel: float, angle: float):
        self.vel = vel
        self.angle = angle
        self.ang_vel = ang_vel
        self.pos = pos
        self.lookV = None
        self.speedV = None
        self.update_calculated_props()

    def clone(self):
        return Bird(self.pos, self.vel, self.ang_vel, self.angle)

    def __repr__(self) -> str:
        return "Bird(%.2f, %.2f)" % (self.pos[0], self.pos[1])

    def update_calculated_props(self) -> None:
        self.lookV = np.array([np.cos(self.angle), np.sin(self.angle)])
        self.speedV = self.lookV * self.vel
