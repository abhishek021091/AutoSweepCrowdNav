import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, dt,
                 output_min=-1.0, output_max=1.0):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.dt = dt

        self.integral = 0.0
        self.prev_error = 0.0

        self.output_min = output_min
        self.output_max = output_max

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, target, current):

        error = target - current

        self.integral += error * self.dt

        derivative = (error - self.prev_error) / self.dt

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        self.prev_error = error

        output = np.clip(output,
                         self.output_min,
                         self.output_max)

        return output