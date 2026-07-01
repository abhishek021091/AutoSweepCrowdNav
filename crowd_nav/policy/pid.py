import numpy as np

from crowd_sim.envs.utils.action import ActionXY


class PIDPolicy:

    def __init__(self, kp, ki, kd):

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.prev_error = 0.0

    def reset(self):

        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):

        self.integral += error * dt

        derivative = (error - self.prev_error) / dt

        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        self.prev_error = error

        return output


class PID:

    def __init__(self, config):

        self.name = "PID"

        self.trainable = False
        self.multiagent_training = None

        self.pid_x = PIDPolicy(config.robot.kp, config.robot.ki, config.robot.kd)
        self.pid_y = PIDPolicy(config.robot.kp, config.robot.ki, config.robot.kd)

        # Change if your simulator uses another timestep
        self.dt = config.env.time_step

    def reset(self):

        self.pid_x.reset()
        self.pid_y.reset()

    def predict(self, state):

        robot = state.self_state

        ex = robot.gx - robot.px
        ey = robot.gy - robot.py

        vx = self.pid_x.update(ex, self.dt)
        vy = self.pid_y.update(ey, self.dt)

        speed = np.hypot(vx, vy)

        if speed > robot.v_pref:

            scale = robot.v_pref / speed

            vx *= scale
            vy *= scale

        return ActionXY(vx, vy)