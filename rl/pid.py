class PID:
    def __init__(self, kp=1.2, ki=0.0, kd=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0
        self.prev_error = 0

    def reset(self):
        self.integral = 0
        self.prev_error = 0

    def update(self, error, dt):

        self.integral += error * dt

        derivative = (error - self.prev_error) / dt

        out = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.prev_error = error

        return out