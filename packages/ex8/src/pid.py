#!/usr/bin/env python3
"""Reusable PID controller class (handy again for Lab 3)."""


class PID:
    def __init__(self, kp, ki, kd, integral_limit=1.0, output_limit=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit  # anti-windup clamp on the integral term
        self.output_limit = output_limit      # optional clamp on the control output
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_error = None
        self.prev_time = None

    def update(self, error, now):
        """Return the control input for this error. `now` is time in seconds."""
        # First call: no dt yet, so just use the P term
        if self.prev_time is None:
            self.prev_time = now
            self.prev_error = error
            return self._clamp(self.kp * error, self.output_limit)

        dt = now - self.prev_time
        if dt <= 0.0:
            dt = 1e-3  # guard against duplicate timestamps

        # Integral with anti-windup
        self.integral += error * dt
        self.integral = self._clamp(self.integral, self.integral_limit)

        # Derivative of the error
        derivative = (error - self.prev_error) / dt

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.prev_error = error
        self.prev_time = now
        return self._clamp(output, self.output_limit)

    @staticmethod
    def _clamp(value, limit):
        if limit is None:
            return value
        return max(-limit, min(limit, value))
