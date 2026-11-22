"""Differential-drive kinematics and odometry (ROS-independent).

Wheel speeds and positions are in wheel turns (turn/s, turn), positive when
the wheel rolls the robot forward. The motor sign (how the motor is mounted)
is applied by the caller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Kinematics:
    wheel_diameter: float  # [m]
    wheel_separation: float  # [m], between the wheel contact points

    def __post_init__(self):
        if not (self.wheel_diameter > 0 and self.wheel_separation > 0):
            raise ValueError("wheel_diameter and wheel_separation must be > 0")

    @property
    def circumference(self) -> float:
        return math.pi * self.wheel_diameter

    def wheel_speeds(self, linear: float, angular: float) -> tuple[float, float]:
        """Body twist (m/s, rad/s) -> (left, right) wheel speeds in turn/s."""
        half = angular * self.wheel_separation / 2
        return (
            (linear - half) / self.circumference,
            (linear + half) / self.circumference,
        )

    def body_twist(self, left: float, right: float) -> tuple[float, float]:
        """(left, right) wheel speeds in turn/s -> body twist (m/s, rad/s)."""
        vl, vr = left * self.circumference, right * self.circumference
        return (vl + vr) / 2, (vr - vl) / self.wheel_separation


def limit_wheel_speeds(
    left: float, right: float, max_speed: float
) -> tuple[float, float]:
    """Scale both wheels by the same factor so neither exceeds max_speed.

    Keeps the path curvature (the ratio of the wheel speeds).
    """
    peak = max(abs(left), abs(right))
    if max_speed <= 0 or peak <= max_speed:
        return left, right
    k = max_speed / peak
    return left * k, right * k


class Odometry:
    """Pose from wheel position increments (midpoint integration)."""

    def __init__(self, kin: Kinematics):
        self.kin = kin
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self._last: tuple[float, float] | None = None

    def update(self, left_pos: float, right_pos: float) -> None:
        """Advance with the wheels' absolute positions (turns)."""
        if self._last is None:
            self._last = (left_pos, right_pos)
            return
        dl = (left_pos - self._last[0]) * self.kin.circumference
        dr = (right_pos - self._last[1]) * self.kin.circumference
        self._last = (left_pos, right_pos)
        ds = (dl + dr) / 2
        dth = (dr - dl) / self.kin.wheel_separation
        mid = self.theta + dth / 2
        self.x += ds * math.cos(mid)
        self.y += ds * math.sin(mid)
        self.theta = math.atan2(math.sin(self.theta + dth), math.cos(self.theta + dth))

    def quaternion(self) -> tuple[float, float, float, float]:
        """(x, y, z, w) of the yaw."""
        return 0.0, 0.0, math.sin(self.theta / 2), math.cos(self.theta / 2)


def advance_setpoint(
    setpoint: float, speed: float, dt: float, position: float, max_lead: float
) -> float:
    """Integrate a position setpoint and keep it within max_lead of position.

    Same scheme as upstream astra_controller base_controller.py: the
    controller runs in position mode and the setpoint moves at the commanded
    speed. The clamp limits the wind-up while the wheel is blocked.
    """
    setpoint += speed * dt
    return min(max(setpoint, position - max_lead), position + max_lead)


def ramp_speed(
    current: float, target: float, accel: float, decel: float, dt: float
) -> float:
    """Move a speed toward target, at most accel*dt per step (decel*dt when slowing).

    Slowing (|speed| decreasing, or reversing) uses decel and stops at zero
    before a reversal; the next step accelerates the other way.
    """
    slowing = current != 0 and (
        target == 0 or (target > 0) != (current > 0) or abs(target) < abs(current)
    )
    if slowing:
        goal = target if (target > 0) == (current > 0) else 0.0
        step = decel * dt
    else:
        goal, step = target, accel * dt
    if abs(goal - current) <= step:
        return goal
    return current + step if goal > current else current - step


class WheelSetpoints:
    """Per-wheel speed and position setpoints in motor turns.

    ramp=True (input_mode "filter"): the speed follows the target within the
    accel/decel limits and is sent as input_vel with the integrated position.
    ramp=False (input_mode "trap"): the target speed is integrated directly
    and the firmware's trapezoidal trajectory limits the acceleration.
    """

    def __init__(
        self,
        positions: list[float],
        accel: float,
        decel: float,
        max_lead: float,
        ramp: bool,
    ):
        self.setpoints = list(positions)
        self.speeds = [0.0] * len(positions)
        self.accel, self.decel, self.max_lead, self.ramp = accel, decel, max_lead, ramp

    def update(
        self, targets: list[float], positions: list[float], dt: float
    ) -> tuple[list[float], list[float]]:
        for i, (target, pos) in enumerate(zip(targets, positions)):
            if self.ramp:
                v = ramp_speed(self.speeds[i], target, self.accel, self.decel, dt)
            else:
                v = target
            self.speeds[i] = v
            self.setpoints[i] = advance_setpoint(
                self.setpoints[i], v, dt, pos, self.max_lead
            )
        return list(self.setpoints), list(self.speeds)
