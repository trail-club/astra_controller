import math

import pytest

from astra_controller.diff_drive import (
    Kinematics,
    Odometry,
    WheelSetpoints,
    advance_setpoint,
    limit_wheel_speeds,
    ramp_speed,
)

KIN = Kinematics(0.139, 0.44)


def test_wheel_speeds_match_upstream():
    # upstream base_controller.set_vel (before the install direction)
    v, w = 0.3, 0.8
    left = (v - w * 0.44 / 2) / (math.pi * 0.139)
    right = (v + w * 0.44 / 2) / (math.pi * 0.139)
    assert KIN.wheel_speeds(v, w) == pytest.approx((left, right))


def test_body_twist_inverts_wheel_speeds():
    assert KIN.body_twist(*KIN.wheel_speeds(-0.2, 1.3)) == pytest.approx((-0.2, 1.3))


def test_invalid_geometry():
    with pytest.raises(ValueError):
        Kinematics(0.0, 0.44)


def test_limit_keeps_ratio():
    left, right = limit_wheel_speeds(1.0, 4.0, 2.0)
    assert (left, right) == pytest.approx((0.5, 2.0))
    assert limit_wheel_speeds(1.0, -1.5, 2.0) == (1.0, -1.5)


def test_odometry_straight_and_turn():
    odom = Odometry(KIN)
    odom.update(10.0, 20.0)  # first sample only sets the reference
    assert (odom.x, odom.y, odom.theta) == (0.0, 0.0, 0.0)
    odom.update(11.0, 21.0)
    assert odom.x == pytest.approx(math.pi * 0.139)
    assert odom.y == pytest.approx(0.0)
    # turn in place by +90 deg: right forward, left backward
    quarter = (math.pi / 2) * 0.44 / 2 / (math.pi * 0.139)
    odom.update(11.0 - quarter, 21.0 + quarter)
    assert odom.theta == pytest.approx(math.pi / 2)
    assert odom.x == pytest.approx(math.pi * 0.139)
    qx, qy, qz, qw = odom.quaternion()
    assert (qz, qw) == pytest.approx((math.sin(math.pi / 4), math.cos(math.pi / 4)))


def test_odometry_wraps_theta():
    odom = Odometry(KIN)
    odom.update(0.0, 0.0)
    half_turn = math.pi * 0.44 / 2 / (math.pi * 0.139)
    for k in range(1, 4):
        odom.update(-half_turn * k / 1.5, half_turn * k / 1.5)
    assert -math.pi <= odom.theta <= math.pi
    assert odom.theta == pytest.approx(0.0, abs=1e-9)


def test_setpoint_clamped_to_lead():
    assert advance_setpoint(0.0, 1.0, 0.1, 0.0, 2.0) == pytest.approx(0.1)
    assert advance_setpoint(1.95, 1.0, 0.1, 0.0, 2.0) == 2.0
    assert advance_setpoint(-1.95, -1.0, 0.1, 0.0, 2.0) == -2.0


def test_ramp_speed_accel_decel_and_reverse():
    assert ramp_speed(0.0, 1.0, 2.0, 4.0, 0.1) == pytest.approx(0.2)
    assert ramp_speed(0.9, 1.0, 2.0, 4.0, 0.1) == 1.0
    assert ramp_speed(1.0, 0.0, 2.0, 4.0, 0.1) == pytest.approx(0.6)  # decel
    assert ramp_speed(0.3, 0.0, 2.0, 4.0, 0.1) == 0.0
    assert ramp_speed(1.0, 0.5, 2.0, 4.0, 0.1) == pytest.approx(0.6)
    # reversing: decelerate to zero first, then accelerate the other way
    assert ramp_speed(0.2, -1.0, 2.0, 4.0, 0.1) == 0.0
    assert ramp_speed(0.0, -1.0, 2.0, 4.0, 0.1) == pytest.approx(-0.2)
    assert ramp_speed(-1.0, 0.0, 2.0, 4.0, 0.1) == pytest.approx(-0.6)


def test_wheel_setpoints_ramp_and_stop_distance():
    w = WheelSetpoints([0.0, 10.0], accel=2.0, decel=2.0, max_lead=2.0, ramp=True)
    dt = 0.05
    for _ in range(20):  # 1 s at target 1 turn/s: 0.5 s ramp + 0.5 s cruise
        sp, v = w.update([1.0, -1.0], [0.0, 10.0], dt)
    assert v == pytest.approx([1.0, -1.0])
    run = sp[0]
    assert run == pytest.approx(0.25 + 0.5, abs=0.03)
    for _ in range(20):
        sp, v = w.update([0.0, 0.0], [0.0, 10.0], dt)
    assert v == [0.0, 0.0]
    # braking distance v^2 / 2a = 0.25 turn
    assert sp[0] - run == pytest.approx(0.25, abs=0.03)
    assert sp[1] == pytest.approx(10.0 - (sp[0] - 0.0), abs=1e-9)


def test_wheel_setpoints_without_ramp_integrates_target():
    w = WheelSetpoints([0.0], accel=2.0, decel=2.0, max_lead=2.0, ramp=False)
    sp, v = w.update([1.0], [0.0], 0.1)
    assert (sp, v) == ([pytest.approx(0.1)], [1.0])
