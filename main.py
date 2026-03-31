#!/usr/bin/env python3

import RocketSim as rs

# Make an arena instance (this is where our simulation takes place, has its own btDynamicsWorld instance)
arena = rs.Arena(rs.GameMode.SOCCAR)

# Make a new car
car = arena.add_car(rs.Team.BLUE)

# Set up an initial state for our car
car.set_state(rs.CarState(pos=rs.Vec(z=17), vel=rs.Vec(x=50)))

# Setup a ball state
arena.ball.set_state(rs.BallState(pos=rs.Vec(y=400, z=100)))

# Make our car drive forward and turn
car.set_controls(rs.CarControls(throttle=1, steer=1))

# Simulate for 100 ticks
arena.step(100)

# Lets see where our car went!
print(f"After {arena.tick_count} ticks, our car is at: {car.get_state().pos:.2f}")
