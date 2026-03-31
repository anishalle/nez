#!/usr/bin/env python3

import RocketSim as rs

# Make an arena instance (this is where our simulation takes place, has its own btDynamicsWorld instance)
arena = rs.Arena(rs.GameMode.SOCCAR)

x = arena.get_boost_pads()
