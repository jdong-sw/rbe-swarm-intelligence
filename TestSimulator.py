from swarm_mapping.simulation import Simulation

# Display size
display_width = 800
display_height = 800

# params: [[num_agents, marker_size, sensor_range]], +agent_velocity?
params = [[100, 3, 3],
          [200, 1, 1]]
height = 100
width = 100
explored_thresh = .98

sim = Simulation(params, height, width, explored_thresh)
sim.start_sim()
