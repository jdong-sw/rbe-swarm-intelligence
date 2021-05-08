from swarm_mapping.simulation import Simulation

# Display size
display_width = 800
display_height = 800

# params: [[num_agents, marker_size, haz_fill, seed]]
params = [[100, 3, .2, 1],
          [100, 3, .2, 2],
          # [100, 3, .2, 3],
          # [100, 3, .2, 4],
          # [100, 3, .2, 5],
          # [100, 3, .2, 6],
          ]
height = 100
width = 100
explored_thresh = .6

sim = Simulation(params, height, width, explored_thresh)
