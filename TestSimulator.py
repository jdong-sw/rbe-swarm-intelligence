from swarm_mapping.simulation import Simulation
import cv2
import numpy as np
# Display size
display_width = 800
display_height = 800

# params: [[num_agents, marker_size, sensor_range]], +agent_velocity?
params = [[100, 3, 3],
          [200, 1, 1]]
height = 100
width = 100

sim = Simulation(params, height, width)
sim.start_sim()

# while True:
#     agents_map = world.update_agents_map()
#     frame = world.render(agents_map)
#     frame = cv2.resize(frame, (display_width, display_height), interpolation = cv2.INTER_AREA)
#     cv2.imshow('Agent Map',cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     frame2 = world.render()
#     frame2 = cv2.resize(frame2, (display_width, display_height), interpolation = cv2.INTER_AREA)
#     cv2.imshow('Sim',cv2.cvtColor((frame2*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     world.step()
#     step += 1
# cv2.destroyAllWindows()
