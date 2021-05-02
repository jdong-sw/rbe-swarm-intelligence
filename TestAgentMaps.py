from swarm_mapping.world import World
import cv2
import numpy as np
# Display size
display_width = 800
display_height = 800

world = World(100, 100, 20,
              space_fill=0.4, hazard_fill=0.2, fast=False,
              sensor_range=3, marker_size=3)
step = 0
world.step()
while True:
    frame = world.render(world.update_agents_map())
    frame = cv2.resize(frame, (display_width, display_height), interpolation = cv2.INTER_AREA)
    cv2.imshow('Agent Map',cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame2 = world.render()
    frame2 = cv2.resize(frame2, (display_width, display_height), interpolation = cv2.INTER_AREA)
    cv2.imshow('Sim',cv2.cvtColor((frame2*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    world.step()
    step += 1
cv2.destroyAllWindows()
