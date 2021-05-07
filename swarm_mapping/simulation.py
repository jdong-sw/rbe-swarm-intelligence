import cv2

from swarm_mapping.world import World
import numpy as np

WIDTH = 100
HEIGHT = 100
EXPLORE_THRESHOLD = 0.8
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 800

class Simulation:
    # takes a list of simulation parameters and runs them in succession (or parallel?)
    # params: [[num_agents, marker_size, sensor_range]], +agent_velocity?
    def __init__(self, parameters: list, map_width, map_height):
        self.params = parameters
        self.map_width = map_width
        self.map_height = map_height
        self.size = (map_width+6)*(map_height+6)

    def start_sim(self):
        # run simulation for each line in parameters
        for sim_params in self.params:
            num_agents = sim_params[0]
            marker_size = sim_params[1]
            sensor_range = sim_params[2]
            world = self.create_world(num_agents, marker_size, sensor_range)
            unexplored = 1
            # run sim until explored or TODO: other reasons: agents all die, nothing new explored for too long, etc
            while unexplored >= EXPLORE_THRESHOLD:
                world.step()
                # calculate percentage of the entire map that is unexplored
                # TODO: maybe base on total empty spaces uncovered?
                unexplored = np.count_nonzero(world.agents_map == -2) / self.size
                print(unexplored)
                agents_map = world.update_agents_map()
                frame = world.render(agents_map)
                frame = cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation = cv2.INTER_AREA)
                cv2.imshow('Agent Map',cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                frame2 = world.render()
                frame2 = cv2.resize(frame2, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation = cv2.INTER_AREA)
                cv2.imshow('Sim',cv2.cvtColor((frame2*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                world.step()
            cv2.destroyAllWindows()

    def create_world(self, num_agents, marker_size, sensor_range):
        world = World(self.map_width, self.map_height, num_agents,
                      space_fill=0.5, hazard_fill=0.2, fast=False,
                      sensor_range=sensor_range, marker_size=marker_size)
        return world

    def check_discovered(self, world):
        agent_map = world.agents_map
        # go into the world and find how many squares != -2
        discovered_cells = agent_map.count(-2)
        discovered = discovered_cells / self.size
        return discovered

    # TODO: create a loop to generate a World for each parameter set
        # check_discovered every iteration and stop when threshold is reached, then calculate statistics

    # calculate/record various statistics about the run:
    # TODO: number of agents dead
    # TODO: steps to reach discovered threshold
