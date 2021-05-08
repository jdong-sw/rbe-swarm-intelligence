import cv2
from swarm_mapping.world import World
import numpy as np

WIDTH = 100
HEIGHT = 100
SPACE_FILL = 0.5
HAZ_FILL = 0.2

DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 800

MAX_ITERATIONS = 100000

class Simulation:
    # takes a list of simulation parameters and runs them in succession (or parallel?)
    # params: [[num_agents, marker_size, sensor_range]], +agent_velocity?
    def __init__(self, parameters: list, map_width, map_height, explore_thresh, show_map=True):
        self.params = parameters
        self.map_width = map_width
        self.map_height = map_height
        self.explore_thresh = explore_thresh
        self.show_map = show_map

        self.size = (map_width+6)*(map_height+6)

    def start_sim(self):
        # run simulation for each line in parameters
        for sim_params in self.params:
            num_agents = sim_params[0]
            marker_size = sim_params[1]
            sensor_range = sim_params[2]
            world = self.create_world(num_agents, marker_size, sensor_range)
            step = 0
            unexplored = 1
            # run sim until explored or TODO: other reasons: agents all die, nothing new explored for too long, etc
            while unexplored >= self.explore_thresh\
                    and step < MAX_ITERATIONS:
                world.step()
                # calculate percentage of the entire map that is unexplored
                # TODO: maybe base on total empty spaces uncovered?
                unexplored = np.count_nonzero(world.agents_map == -2) / self.size
                print(unexplored)

                if self.show_map:
                    image = world.render()
                    shared_map = world.render(world.agents_map)
                    frame = np.concatenate((image, shared_map), axis=1)
                    frame = cv2.resize(frame, (DISPLAY_WIDTH*2, DISPLAY_HEIGHT), interpolation = cv2.INTER_AREA)
                    cv2.imshow('Agent Map',cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                world.step()
                step += 1
            cv2.destroyAllWindows()

    def create_world(self, num_agents, marker_size, sensor_range):
        world = World(self.map_width, self.map_height, num_agents,
                      space_fill=SPACE_FILL, hazard_fill=HAZ_FILL, fast=False,
                      sensor_range=sensor_range, marker_size=marker_size)
        return world

    def check_discovered(self, world):
        # check how many tiles the agents have revealed by checking how many squares != -2 ('unknown')
        agent_map = world.agents_map
        discovered_cells = agent_map.count(-2)
        discovered = discovered_cells / self.size
        return discovered

    @staticmethod
    def check_dead(self, world):
        # returns number of agents dead
        agents_total = len(world.agents)
        agents_dead = 0
        for i in range(agents_total):
            alive = world.agents[i].alive
            if not alive:
                agents_dead += 1
        return agents_dead

# TODO: create a loop to generate a World for each parameter set
        # check_discovered every iteration and stop when threshold is reached, then calculate statistics

    # calculate/record various statistics about the run:
