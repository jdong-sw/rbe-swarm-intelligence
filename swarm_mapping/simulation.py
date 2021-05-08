import cv2
from swarm_mapping.world import World
import numpy as np

# Map settings
WIDTH = 100
HEIGHT = 100
SPACE_FILL = 0.5
HAZ_FILL = 0.2

# Display settings
DISPLAY_WIDTH = 800
DISPLAY_HEIGHT = 800
PRINT_FREQ = 50

# Sim settings
MAX_ITERATIONS = 1000


# TODO: some kind of controls for the maps, fixed seeds or something (maps are highly imbalanced)

class Simulation:
    # takes a list of simulation parameters and runs them in succession (or parallel?)
    # params: [[num_agents, marker_size, sensor_range]], +agent_velocity?
    def __init__(self, parameters: list, map_width, map_height, explore_thresh, show_map=True):
        self.params = parameters
        self.map_width = map_width
        self.map_height = map_height
        self.explore_thresh = explore_thresh
        self.show_map = show_map

        # Stats for analysis
        self.explored_data = []
        self.dead_data = []

        # +6 is to match the border present on the Map() class
        self.size = (map_width+6)*(map_height+6)

    def start_sim(self):
        # run simulation for each line in parameters
        # check every iteration and stop when any threshold is reached
        for sim_params in self.params:
            print("Running new simulation...")
            num_agents = sim_params[0]
            marker_size = sim_params[1]
            sensor_range = sim_params[2]
            world = self.create_world(num_agents, marker_size, sensor_range)
            step = 0
            explored = [0]
            dead = [0]
            # run sim until explored or until max_iterations is reached
            while explored[-1] <= self.explore_thresh\
                    and step < MAX_ITERATIONS:
                world.step()
                step += 1
                # calculate percentage of the entire map that is explored
                explored.append(self.check_spaces_discovered(world))
                dead.append(self.check_dead(world))

                if step % PRINT_FREQ == 0:
                    print("--------------")
                    print("Step: " + str(step))
                    print("Explored: " + str("{:.2f}". format(explored[-1]*100)) + "%")
                    print("Dead: " + str(dead[-1]))

                if self.show_map:
                    image = world.render()
                    shared_map = world.render(world.agents_map)
                    frame = np.concatenate((image, shared_map), axis=1)
                    frame = cv2.resize(frame, (DISPLAY_WIDTH*2, DISPLAY_HEIGHT), interpolation = cv2.INTER_AREA)
                    cv2.imshow('Simulation Map', cv2.cvtColor((frame*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            self.explored_data.append(explored)
            self.dead_data.append(dead)
            print("=================================")
            print("Simulation finished on step " + str(step))
            print("Explored percentage: " + str("{:.2f}". format(explored[-1]*100)) + "%" + ", Dead: " + str(dead[-1]))

            cv2.destroyAllWindows()

    def create_world(self, num_agents, marker_size, sensor_range):
        world = World(self.map_width, self.map_height, num_agents,
                      space_fill=SPACE_FILL, hazard_fill=HAZ_FILL, fast=False,
                      sensor_range=sensor_range, marker_size=marker_size)
        return world

    @staticmethod
    def check_spaces_discovered(world):
        # check how many tiles the agents have revealed by checking how many squares != -2 ('unknown')
        discovered_empty_cells = np.count_nonzero(world.agents_map == 0)
        total_empty_cells = np.count_nonzero(world.map.grid == 0)
        return discovered_empty_cells / total_empty_cells

    @staticmethod
    def check_dead(world):
        # returns number of agents dead
        agents_total = len(world.agents)
        agents_dead = 0
        for i in range(agents_total):
            alive = world.agents[i].alive
            if not alive:
                agents_dead += 1
        return agents_dead

