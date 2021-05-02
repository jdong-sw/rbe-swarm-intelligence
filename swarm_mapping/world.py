# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:18:10 2021

@authors: John Dong, Adam Santos
"""

from .map import Map
import cv2
import matplotlib.pyplot as plt
import numpy as np
import random

# Grid values
_EMPTY = 0
_WALL = 1
_HAZARD = -1
_AGENT = 2
_MARKER = 3
_UNEXPLORED = -2

# Colors for rendering
_EMPTY_COLOR = [1,1,1]
_WALL_COLOR = [0,0,0]
_HAZARD_COLOR = [1,0,0]
_AGENT_COLOR = [0,0,1]
_MARKER_COLOR = [0,1,0]
_UNEXPLORED_COLOR = [.5,.5,.5]

# Agent parameters
_VEL = 1

class World:
    def __init__(self, width, height, num_agents,
                 space_fill=0.5, hazard_fill=0.2, fast=True,
                 sensor_range=1, marker_size=3):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.sensor_range = sensor_range
        self.marker_size = marker_size

        # Generate map
        self.map = Map(width, height, space_fill, hazard_fill, fast)
        self._add_borders()
        self.state = self.map.grid.copy()
        self.visual_map = self.map.render()
        self.agents_map = np.full((width+self.sensor_range, height+self.sensor_range), -2)
        # Populate with agents
        self._populate()

        # Update map state
        self._update_state()


    def step(self, debug=False):
        for agent in self.agents:
            agent.update(1, debug)
        self._update_state()


    def render(self, frame=None):
        """
        Renders the current state of the world as a RGB image numpy array.

        Returns
        -------
        image : numpy.ndarray
            (Height x Width x 3) RGB image of the world.

        """
        if frame is None:
            frame = self.state
        empty_mask = frame == _EMPTY
        wall_mask = frame == _WALL
        image = frame.copy()
        image[empty_mask] = _WALL
        image[wall_mask] = _EMPTY
        image = np.expand_dims(image, axis=2)
        image = np.where(image == _EMPTY, _WALL_COLOR, image)
        image = np.where(image == _WALL, _EMPTY_COLOR, image)
        image = np.where(image == _HAZARD, _HAZARD_COLOR, image)
        image = np.where(image == _AGENT, _AGENT_COLOR, image)
        image = np.where(image == _MARKER, _MARKER_COLOR, image)
        image = np.where(image == _UNEXPLORED, _UNEXPLORED_COLOR, image)
        return image


    def show(self, title=None, size=(5,5), fignum=21):
        """
        Plots the current state of the world using matplotlib.

        Parameters
        ----------
        title : string, optional
            Title of the plot. The default is None.
        size : (float, float), optional
            (width, height) of the plot in inches. The default is (5,5).
        fignum : int, optional
            Number for the figure. The default is 21.

        Returns
        -------
        None.

        """
        image = self.render()
        plt.figure(num=fignum,figsize=size);
        plt.imshow(image);

        if title is not None:
            plt.title(title);

    def update_agents_map(self, title=None, size=(5,5), fignum=21):
        #iterate each map space until an agent has that place discovered (!=-2)
        for row in range(self.width):
            for col in range(self.height):
                if self.agents_map[row, col] == -2:
                    for agent in self.agents:
                        if agent.agent_map[row, col] != -2:
                            self.agents_map[row, col] = agent.agent_map[row, col]
                            break
        return self.agents_map


    def _populate(self):
        occupied = set()
        self.agents = []
        for i in range(self.num_agents):
            # Get random position in safe zone
            while True:
                # Get safe zone
                safe_x, safe_y, safe_size = self.map.safe_zone

                # Choose random x inside safe zone
                x = random.randint(0, int(safe_size))
                x = (1 if random.random() < 0.5 else -1) * x

                # Calculate possible y values inside zone given x
                y_max = round(np.sqrt(pow(safe_size, 2) - pow(x, 2)))
                y = random.randint(0, y_max)
                y = (1 if random.random() < 0.5 else -1) * y

                # Make sure tile is empty
                x = x + safe_x
                y = y + safe_y
                pos = (x, y)
                pixel = (round(x), round(y))
                if (self.state[pixel[1], pixel[0]] == _EMPTY) and \
                        (pixel not in occupied):
                    pos = np.array(pos)
                    break

            # Get random velocity
            vx = random.random()*2 - 1
            vy = random.random()*2 - 1
            vel = [vx, vy]
            vel = vel / np.linalg.norm(vel) * _VEL

            # Add agent
            a = Agent(i, self, pos, vel, self.sensor_range, self.marker_size)
            self.agents.append(a)
            occupied.add((pixel[0], pixel[1]))


    def _update_state(self):
        state = self.map.grid.copy()
        for agent in self.agents:
            x, y = np.round(agent.pos).astype(int)
            state[y, x] = _AGENT
            if not agent.alive:
                cv2.circle(self.map.grid, (x,y), self.marker_size, _MARKER, 1)
                cv2.circle(self.agents_map, (x,y), self.marker_size, _MARKER, 1)

        self.state = state


    def _add_borders(self):
        self.map.grid = cv2.copyMakeBorder(
            self.map.grid,
            top=self.sensor_range,
            bottom=self.sensor_range,
            left=self.sensor_range,
            right=self.sensor_range,
            borderType=cv2.BORDER_CONSTANT,
            value=_WALL
        )



class Agent:
    def __init__(self, num, world, init_pos, init_vel, sensor_range, marker_size):
        self.num = num
        self.world = world
        self.pos = np.array(init_pos)
        self.vel = np.array(init_vel)
        self.range = sensor_range
        self.marker_size = marker_size
        self.alive = True

        # Agent's discovered map
        self.agent_map = np.full((world.width+world.sensor_range*2, world.height+world.sensor_range*2), -2)

        # Mask to help with proximity sensing
        self._init_sensor()


    def update(self, dt, debug=False):
        # Do nothing if dead
        if not self.alive:
            return

        # Update velocity
        self._diffuse()

        # Update position
        if debug:
            print("Current position:", self.pos)
            print("Current velocity:", self.vel)

        new_pos = self.pos + self.vel*dt
        if debug:
            print("New position:", new_pos)

        old_pixel = np.round(self.pos).astype(int)
        new_pixel = np.round(new_pos).astype(int)
        if debug:
            print("Pixel position:", new_pixel)

        # Make sure no collisions and update position
        tile = self.world.state[new_pixel[1], new_pixel[0]]
        pxl_distance = _get_distance(new_pixel, old_pixel)
        if tile == _EMPTY or tile == _HAZARD or pxl_distance == 0:
            self.pos = new_pos

        # Check if on hazard, update alive state
        if self.world.state[new_pixel[1], new_pixel[0]] == _HAZARD:
            self.alive = False

        # Update agents map
        self.update_map()
    def proximity(self):
        x, y = np.round(self.pos).astype(int)
        proximity = self.world.state[y - self.range : y + self.range + 1,
                                     x - self.range : x + self.range + 1]
        proximity = np.multiply(proximity, self._mask).clip(0, 1)
        return proximity


    def update_map(self):
        x, y = np.round(self.pos).astype(int)
        observation = self.world.map.grid[y - self.range: y + self.range + 1,
                                       x - self.range: x + self.range + 1]

        # Remove observations of self/other drones and hazards
        observation = np.where(observation==2, 0, observation)
        observation = np.where(observation==-1, 0, observation)

        # Save the observation in the agent's map
        self.agent_map[y - self.range: y + self.range + 1,
                       x - self.range: x + self.range + 1] = observation

    def show_agent_map(self, title=None, size=(5, 5), fignum=21):
        """
        Plots the current state of the world using matplotlib.

        Parameters
        ----------
        title : string, optional
            Title of the plot. The default is None.
        size : (float, float), optional
            (width, height) of the plot in inches. The default is (5,5).
        fignum : int, optional
            Number for the figure. The default is 21.
        Returns
        -------
        None.
        """
        image = self.agent_map.copy()
        image = np.expand_dims(image, axis=2)
        image = np.where(image == _EMPTY, _WALL_COLOR, image)
        image = np.where(image == _WALL, _EMPTY_COLOR, image)
        image = np.where(image == _HAZARD, _HAZARD_COLOR, image)
        image = np.where(image == _UNEXPLORED, _UNEXPLORED_COLOR, image)
        image = np.where(image == _MARKER, _MARKER_COLOR, image)
        plt.figure(num=fignum, figsize=size)
        plt.imshow(image)

        if title is not None:
            plt.title(title)

        return image


    def _init_sensor(self):
        size = 2*self.range + 1
        self._mask = np.zeros((size, size), int)
        if size > 3:
            cv2.circle(self._mask, (self.range, self.range), self.range, 1, -1)
        else:
            self._mask.fill(1)


    def _diffuse(self):
        M = cv2.moments(self.proximity())
        if M['m00'] == 0:
            return
        cx = M['m10']/M['m00'] - self.range
        cy = M['m01']/M['m00'] - self.range
        obj = np.array([cx, cy])
        mag = np.linalg.norm(obj)
        if mag == 0:
            return
        obj = obj / mag
        self.vel = -obj * _VEL


    def __str__(self):
        return f"Agent at ({self.pos[0]},{self.pos[1]}), with " + \
            f"velocity ({self.vel[0]}, {self.vel[1]}), alive: {self.alive}"


def _get_distance(a, b):
        return np.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))