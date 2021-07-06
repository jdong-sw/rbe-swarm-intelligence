# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 14:18:10 2021

@authors: John Dong, Adam Santos
"""

from .map import Map
from .motion import MotionGenerator
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
_DEAD = 4

# Colors for rendering
_EMPTY_COLOR = [1,1,1]
_WALL_COLOR = [0,0,0]
_HAZARD_COLOR = [1,0,0]
_AGENT_COLOR = [0,.8,.8]
_MARKER_COLOR = [0,0,1]
_UNEXPLORED_COLOR = [.5,.5,.5]
_DEAD_COLOR = [0,0,1]


# Motion Parameters
_ALPHA = 1
_BETA = 0.4
_GAMMA = 0.1
_DELTA = 0.1

# Agent parameters
_VEL = 1

class World:
    def __init__(self, width, height, num_agents,
                 space_fill=0.5, hazard_fill=0.2, fast=True,
                 sensor_range=1, imaging_range=5, marker_size=3, m=None,
                 motion="diffuse"):
        self.width = width
        self.height = height
        self.num_agents = num_agents
        self.sensor_range = sensor_range
        self.imaging_range = imaging_range
        self.marker_size = marker_size
        self.motion = motion

        # Generate map
        if m is None:
            self.map = Map(width, height, space_fill, hazard_fill, fast)
            self.grid = self.map.grid.copy()
        else:
            self.map = m
            self.grid = m.grid.copy()
        self.state = self.grid.copy()
        self._add_borders()
        self.visual_map = self.map.render()
        self.agents_map = np.full((width + 2*self.imaging_range, 
                                   height + 2*self.imaging_range),
                                  _UNEXPLORED)
        
        # Populate with agents
        self._populate()

        # Update map state
        self._update_state()


    def step(self, debug=False):
        """
        Run a step of the simulation, which involves updating the state of
        every agent and updating the explored map.

        Parameters
        ----------
        debug : bool, optional
            Whether to print debug statements. The default is False.

        Returns
        -------
        None.

        """
        for agent in self.agents:
            agent.update(1, debug)
        self._update_state()


    def set_marker(self, radius):
        """
        Set the marker radius

        Parameters
        ----------
        radius : int
            Radius of the hazard marker.

        Returns
        -------
        None.

        """
        self.marker_size = radius
        for agent in self.agents:
            agent.set_marker(radius)


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
        image = np.where(image == _DEAD, _DEAD_COLOR, image)
        return image


    def show(self, title=None, size=(5,5), fignum=21, show_map=True):
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
        show_map : bool, optional
            Whether to show the currently mapped area

        Returns
        -------
        None.

        """
        image = self.render()
        
        if show_map:
            shared_map = self.render(self.agents_map)
            image = np.concatenate((image, shared_map), axis=1)
        
        plt.figure(num=fignum,figsize=size);
        plt.imshow(image);

        if title is not None:
            plt.title(title);


    def _populate(self):
        occupied = set()
        self.agents = []
        for i in range(self.num_agents):
            # Get random position in safe zone
            while True:
                # Get safe zone
                safe_x, safe_y, safe_size = self.map.safe_zone
                safe_x += self.imaging_range
                safe_y += self.imaging_range

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
            a = Agent(i, self, pos, vel, self.sensor_range, self.imaging_range, self.marker_size)
            a.motion_generator.select_motion(self.motion)
            self.agents.append(a)
            occupied.add((pixel[0], pixel[1]))


    def _update_state(self):
        state = self.grid.copy()
        for agent in self.agents:
            x, y = np.round(agent.pos).astype(int)
            if agent.alive:
                state[y, x] = _AGENT
            else:
                state[y, x] = _DEAD
                cv2.circle(state, (x,y), self.marker_size, _MARKER, 1)
                # No need to update the shared map, let agents discover
                #cv2.circle(self.agents_map, (x,y), self.marker_size, _MARKER, 1)

        self.state = state


    def _add_borders(self):
        self.grid = cv2.copyMakeBorder(
            self.map.grid,
            top=self.imaging_range,
            bottom=self.imaging_range,
            left=self.imaging_range,
            right=self.imaging_range,
            borderType=cv2.BORDER_CONSTANT,
            value=_WALL
        )



class Agent:
    def __init__(self, num, world, init_pos, init_vel, 
                 sensor_range, imaging_range, marker_size):
        self.num = num
        self.world = world
        self.pos = np.array(init_pos)
        self.vel = np.array(init_vel)
        self.prox_range = sensor_range
        self.imaging_range = imaging_range
        self.marker_size = marker_size
        self.alive = True
        self.motion_generator = MotionGenerator(self)

        # Agent's discovered map
        self.agent_map = np.full((world.width + imaging_range*2,
                                  world.height + imaging_range*2), 
                                 _UNEXPLORED)
        
        # Shared map between all agents
        self.shared_map = world.agents_map

        # Mask to help with proximity sensing and imaging
        self._init_sensor()


    def update(self, dt, debug=False):
        """
        Update current position and velocity of agent based on surroundings.

        Parameters
        ----------
        dt : float
            Passage of time, typically use one.
        debug : bool, optional
            Whether to print debug info. The default is False.

        Returns
        -------
        None.

        """
        # Do nothing if dead
        if not self.alive:
            return
        
        self._update_vel_beta()

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
        if debug:
            print("Tile:", tile)
        pxl_distance = _get_distance(new_pixel, old_pixel)
        if tile == _EMPTY or tile == _HAZARD or tile == _MARKER or pxl_distance == 0:
            self.pos = new_pos
        elif debug:
            print("Collision.")

        # Check if on hazard, update alive state
        if self.world.state[new_pixel[1], new_pixel[0]] == _HAZARD:
            self.alive = False

        # Update agents map
        self._update_map()
        
        
    def set_marker(self, radius):
        self.marker_size = radius
        self._init_sensor()
        
        
    def multisense(self):
        """
        Get proximity and camera sensor data.

        Returns
        -------
        proximity : numpy.ndarray
            2D array of nearby obstacles.
        image : TYPE
            2D array of environment.

        """
        proximity = self.proximity()
        image = self.camera()
        return proximity, image
        
        
    def proximity(self):
        """
        Get proximity data as a 2D array where 0 is empty space and 1 is
        obstacle.

        Returns
        -------
        proximity : numpy.ndarray
            2D array of nearby obstacles.

        """
        x, y = np.round(self.pos).astype(int)
        proximity = self.world.state[y - self.prox_range : y + self.prox_range + 1,
                                     x - self.prox_range : x + self.prox_range + 1]
        if self.prox_range > 1:
            proximity = np.multiply(proximity, self._mask).clip(0, 1)
        else:
            proximity = proximity.clip(0, 1)
        return proximity
    
    
    def camera(self):
        """
        Get nearby environmental data as a 2D array. Can see everything except
        hazard tiles.

        Returns
        -------
        imaging : numpy.ndarray
            2D array of environment.

        """
        x, y = np.round(self.pos).astype(int)
        imaging = self.world.state[y - self.imaging_range : y + self.imaging_range + 1,
                                   x - self.imaging_range : x + self.imaging_range + 1]
        return imaging
        

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
        empty_mask = image == _EMPTY
        wall_mask = image == _WALL
        image[empty_mask] = _WALL
        image[wall_mask] = _EMPTY
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
        size = 2*self.imaging_range + 1
        self._mask = np.zeros((size, size), int)
        if size > 3:
            cv2.circle(self._mask, (self.imaging_range, self.imaging_range), 
                       self.marker_size+1, 1, -1)
        else:
            self._mask.fill(1)
            
            
    def _update_map(self):
        if not self.alive:
            return
        
        x, y = np.round(self.pos).astype(int)
        observation = self.camera();

        # Remove observations of self/other drones and hazards
        observation = np.where(observation==_AGENT, 0, observation)
        observation = np.where(observation==_DEAD, 0, observation)
        observation = np.where(observation==_HAZARD, 0, observation)

        # Save the observation in the agent's map
        self.agent_map[y - self.imaging_range: y + self.imaging_range + 1,
                       x - self.imaging_range: x + self.imaging_range + 1] = observation
        
        # Also update shared central map
        self.shared_map[y - self.imaging_range: y + self.imaging_range + 1,
                        x - self.imaging_range: x + self.imaging_range + 1] = observation
            
    
    def _update_vel(self):
        proximity, image = self.multisense()
        obj = self.motion_generator.get_vel(proximity)
        search = self._search(image)
        escape = self._escape(image)
        noise = 2 * np.random.rand(2) - 1

        if not (escape == 0).all():
            self.vel = escape * _VEL
            return
        
        vel = obj + 0.4 * search + 0.1 * noise
        mag = np.linalg.norm(obj)
        vel = vel / mag
        self.vel = vel * _VEL

    def _update_vel_beta(self):
        # Bias towards unexplored local areas
        proximity, image = self.multisense()

        obj = self.motion_generator.get_vel(proximity)
        search = self._search(image)
        noise = 0

        # old noise used on every step:
        # noise = 2 * np.random.rand(2) - 1

        # Escape if caught inside a hazard circle
        escape = self._escape(image)
        if not (escape == 0).all():
            self.vel = escape * _VEL
            return

        vel = self.motion_generator.get_vel(proximity, image, _ALPHA, _BETA, _GAMMA, _DELTA)
        
        vel = obj + 0.4 * search + noise
        mag = np.linalg.norm(obj)
        vel = vel / mag
        self.vel = vel * _VEL


    def _search(self, image):
        # Get only unexplored areas
        image = image.clip(_UNEXPLORED, 0)
        image = np.where(image == _HAZARD, 0, image)
        image = -0.5 * image
        
        # Get unit vector to centroid of unexplored area
        vect = _calc_centroid(image)
        if vect is None:
            return np.zeros(2)
        else:
            return vect
    
    
    def _escape(self, image):
        # Check if within marker
        if image.shape != self._mask.shape:
            return np.zeros(2)
        image = np.multiply(image, self._mask)
        if _DEAD in image:
            # Return strong velocity away from marker
            image = (image == _DEAD).astype(float)
            vect = _calc_centroid(image)
            if vect is None:
                return np.zeros(2)
            return -1*vect
        else:
            return np.zeros(2)


    def __str__(self):
        return f"Agent at ({self.pos[0]},{self.pos[1]}), with " + \
            f"velocity ({self.vel[0]}, {self.vel[1]}), alive: {self.alive}"



def _get_distance(a, b):
    return np.sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2))
    

def _calc_centroid(im):
    height, width = im.shape
    size = np.floor(width / 2.0)
    
    M = cv2.moments(im)
    if M['m00'] == 0:
        return None
    
    cx = M['m10']/M['m00'] - size
    cy = M['m01']/M['m00'] - size
    vect = np.array([cx, cy])
    
    mag = np.linalg.norm(vect)
    if mag == 0:
        return None

    return vect / mag