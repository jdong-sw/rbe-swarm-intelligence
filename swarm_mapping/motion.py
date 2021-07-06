# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:29:20 2021

@author: John Dong
"""

import numpy as np
import cv2


# Grid values
from swarm_mapping.cauchy import SampleCauchyDistribution

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

# Agent parameters
_VEL = 1


class MotionGenerator:
    def __init__(self, agent):
        self._agent = agent
        self._motion = self._diffuse
        self.steps_since_change = 0
        
        
    def get_vel(self, proximity, image, alpha, beta, gamma, delta):
        """
        Obtains the velocity vector given the proximity sensor readings and 
        choice of motion algorithm.

        Parameters
        ----------
        proximity : numpy.ndarray
            Agent's proximity sensor reading.

        Returns
        -------
        numpy.ndarray
            Velocity vector from current motion algorithm.

        """
        return self._motion(proximity, image, alpha, beta, gamma, delta)
        
    
    def select_motion(self, motion, alpha=0):
        """
        Select motion algorithm for velocity updating.

        Parameters
        ----------
        motion : string
            Name of motion algorithm to use. Accepts "diffuse", "rcw".

        Returns
        -------
        None.

        """
        if (motion.lower() == "diffuse"):
            self._motion = self._diffuse

        if (motion.lower() == "rcw"):
            self._motion = self._random_walk
        
    def _diffuse(self, proximity, image, alpha, beta, gamma, delta):
        # Implementation of basic diffusive behavior
        
        #explored vector
        search = self._search(image)

        #noise vector
        noise = 2 * np.random.rand(2) - 1
        
        # Get unit vector to obstacle
        obj = _calc_centroid(proximity)
        if obj is None:
            return self._agent.vel
        
        # Apply reflection across object vector
        obj_vel = -1*(2 * np.dot(self._agent.vel, obj)/np.dot(obj, obj) * obj - self._agent.vel)

        vel = alpha*obj_vel + beta*search + delta*noise
        mag = np.linalg.norm(obj_vel)
        vel = vel / mag
        return vel


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


    def _random_walk(self):
        # Initialize a cauchy sample distribution
        sca = SampleCauchyDistribution(rho=self.rho, sample_size=10**5)
        # count how many steps since last direction change
        self.steps_since_change += 1
        # check for direction change
        change = _sample_custom_power(self.alpha, self.steps_since_change)
        if change:
            # change direction
            velx = self._agent.vel[0]
            vely = self._agent.vel[1]
            theta = np.arctan(vely, velx)
            theta += sca.pick_sample()
            new_vel = [np.cos(theta), np.sin(theta)]
            self.steps_since_change = 0

        vel = alpha*obj_vel + beta*search + delta*noise

        return new_vel


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


def _sample_custom_power(alpha, step):
    # beta: scale factor
    beta = 5
    return np.exp(-(step**alpha) / (2*alpha*beta)) / np.sqrt(alpha*beta)
