# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 15:29:20 2021

@author: John Dong
"""

import numpy as np
import cv2

class MotionGenerator:
    def __init__(self, agent):
        self._agent = agent
        self._motion = self._diffuse
        
        
    def get_vel(self, proximity):
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
        return self._motion(proximity)
        
    
    def select_motion(self, motion):
        """
        Select motion algorithm for velocity updating.

        Parameters
        ----------
        motion : string
            Name of motion algorithm to use. Accepts "diffuse".

        Returns
        -------
        None.

        """
        if (motion.lower() == "diffuse"):
            self._motion = self._diffuse
            
        
    def _diffuse(self, proximity):
        # Implementation of basic diffusive behavior
        
        # Get unit vector to obstacle
        obj = _calc_centroid(proximity)
        if obj is None:
            return self._agent.vel
        
        # Apply reflection across object vector
        vel = -1*(2 * np.dot(self._agent.vel, obj)/np.dot(obj, obj) * obj - self._agent.vel)
        return vel
    
    
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
