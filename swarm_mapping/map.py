# -*- coding: utf-8 -*-
"""
Procedural map generation for swarm exploration and mapping. Walls and 
hazardous regions are randomly placed throughout.
Created on Mon Apr 19 21:32:39 2021

@author: John Dong
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import deque

# Grid values
_EMPTY = 0
_WALL = 1
_HAZARD = -1

# Colors for rendering
_EMPTY_COLOR = [1,1,1]
_WALL_COLOR = [0,0,0]
_HAZARD_COLOR = [1,0,0]

# Grid connectivity
_CONNECTED = 4

class Map:
    def __init__(self, width, height, space_fill=0.4, hazard_fill=0.2, fast=False):
        """
        Parameters
        ----------
        width : int
            Width of world in pixels.
        height : int
            Height of world in pixels.
        space_fill : float, optional
            Fraction of world to fill with open space (the rest is wall). 
            The default is 0.4.
        hazard_fill : float, optional
            Fraction of open space to fill with hazards.
            The default is 0.2.
        fast : bool, optional
            Make map generation faster, but regions will look a little less 
            natural.
            The default is False.

        Returns
        -------
        None.

        """
        self.width = 100
        self.height = 100
        self.space_fill = space_fill
        self.hazard_fill = hazard_fill
        if fast:
            self._grow_region = self._grow_region2
        else:
            self._grow_region = self._grow_region1
        
        # Initialize grid
        self.grid = np.zeros((100, 100), float)
        if space_fill < 1:
            self.grid.fill(_WALL)

            # Generate empty spaces
            self._generate_rooms()
            self._smooth_rooms()
        
        # Generate hazards
        self._generate_hazards()
        self._smooth_hazards()
        
        # Make sure regions are connected
        self._connect_rooms()
        self._expand_rooms()
        
        # Remove tiny hazards
        self._fix_hazards()
        
        # Scale to desired size
        self._scale_map(width, height)
        
        
    def render(self):
        """
        Renders the current state of the world as a RGB image numpy array.

        Returns
        -------
        image : numpy.ndarray
            (Height x Width x 3) RGB image of the world.

        """
        empty_mask = self.grid == _EMPTY
        wall_mask = self.grid == _WALL
        image = self.grid.copy()
        image[empty_mask] = _WALL
        image[wall_mask] = _EMPTY
        image = np.expand_dims(image, axis=2)
        image = np.where(image == _EMPTY, _WALL_COLOR, image)
        image = np.where(image == _WALL, _EMPTY_COLOR, image)
        image = np.where(image == _HAZARD, _HAZARD_COLOR, image)
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
            
        
    def _generate_rooms(self):
        # Generate room in center for agents starting point
        x = round(self.width/2)
        y = round(self.height/2)
        size = round(self.width/8)
        self._grow_region((x, y), size - 2, size + 2, _EMPTY, [_EMPTY])
        
        # Keep adding rooms until fill parameter is achieved
        area = self.width * self.height
        fill = np.sum(_WALL - self.grid) / area
        while fill < self.space_fill:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            val = self.grid[y, x]
            if val == _EMPTY: continue
            
            size = random.randint(2, 12)
            self._grow_region((x, y), size - 2, size + 2, _EMPTY, [_EMPTY])
            fill = np.sum(_WALL - self.grid) / area
            
    
    def _generate_hazards(self):
        # Keep adding hazards until fill parameter is achieved
        fill = 0
        while fill < self.hazard_fill:
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            
            # Make sure hazard is in empty space
            if not self.grid[y, x] == _EMPTY:
                continue
            
            # Make sure hazard is not in starting zone
            x0 = round(self.width/2)
            y0 = round(self.height/2)
            dist = np.sqrt(pow(x - x0, 2) + pow(y - y0, 2))
            if dist < self.width/8:
                continue
            
            # Add hazard of random size and location
            size = random.randint(2, 5)
            self._grow_region((x, y), size - 2, size + 2, _HAZARD, [_WALL, _HAZARD])
            
            # Update hazard fill value
            empty_area = np.sum(_WALL - self.grid)
            hazard_area = np.sum((-1*self.grid)[(-1*self.grid) > 0.0])
            fill = hazard_area / empty_area
            
            
    def _smooth_rooms(self):
        kernel = np.ones((3,3),np.uint8)
        self.grid = cv2.morphologyEx(self.grid, cv2.MORPH_OPEN, kernel, iterations=2)
        
    
    def _smooth_hazards(self):
        kernel = np.ones((3,3),np.uint8)
        hazards_only = np.clip(-1*self.grid, 0, 1)
        smoothed = cv2.morphologyEx(hazards_only, cv2.MORPH_CLOSE, kernel, iterations=1)
        self.grid = np.clip(-1*smoothed + self.grid, _HAZARD, _WALL)
        
    
    def _fix_hazards(self):
        kernel = np.ones((3,3),np.uint8)
        no_hazards = np.clip(self.grid, 0, 1)
        hazards_only = np.clip(-1*self.grid, 0, 1)
        smoothed = cv2.morphologyEx(hazards_only, cv2.MORPH_OPEN, kernel, iterations=1)
        self.grid = -1*smoothed + no_hazards
    
    
    def _expand_rooms(self):
        kernel = np.ones((3,3),np.uint8)
        expanded = cv2.erode(np.abs(self.grid), kernel, iterations=1)
        self.grid = np.multiply(self.grid, expanded)
        
        
    def _connect_rooms(self):
        # Get grid where empty space = 1 and everything else is 0
        im = (_WALL - np.abs(self.grid)).astype(np.uint8)
        
        # Find contours to get separated regions
        contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate centroids of regions, and only keep contours that are
        # of significant size
        centroids = []
        filtered_contours = []
        for i, contour in enumerate(contours):
            M = cv2.moments(contour)
            if (M['m00'] == 0): continue
            x = int(M['m10']/M['m00'])
            y = int(M['m01']/M['m00'])
            centroids.append((x,y))
            filtered_contours.append(contour)
        contours = filtered_contours

        # Keep connecting regions until there is only one region
        while (len(contours) > 1):
            
            # Go through each region
            connections = set()
            for i, contour in enumerate(contours):
                
                # Find the next closest region
                min_dist = 2000
                for j, contour2 in enumerate(contours):
                    connection = (min(i, j), max(i, j))
                    if i == j or connection in connections:
                        continue

                    # Get their centroid coordinates
                    x1, y1 = centroids[i][0], centroids[i][1]
                    x2, y2 = centroids[j][0], centroids[j][1]
                    
                    # Calculate their two closest pixels (roughly)
                    p1 = min(contour, key=lambda x: 
                             np.sqrt(pow(x2 - x[0][0], 2) + pow(y2 - x[0][1], 2)))[0]
                    p2 = min(contour2, key=lambda x: 
                             np.sqrt(pow(x1 - x[0][0], 2) + pow(y1 - x[0][1], 2)))[0]

                    # See if this distance is smaller than that of other regions
                    dist = np.sqrt(pow(p2[0] - p1[0], 2) + pow(p2[1] - p1[1], 2))
                    if dist < min_dist:
                        min_dist = dist
                        min_connection = connection
                        min_p1 = p1
                        min_p2 = p2

                # Draw a line between the two closest pixels of the two regions
                connections.add(min_connection)
                cv2.line(self.grid, 
                         (int(min_p1[0]), int(min_p1[1])), (int(min_p2[0]), int(min_p2[1])), 
                         _EMPTY, 4)
            
            # Update new regions and their centroids
            im = (_WALL - np.abs(self.grid)).astype(np.uint8)
            contours, hierarchy = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            centroids = []
            filtered_contours = []
            for i, contour in enumerate(contours):
                M = cv2.moments(contour)
                if (M['m00'] == 0): continue
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])
                centroids.append((x,y))
                filtered_contours.append(contour)
            contours = filtered_contours
            
            
    def _scale_map(self, width, height):
        self.grid = cv2.resize(self.grid, (height, width), interpolation=cv2.INTER_NEAREST)
        self.width = width
        self.height = height
        
    
    def _grow_region1(self, start, min_r ,max_r, val, block):
        # Calculate sigmoid probability distribution parameters
        p0, pf = 0.9, 0.1
        l0 = np.log(p0/(1 - p0))
        lf = np.log(pf/(1 - pf))
        B = (l0*max_r - lf*min_r)/(l0 - lf)
        A = l0/(B - min_r)

        # Start region growing
        self.grid[start[1], start[0]] = val
        neighbors = self._get_neighbors(start)
        queue = deque(neighbors)
        visited = set()
        visited.add(start)
        while queue:
            node = queue.popleft()
            if node in visited: continue
            visited.add(node)

            # Check if node should be added
            dist = self._get_distance(start, node)
            p_thresh = 1.0 - 1.0 / (1 + np.exp(A*(B - dist)))
            p = random.random()
            if p < p_thresh:
                self.grid[node[1], node[0]] = val
                neighbors = self._get_neighbors(node)
                for neighbor in neighbors:
                    if neighbor not in visited\
                    and neighbor not in queue\
                    and not self.grid[neighbor[1], neighbor[0]] in block:
                        queue.append(neighbor)
                        
                        
    def _grow_region2(self, start, min_r, max_r, val, block):
        # Faster region generation using circles
        r = random.randint(min_r, max_r)
        cv2.circle(self.grid, start, r, val, -1)
                        
                        
    def _get_neighbors(self, pos):
        neighbors = []
        x, y = pos
        w = self.width
        h = self.height
        
        if y > 0: # North
            neighbors.append((x, y - 1))
        if y < h - 1: # South
            neighbors.append((x, y + 1))
        if x > 0: # West
            neighbors.append((x - 1, y))
        if x < w - 1: # East
            neighbors.append((x + 1, y))
        
        if _CONNECTED == 8:
            if x > 0 and y > 0: # North West
                neighbors.append((x - 1, y - 1))
            if x > 0 and y < h - 1: # South West
                neighbors.append((x - 1, y + 1))
            if x < w - 1 and y > 0: # North East
                neighbors.append((x + 1, y - 1))
            if x < w - 1 and y < h - 1: # South East
                neighbors.append((x + 1, y + 1))
        return neighbors
    
    
    def _get_distance(self, node_A, node_B):
        return np.sqrt(pow(node_A[0] - node_B[0], 2) + pow(node_A[1] - node_B[1], 2))
            
        
    def __getitem__(self, key):
        return self.grid[key[1]][key[0]]