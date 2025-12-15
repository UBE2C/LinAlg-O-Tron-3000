#Think about how to render a rolling triangle or square by utilizing the linalg knowledge I gained so far and using the numpy package!

from numpy._typing._array_like import NDArray

#----- Import modules ------#

import numpy as np
from numpy import linalg
from numpy.typing import NDArray
import copy
import math
import warnings
import os
import time
import termios
import tty
import select
import sys

#----- End section ------#




#----- Define the shape object and it's transformations ------#

class Shape:
    def __init__(self, x_coord_lst: list[int], y_coord_lst: list[int]) -> None:
        self.init_x_coords: tuple[int, ...] = tuple(x_coord_lst) #the initial x coords used for shape initialization
        self.init_y_coords: tuple[int, ...] = tuple(y_coord_lst) #the initial y coords used for shape initialization
        self.init_vertexes: NDArray[np.int64] = np.ndarray(shape = (len(x_coord_lst), 2), dtype = np.int64) #the initial vertexes which will be used for every rotation
        self.x_coords: list[np.int64] = [] #transformed x coords used for printing a transformed shape and grid
        self.y_coords: list[np.int64] = [] #transformed y coords used for printing a transformed shape and grid
        self.vertexes: NDArray[np.int64] = np.ndarray(shape = (len(x_coord_lst), 2), dtype = np.int64) #the transformed vertexes which are used for visualization
        self.sides: dict[int, NDArray[np.int64]] = {}
        self.side_lines: list[NDArray[np.int64]] = []
        self.fill_table: NDArray[np.int64] | None = None
        self.identity_matrix: NDArray[np.int64] = np.array([[1, 0], [0, 1]])
        self.rotation_matrix: NDArray[np.float64] = np.zeros(shape = (2, 2), dtype = np.float64)
        self.scaling_matrix: NDArray[np.float64] = np.zeros(shape = (2, 2), dtype = np.float64)
        self.transform_matrix: NDArray[np.float64] = np.zeros(shape = (2, 2), dtype = np.float64)
        self.rotation: int | float = 0
        self.scale: int | float = 1
        self.scale_direction: int = 0
        


    def normal_round(self, value: float | np.float64) -> np.int64:
        output_int: np.int64 = np.int64(0)

        if value - math.floor(value) < 0.5:
            output_int = np.int64(math.floor(value))
        
        else:
            output_int = np.int64(math.ceil(value))
        
        return output_int


    def initialize_vertex_matrix(self) -> None:
        matrix: NDArray[np.int64] = np.zeros((len(self.init_x_coords), 2), np.int64)

        #Covert both the matrix and the mutable coords to a Braille grid (2x4)
        for i in range(len(matrix)):
            matrix[i][0] = self.init_x_coords[i] * 2
            matrix[i][1] = self.init_y_coords[i] * 4


        #Also initialize the mutable x y coordinates for each vertex
        for i in range(len(self.init_x_coords)):
            self.x_coords.append(np.int64(self.init_x_coords[i] * 2))
            self.y_coords.append(np.int64(self.init_y_coords[i] * 4)) 

        
        self.init_vertexes = matrix
        self.vertexes = matrix

        return None

    
    def define_sides(self, circular: bool = True) -> None:
        n_sides: int = len(self.vertexes)
        sides: dict[int, NDArray[np.int64]] = {}
        
        if circular == True:
            for side in range(n_sides):
                if side == n_sides - 1:
                    sides[side] = np.array([self.vertexes[side], self.vertexes[0]], np.int64)
                
                else:
                    sides[side] = np.array([self.vertexes[side], self.vertexes[side + 1]], np.int64)

        else:
            for side in range(n_sides):
                if side != n_sides - 1:
                    sides[side] = np.array([self.vertexes[side], self.vertexes[side + 1]], np.int64)


        self.sides = sides

        return None


    #Bresenham's Line Algorithm
    def draw_sides(self, vertex_0: NDArray[np.int64], vertex_1: NDArray[np.int64]) -> list[NDArray[np.int64]]:
        new_vect_lst: list[NDArray[np.int64]] = []

        x0: int = vertex_0[0]
        y0: int = vertex_0[1]
        
        x1: int = vertex_1[0]
        y1: int = vertex_1[1]
        
        dx: int = abs(x0 - x1)
        dy: int = abs(y0 - y1)

         # Check if line is steep (more vertical than horizontal)
        steep: bool = dy > dx
        
        # For steep lines, we swap the roles of x and y
        if steep:
            dx, dy = dy, dx  # Swap deltas

        error: int = 2*dy - dx
        
        x: int = x0
        y: int = y0

        step_x: int = 1 if x0 < x1 else -1
        step_y: int = 1 if y0 < y1 else -1

        for _ in range(dx + 1): #this and the first if statement ensures that both vertexes are in the side line array
            new_vector: NDArray[np.int64] = np.zeros((1, 2), np.int64)

            if _ == 0:
                new_vector[0][0] = x
                new_vector[0][1] = y
            
            else:

                if error >= 0:
                    
                    if steep == True:
                        x += step_x
                    else:
                        y += step_y
                    
                    error -= 2*dx
                
                if steep == True:
                    y += step_y
                
                else:
                    x += step_x

                error += 2*dy

                new_vector[0][0] = x
                new_vector[0][1] = y
            
            new_vect_lst.append(new_vector)

        return new_vect_lst


    def connect_vertexes(self) -> None:
        sides: dict[int, NDArray[np.int64]] = copy.deepcopy(x = self.sides)
        output: list[NDArray[np.int64]] = []

        for key in sides.keys():
            side: NDArray[np.int64] = sides[key]
            vector_list: list[NDArray[np.int64]] = self.draw_sides(vertex_0 = side[0], vertex_1 = side[1])

            vector_array: NDArray[np.int64] = np.zeros((len(vector_list), 2), np.int64)
            for i, vector in enumerate(vector_list):
                vector_array[i] = vector

            output.append(vector_array)
            
        self.side_lines = output

        return None
    

    def point_in_polygon(self) -> None:
        #Transfer ownership
        vertexes: NDArray[np.int64] = copy.deepcopy(x = self.vertexes)
        edges: dict[int, NDArray[np.int64]] = copy.deepcopy(x = self.sides)
        sidelines: list[NDArray[np.int64]] = copy.deepcopy(x = self.side_lines)


        #Declare filter list for the output
        points_in_lst: list[NDArray[np.int64]] = []


        #Define the bounding box
        min_x: np.int64 = np.min(vertexes[:,0])
        max_x: np.int64 = np.max(vertexes[:,0])

        min_y: np.int64 = np.min(vertexes[:,1])
        max_y: np.int64 = np.max(vertexes[:,1])

        x_coords: list[int] = [x for x in range(min_x, max_x + 1)]
        y_coords: list[int] = [y for y in range(min_y, max_y + 1)]

        bbox: NDArray[np.int64] = np.array([[x, y] for y in y_coords for x in x_coords ], dtype=np.int64)


        #Calculate the intersections of the cast rays
        for xy in bbox:
            #If a point is in the rendered side lines it does not need to be checked and rendered so jump over it
            jump: bool = False
            
            for line in sidelines:
                if np.any(np.all(xy == line, axis = 1)):
                    jump = True

            if jump == True:
                continue                

            #Declare variables for the ray casting loop
            m: np.float64 = np.float64(0)
            b: np.float64 = np.float64(0)
            n_crossings: int = 0

            for key in edges.keys():
                x_intercept: np.float64 | None = None

                dx: np.int64 = edges[key][0][0] - edges[key][1][0]
                dy: np.int64 = edges[key][0][1] - edges[key][1][1]

                if dx == 0:
                    # vertical line: x_intercept = x1
                    x_intercept = edges[key][0][0]
                
                else:
                    m = np.float64(dy / dx) #m = dY/dX
                    b = edges[key][0][1] - (m * edges[key][0][0]) # b = y - mx

                    if m == 0: #horizontal edges are skipped
                        continue
                    
                    y_low: NDArray[np.int64] = edges[key][:,1][0]
                    y_high: NDArray[np.int64] = edges[key][:,1][1]

                    if np.all((xy[1] >= y_low) & (xy[1] < y_high)) or np.all((xy[1] < y_low) & (xy[1] >= y_high)): #y between the vertexes of the given edge calculate intercept
                        x_intercept: np.float64 = (xy[1] - b) / m

                if x_intercept is not None and x_intercept > xy[0]:
                    n_crossings += 1
                    
                    
            #Check if the number of crossings is odd or even
            if n_crossings % 2 != 0:
                points_in_lst.append(xy)


        #Convert the points in list to an NDarray
        output: NDArray[np.int64] = np.asarray(points_in_lst)

        self.fill_table = output

        return None
            
    
    def draw_shape(self) -> str:
        #Transfer ownership
        vertexes: NDArray[np.int64] = copy.deepcopy(x = self.vertexes)
        sidelines: list[NDArray[np.int64]] = copy.deepcopy(x = self.side_lines)
        
        sides_to_draw: list[NDArray[np.int64]] = []
        
        for v in sidelines:
            sides_to_draw.extend(v)
        
        x_min: np.int64 = np.min(self.x_coords)
        y_min: np.int64 = np.min(self.y_coords)

        x_max: np.int64 = np.max(self.x_coords)
        y_max: np.int64 = np.max(self.y_coords)
        
        ncol: np.int64 = x_max - x_min + 1 #the plus 1 is to be inclusive
        nrow: np.int64 = y_max - y_min + 1
        
        grid: list[list[str]] = [[" " for _ in range(ncol)] for _ in range(nrow)]

        #Correct for the positioning by flipping the y-axis (smallest y in bottom, largest in top) and re-centering along the x-axis
        for vertex in vertexes:
            vertex[0] = vertex[0] - x_min
            vertex[1] = y_max - vertex[1]

        for vector in sides_to_draw:
            vector[0] = vector[0] - x_min
            vector[1] = y_max - vector[1]

        #Fill the grid using the coordinates
        for coord in sides_to_draw:
            grid[coord[1]][coord[0]] = "*"

        for coord in vertexes:
            grid[coord[1]][coord[0]] = "x"

        #Convert the grid into a single list of strings (every row is a single string)
        coord_lst: list[str] = ["".join(_) for _ in grid]
        
        #Convert the list into a unified string
        shape_string: str = "\r\n".join(coord_lst)

        return shape_string

    def calculate_rotation(self, matrix: NDArray[np.int64] | NDArray[np.float64] | None = None, direction: str = "cc", angle: int | float = 3) -> None:
        if matrix is None:
            input_mat: NDArray[np.int64] | NDArray[np.float64] = copy.deepcopy(x = self.identity_matrix)
        
        else:
            input_mat: NDArray[np.int64] | NDArray[np.float64] = copy.deepcopy(x = matrix)
        
        output_mat: NDArray[np.float64] = np.zeros(shape = (2, 2), dtype = np.float64)
        
        #Count with the previous rotation
        base_angle: int | float = copy.copy(self.rotation)

        #Reset the base rotation to zero after a full rotation to any direction
        if base_angle == 360 or base_angle == -360:
            self.rotation = 0
        
        full_tilt_angle: int | float = base_angle + angle

        
        if direction == "cc":
            theta_rad_i: float = math.radians(full_tilt_angle)
            theta_rad_j: float = math.radians(full_tilt_angle)

        elif direction == "c":
            theta_rad_i: float = -(math.radians(full_tilt_angle))
            theta_rad_j: float = -(math.radians(full_tilt_angle))
        
        else:
            raise ValueError(f"calculate_rotation: direction must be either 'c' (clockwise) or 'cc' (counter clockwise), instead {direction} was given.")

        vector_magnitudes: list[np.float64] = []
        for i in range(len(input_mat)):
            vector_magnitudes.append(np.sqrt((input_mat[i][0]) ** 2 + (input_mat[i][1]) ** 2))

        
        #Construct rotation matrix
        """
        [Xi = cos theta * ||v|| Xj = -sin * ||v||] -> -sin due to j^ starting from a 90° angle
        [Yi = sin theta * ||v|| Yj = cos * ||v||]
        """
        output_mat[0][0] = math.cos(theta_rad_i) * vector_magnitudes[0]
        output_mat[0][1] = math.sin(theta_rad_i) * vector_magnitudes[0]

        output_mat[1][0] = -(math.sin(theta_rad_j) * vector_magnitudes[1])
        output_mat[1][1] = math.cos(theta_rad_j) * vector_magnitudes[1]


        self.rotation_matrix = output_mat
        self.rotation = full_tilt_angle

        return None
    

    def calculate_scaling(self, matrix: NDArray[np.int64] | NDArray[np.float64] | None = None, scaler: int | float = 2, scaling_mode: str = "up", max_scale: int | float = 20, min_scale: int | float = 0.1) -> None:
        if matrix is None:
            input_matrix: NDArray[np.int64] | NDArray[np.float64] = copy.deepcopy(x = self.identity_matrix)
            
        else:
            input_matrix: NDArray[np.int64] | NDArray[np.float64] = copy.deepcopy(x = matrix)


        #Cast the input matrix to a float matrix to avoid type collisions
        input_mat: NDArray[np.float64] = np.array(input_matrix, dtype=np.float64)

        base_scale: int | float = self.scale


        #Helper functions 
        def step_up(step: int | float = scaler, base: int | float = base_scale) -> int | float: 
            scaler: int | float = step 
            if scaler < 1:
                    new_scale: int | float = base / scaler

                    if new_scale >= max_scale:
                        full_scaler = max_scale
                        self.scale_direction = -1

                    else:
                        full_scaler = new_scale
                
            else:
                new_scale: int | float = base * scaler

                if new_scale >= max_scale:
                    full_scaler = max_scale
                    self.scale_direction = -1

                else:
                    full_scaler = new_scale

            return full_scaler
        
        def step_down(step: int | float = scaler, base: int | float = base_scale) -> int | float: 
            scaler: int | float = step 
            
            if scaler > 1:
                    new_scale: int | float = base / scaler

                    if new_scale <= min_scale:
                        full_scaler = min_scale
                        self.scale_direction = 1

                    else:
                        full_scaler = new_scale

            else:
                new_scale: int | float = base * scaler

                if new_scale <= min_scale:
                    full_scaler = min_scale
                    self.scale_direction = 1
                
                else:
                    full_scaler = new_scale
            
            return full_scaler


        #Selecting the scaling mode
        if scaling_mode == "pulse":
            if self.scale_direction == 0 and scaler > 1:
                if base_scale >= max_scale:
                    self.scale_direction = -1

                else:
                    self.scale_direction = 1

            elif self.scale_direction == 0 and scaler < 1:
                if base_scale <= min_scale:
                    self.scale_direction = 1

                else:
                    self.scale_direction = -1

            if self.scale_direction == 1:
                full_scaler = step_up()
            
            else:
                full_scaler = step_down()

        elif scaling_mode == "up":
            self.scale_direction = 1
            full_scaler = step_up()

        elif scaling_mode == "down":
            self.scale_direction = -1
            full_scaler = step_down()

        else:
            raise ValueError(f"calculate_scaling: the scaling_mode should be 'up', 'down' or 'pulse', {scaling_mode} was given.")


        #Scalar - matrix multiplication
        output_mat: NDArray[np.float64] | NDArray[np.int64] = full_scaler * input_mat

        #Update shape attributes
        self.scaling_matrix = output_mat
        self.scale = full_scaler

    
    def compose_transform(self, matrix_1: NDArray[np.float64], matrix_2: NDArray[np.float64]) -> None:
        #Transfer ownership
        mat_1: NDArray[np.float64] = copy.deepcopy(matrix_1)
        mat_2: NDArray[np.float64] = copy.deepcopy(matrix_2)

        output_matrix: NDArray[np.float64] = np.matmul(mat_1, mat_2)

        self.transform_matrix = output_matrix

    
    def apply_transform(self, transform: str) -> None:
        """
        Should apply the transformation around the center of the object.
        """
        
        if transform == "rotation":
            transform_matrix: NDArray[np.float64] = copy.deepcopy(self.rotation_matrix)

        elif transform == "scaling":
            transform_matrix: NDArray[np.float64] = copy.deepcopy(self.scaling_matrix)

        elif transform == "composition":
            transform_matrix: NDArray[np.float64] = copy.deepcopy(self.transform_matrix)

        else:
            raise ValueError(f"apply_transform: the chosen transformation has to be 'rotation', 'scaling' or 'composition', {transform} was given.")
        
        vertexes: NDArray[np.int64] = copy.deepcopy(x = self.init_vertexes)

        
        new_vertexes: NDArray[np.int64] = np.ndarray(shape = (len(self.vertexes), 2), dtype = np.int64)
        new_x_coords: list[np.int64] = []
        new_y_coords: list[np.int64] = []
        

        #Get the center points of the shape along both axes
        center_point_x: np.float64 = np.mean(vertexes[:, 0], dtype=np.float64) #[:, 0] or [:, 0] is a super neat np syntax to get the elements of an array
        center_point_y: np.float64 = np.mean(vertexes[:, 1], dtype=np.float64)

        #Center the shape to the origin (0, 0 in R2)
        centered_vertexes: NDArray[np.float64] = vertexes - np.array([center_point_x, center_point_y], dtype = np.float64)

        #Apply the rotation onto the centered vertexes
        centered_vertexes = linalg.matmul(centered_vertexes, transform_matrix).astype(np.float64)

        #Move the vertexes back to the original center
        intermed_vertexes: NDArray[np.float64] = centered_vertexes + np.array([center_point_x, center_point_y], dtype = np.float64)

        for i in range(len(intermed_vertexes)):
            new_vertexes[i][0] = self.normal_round(value = intermed_vertexes[i][0])
            new_vertexes[i][1] = self.normal_round(value = intermed_vertexes[i][1])

            new_x_coords.append(self.normal_round(value = intermed_vertexes[i][0]))
            new_y_coords.append(self.normal_round(value = intermed_vertexes[i][1]))

        self.vertexes = new_vertexes
        self.x_coords = new_x_coords
        self.y_coords = new_y_coords

        return None
    
#----- End section ------#




#----- Rendering ------#

#A helper function to map the coordinates to the appropriate braille grid and pixel
def map_to_braille(vector_list: list[NDArray[np.int64]] | NDArray[np.int64] | None, braille_grid: list[list[int]]) -> list[list[int]]:
    
    if isinstance(vector_list, (list)):
        flat_vector_list: NDArray[np.int64] = np.vstack(vector_list)
    
    elif vector_list is not None:
        flat_vector_list: NDArray[np.int64] = copy.deepcopy(vector_list)

    else:
        raise ValueError(f"map_to_braille: the supplied vector list is of type None")

    grid: list[list[int]] = copy.deepcopy(braille_grid)

    braille_conv_table: dict[int, dict[int, int]] = {
            0: {  # Left column
                0: 0x1,   # dot 1 (top)
                1: 0x2,   # dot 2
                2: 0x4,   # dot 3
                3: 0x40   # dot 7 (bottom)
            },
            1: {  # Right column
                0: 0x8,   # dot 4 (top)
                1: 0x10,  # dot 5
                2: 0x20,  # dot 6
                3: 0x80   # dot 8 (bottom)
            }
        }

    block_x: NDArray[np.int64] = flat_vector_list[:,0] // 2
    block_y: NDArray[np.int64] = flat_vector_list[:,1] // 4

    pixel_col: NDArray[np.int64] = flat_vector_list[:,0] % 2
    pixel_row: NDArray[np.int64] = flat_vector_list[:,1] % 4

    for i in range(len(flat_vector_list)):
        grid[block_y[i]][block_x[i]] |= braille_conv_table[pixel_col[i]][pixel_row[i]]

    return grid


def draw_shape(shape: Shape) -> str:
        #Transfer ownership
        vertexes: NDArray[np.int64] = copy.deepcopy(x = shape.vertexes)
        sidelines: list[NDArray[np.int64]] = copy.deepcopy(x = shape.side_lines)
        filltable: NDArray[np.int64] | None = copy.deepcopy(x = shape.fill_table)
        x_coord: list[np.int64] = copy.deepcopy(shape.x_coords)
        y_coord: list[np.int64] = copy.deepcopy(shape.y_coords)
        

        #I have to build a grid containing braille blocks
        x_min: np.int64 = np.int64(np.floor(np.min(x_coord) / 2))
        y_min: np.int64 = np.int64(np.floor(np.min(y_coord) / 4))

        x_max: np.int64 = np.int64(np.ceil(np.max(x_coord) / 2))
        y_max: np.int64 = np.int64(np.ceil(np.max(y_coord) / 4))

        ncol: np.int64 = x_max - x_min + 1 #the plus 1 is to be inclusive
        nrow: np.int64 = y_max - y_min + 1
        
        grid: list[list[int]] = [[0 for _ in range(ncol)] for _ in range(nrow)]


        #Calculate the min and max braille x and y values to flip along the y and re-center along the x axes
        bx_min: np.int64 = np.min(x_coord)
        #by_min: np.int64 = np.min(y_coord)
        by_max: np.int64 = np.max(y_coord)
        

        #Correct for the positioning by flipping the y-axis (smallest y in bottom, largest in top) and re-centering along the x-axis
        vertexes[:, 0] = vertexes[:, 0] - bx_min
        vertexes[:, 1] = by_max - vertexes[:, 1]
            

        #Correct for the positioning by flipping the y-axis (smallest y in bottom, largest in top) and re-centering along the x-axis
        for vector in sidelines:
            vector[:, 0] = vector[:, 0] - bx_min
            vector[:, 1] = by_max - vector[:, 1]


        #Correct for the positioning by flipping the y-axis (smallest y in bottom, largest in top) and re-centering along the x-axis
        if filltable is not None:
            filltable[:, 0] = filltable[:, 0] - bx_min
            filltable[:, 1] = by_max - filltable[:, 1]


        #Map the coordinates to the appropriate braille grid and pixel
        grid = map_to_braille(vector_list = sidelines, braille_grid = grid)

        if filltable is not None:
            grid = map_to_braille(vector_list = filltable, braille_grid = grid)

        
        printing_grid: list[list[str]] = [["" for _ in range(ncol)] for _ in range(nrow)]
        for col in range(ncol):
            for row in range(nrow):
                printing_grid[row][col] = chr( 0x2800 + grid[row][col])

        
        #Convert the grid into a single list of strings (every row is a single string)
        coord_lst: list[str] = ["".join(_) for _ in printing_grid]
        

        #Convert the list into a unified string
        shape_string: str = "\r\n".join(coord_lst)


        print(shape_string)
        return shape_string


def render_objects(shape_lst: list[Shape]) -> str:
        #Pool all x and y coordinates from all shapes to draw the common grid
        all_x_coords: list[np.int64] = []
        all_y_coords: list[np.int64] = []

        for obj in shape_lst:
            all_x_coords.extend(copy.deepcopy(obj.x_coords))
            all_y_coords.extend(copy.deepcopy(obj.y_coords))

        #I have to build a grid containing braille blocks
        x_min: np.int64 = np.int64(np.floor(np.min(all_x_coords) / 2))
        y_min: np.int64 = np.int64(np.floor(np.min(all_y_coords) / 4))

        x_max: np.int64 = np.int64(np.ceil(np.max(all_x_coords) / 2))
        y_max: np.int64 = np.int64(np.ceil(np.max(all_y_coords) / 4))

        
        ncol: np.int64 = x_max - x_min + 1 #the plus 1 is to be inclusive
        nrow: np.int64 = y_max - y_min + 1
        
        grid: list[list[int]] = [[0 for _ in range(ncol)] for _ in range(nrow)]


        #Calculate the min and max braille x and y values to flip along the y and re-center along the x axes
        bx_min: np.int64 = np.min(all_x_coords)
        #by_min: np.int64 = np.min(y_coord)
        by_max: np.int64 = np.max(all_y_coords)

        
        for obj in shape_lst:
            #Transfer ownership
            vertexes: NDArray[np.int64] = copy.deepcopy(x = obj.vertexes)
            sidelines: list[NDArray[np.int64]] = copy.deepcopy(x = obj.side_lines)
            filltable: NDArray[np.int64] | None = copy.deepcopy(x = obj.fill_table)

            sides_to_draw: list[NDArray[np.int64]] = []
            for v in sidelines:
                sides_to_draw.extend(v)


            #Correct for the positioning by flipping the y-axis (smallest y in bottom, largest in top) and re-centering along the x-axis
            for vertex in vertexes:
                vertex[0] = vertex[0] - bx_min
                #vertex[1] = vertex[1] - by_min
                vertex[1] = by_max - vertex[1]
                

            #Correct for the positioning by flipping the y-axis (smallest y in bottom, largest in top) and re-centering along the x-axis
            for vector in sidelines:
                for x_y in vector:
                    x_y[0] = x_y[0] - bx_min
                    #x_y[1] = x_y[1] - by_min
                    x_y[1] = by_max - x_y[1]

            
            #Correct for the positioning by flipping the y-axis (smallest y in bottom, largest in top) and re-centering along the x-axis
            if filltable is not None:
                filltable[:, 0] = filltable[:, 0] - bx_min
                filltable[:, 1] = by_max - filltable[:, 1]
                    

            #Map the coordinates to the appropriate braille grid and pixel
            grid = map_to_braille(vector_list = sidelines, braille_grid = grid)

            if filltable is not None:
                grid = map_to_braille(vector_list = filltable, braille_grid = grid)
        
        printing_grid: list[list[str]] = [["" for _ in range(ncol)] for _ in range(nrow)]
        for col in range(ncol):
            for row in range(nrow):
                printing_grid[row][col] = chr( 0x2800 + grid[row][col])

        
        #Convert the grid into a single list of strings (every row is a single string)
        coord_lst: list[str] = ["".join(_) for _ in printing_grid]
        

        #Convert the list into a unified string
        shape_string: str = "\r\n".join(coord_lst)


        print(shape_string)
        return shape_string



triangle: Shape = Shape(x_coord_lst = [1, 5, 9], y_coord_lst = [1, 6, 1])

square: Shape = Shape(x_coord_lst = [2, 15, 15, 2], y_coord_lst = [2, 2, 7, 7])
o_shadow_1: Shape= Shape(x_coord_lst = [0, 17, 17, 0], y_coord_lst = [0, 0, 9, 9])
o_shadow_2: Shape= Shape(x_coord_lst = [1, 16, 16, 1], y_coord_lst = [1, 1, 8, 8])
i_shadow_1: Shape= Shape(x_coord_lst = [3, 14, 14, 3], y_coord_lst = [3, 3, 6, 6])
i_shadow_2: Shape= Shape(x_coord_lst = [4, 13, 13, 4], y_coord_lst = [4, 4, 5, 5])

squares: list[Shape] = [o_shadow_1, o_shadow_2, square, i_shadow_2, i_shadow_1]
one_square: list[Shape] = [square]
triangles: list[Shape] = [triangle]


def main(object_list: list[Shape], transform: str | None = None, angle: int = -15, direction: str = "c", 
         scaler: float | int = 1.1, scaling_mode: str = "pulse", min_scale: float | int = 1, max_scale: float | int = 2, 
         fill: bool = False, fill_objects: list[int] | None = None) -> None:
    prompt: str = ""

    objects: list[Shape] = copy.deepcopy(object_list)


    #Get the original terminal settings
    file_descriptor: int = sys.stdin.fileno()
    original_settings = termios.tcgetattr(file_descriptor)

    #Initialize the vertexes and sides for every object
    for i, shp in enumerate(objects):
        shp.initialize_vertex_matrix()
        shp.define_sides()
        shp.connect_vertexes()

        if fill == True and fill_objects is None:
            shp.point_in_polygon()
        
        elif fill == True and fill_objects is not None:
            if i in fill_objects:
                shp.point_in_polygon()

    try:
        tty.setcbreak(file_descriptor)

        iter: int = 0
        while iter < 1000 and prompt != "q":
            if iter == 0:
                
                render_objects(shape_lst = objects)
               
            else:
                
                if transform is None:
                    render_objects(shape_lst = objects)

                else:
                    for i, shp in enumerate(objects):
                        if transform == "rotation":
                            shp.calculate_rotation(angle = angle, direction = direction)
                            shp.apply_transform("rotation")

                        elif transform == "scaling":
                            shp.calculate_scaling(scaler = scaler, scaling_mode = scaling_mode, max_scale = max_scale, min_scale = min_scale)
                            shp.apply_transform("scaling")

                        elif transform == "composition":
                            shp.calculate_rotation(angle = angle, direction = direction)
                            shp.calculate_scaling(scaler = scaler, scaling_mode = scaling_mode, max_scale = max_scale, min_scale = min_scale)
                            shp.compose_transform(shp.rotation_matrix, shp.scaling_matrix)
                            shp.apply_transform("composition")
                        
                        shp.define_sides()
                        shp.connect_vertexes()
                        
                        if fill == True and fill_objects is None:
                            shp.point_in_polygon()
                        
                        elif fill == True and fill_objects is not None:
                            if i in fill_objects:
                                shp.point_in_polygon()

                        
                    render_objects(shape_lst = objects)
            
            time.sleep(0.1)

            if os.name == 'posix':
                #os.system("clear") #slow, causes choppy animation

                sys.stdout.write("\x1b[H\x1b[J") #fast, no choppy animation
                #\x1b[H → move cursor to 0,0
                #\x1b[J → clear from cursor to bottom of screen

            else:
                os.system("cls")
            
            input: list[str] = select.select([sys.stdin], [], [], 0)[0]
            if input:
                prompt: str = sys.stdin.read(1)

            iter += 1

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, original_settings)


if __name__ == "__main__":
    main(object_list = squares, transform = "composition", angle = -15, direction = "c", 
        scaler = 1.1, scaling_mode = "pulse", min_scale = 1, max_scale = 2, fill = True, fill_objects=[2])
    
    



