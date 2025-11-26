import numpy as np
import matplotlib.pyplot as plt

class Checker:
    tile_size = None
    output = None

    resolution = ()

    def __init__(self, resolution, tile_size):
        self.tile_size = tile_size
        self.resolution = resolution

    def draw(self):
        # creates checkerboard pattern in numpy array 
        # the tile on the top left corner is black
        # resolution values should be dividable with 2 * tile_size
        # store the pattern in instance variable "output" and return a copy. 
        t = self.tile_size
        r = self.resolution
        if r % (t * 2) != 0:
            raise ValueError("Resolution values must be divisible with tile size") 
         
        x = y = np.arange(r) // t
        x, y = np.meshgrid(x, y)

        self.output = ( x + y ) % 2

        return self.output.copy()
    
    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.axis("off")
        plt.show()

class Circle:
    output = None

    def __init__(self, resolution, radius, position):
        """
        resolution: int (image is resolution x resolution)
        radius: int (radius of the circle)
        position: tuple (cx, cy) center coordinates
        """
        self.resolution = resolution
        self.radius = radius
        self.position = position  # (cx, cy)

    def draw(self):
        h = w = self.resolution
        cx, cy = self.position
        r = self.radius

        x, y = np.meshgrid(np.arange(h), np.arange(w))  

        self.output = (((x - cx)**2 + (y - cy)**2) <= r**2).astype(float)

        return self.output.copy()

    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.axis("off")
        plt.show()

class Spectrum:
    resolution = None
    output = None
    def __init__(self, resolution):
        self.resolution = resolution
    
    def draw(self):

        res = self.resolution
        R = np.linspace(0, 1, res)
        G = np.linspace(0, 1, res)

        R, G = np.meshgrid(R, G)

        B = 1 - R

        self.output = np.dstack([R,G,B])

        return self.output.copy()
    def show(self):
        plt.imshow(self.output, cmap="gray")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    c = Circle(resolution=300, radius=30, position=(50,50))
    c.draw()
    c.show()
