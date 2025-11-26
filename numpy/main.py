from pattern import Checker
import numpy as np

# Example:
checker = Checker(tile_size=20, resolution=(200, 200))
board = checker.draw()

checker.show()