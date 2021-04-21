import numpy as np
import torch

print(''' This script should load the checkpoint file, extract the 1st, 2nd, and perhaps 3rd level generators and discriminators, and generates output data to the ./data/ directory. It then loads the respective numpy/tensor objects in that directory and converts them all to text, saving the text files back in the same directory. ''')
