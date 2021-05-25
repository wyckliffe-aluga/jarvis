import os, sys, inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from nets.actor import GaussianPolicy as Actor 
from nets.critic import Concatenate
from memory import Memory
from flow_diagram_maker import Visualizer
from env.column_gym import Column_Gym
from env.config import CONFIG

import tensorflow as tf 
import numpy as np 
import time 
import pickle

physical_devices = tf.config.list_physical_devices('cpu')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.keras.backend.set_floatx('float32')
