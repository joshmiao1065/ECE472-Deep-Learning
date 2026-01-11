from dataclasses import dataclass, InitVar, field
import numpy as np
from numpy import pi
from numpy.random import Generator

@dataclass
class Data:
    """generate spiral data"""
    rng: InitVar[Generator]
    num_points: int
    num_turns: float #number of convolutions in spiral
    scale: float
    noise: float
    x: np.ndarray = field(init=False)
    target: np.ndarray = field(init=False)

    def __post_init__(self, rng):

        #bitgenerator or smth, as required by you
        #rng = Generator(PCG64(seed=472))

        #angular sweep, 2pi represents one full convolution
        theta = np.sqrt(rng.random(self.num_points)) * self.num_turns * pi

        #spiral tightness and offset
        r_a = self.scale * theta
        #spiral b will just mirror spiral a across origin
        r_b = -r_a

        #generate cartesian data
        data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
        data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
        #noise using rng 
        x_a = data_a + rng.normal(0, self.noise, (self.num_points, 2))
        x_b = data_b + rng.normal(0, self.noise, (self.num_points, 2))

        self.x = np.vstack((x_a, x_b))
        self.target = np.concatenate((np.zeros(self.num_points), np.ones(self.num_points)))

    def get_batch(self, rng: Generator, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        indices = rng.choice(len(self.x), size=batch_size, replace=False)
        return self.x[indices], self.target[indices]