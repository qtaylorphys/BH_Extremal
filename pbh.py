from abc import ABC

from numbers import Real

from predict_size import predict_size
from temperature import Hawking_temperature

class PrimordialBlackHole(ABC):
    def __init__(
        self,
        spacetime: str,
        M_init: Real,
        M_final: Real = 1,
        J_init: Real = 0,
        eps: Real = 1,
    ):
        self.spacetime = spacetime
        self.M_init = M_init
        self.M_final = M_final
        self.J_init = J_init
        self.eps = eps
        
        self.N = predict_size(self.M_init)
        self.temperature_f = Hawking_temperature(self.spacetime)

    def evolve(self):
        