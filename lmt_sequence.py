from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pulse:
    k: int
    detuning_hz: float
    phi: float
    label: str
    rabi_frequency: float
    duration: float

    def __post_init__(self):
        if self.k not in (-1, +1):
            raise ValueError("Pulse k must be either +1 or -1")
        if self.rabi_frequency <= 0.0:
            raise ValueError("Pulse rabi_frequency must be positive")
        if self.duration < 0.0:
            raise ValueError("Pulse duration must be non-negative")

    @property
    def pulse_area(self):
        return self.duration * 2 * np.pi * self.rabi_frequency


@dataclass(frozen=True)
class Clearout:
    duration: float
    label: str = "clearout"

    def __post_init__(self):
        if self.duration < 0.0:
            raise ValueError("Clearout duration must be non-negative")


@dataclass(frozen=True)
class Freefall:
    duration: float
    label: str = "freefall"

    def __post_init__(self):
        if self.duration < 0.0:
            raise ValueError("Freefall duration must be non-negative")
