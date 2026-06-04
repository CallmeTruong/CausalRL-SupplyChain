import numpy as np
from dataclasses import dataclass


@dataclass
class DisruptionState:
    active:           bool
    dtype:            int    # 0=none 1=port 2=supplier 3=surge
    days_remaining:   int
    lead_time_delta:  float  # delay time
    capacity_ratio:   float  # 0–1
    demand_mult:      float  # 1.0 = normal


class DisruptionEngine:

    def __init__(self, mean_inter_arrival=60, seed=None):
        self.mean_inter_arrival = mean_inter_arrival
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._state     = self._make_none()
        self._countdown = self._sample_inter_arrival()

    def step(self) -> DisruptionState:
        if self._state.active:
            remaining = self._state.days_remaining - 1
            if remaining <= 0:
                self._state = self._make_none()
            else:
                self._state = DisruptionState(
                    **{**vars(self._state), "days_remaining": remaining}
                )
        else:
            self._countdown -= 1
            if self._countdown <= 0:
                self._state     = self._sample_disruption()
                self._countdown = self._sample_inter_arrival()

        return self._state

    def as_vector(self) -> np.ndarray:

        s = self._state
        return np.array([
            s.dtype / 3.0,
            s.days_remaining / 30.0,
            s.lead_time_delta / 20.0,
            s.capacity_ratio,
            (s.demand_mult - 1.0) / 2.0,
        ], dtype=np.float32)

    # ---------- private ----------

    def _make_none(self):
        return DisruptionState(False, 0, 0, 0.0, 1.0, 1.0)

    def _sample_inter_arrival(self):
        return max(1, int(self.rng.exponential(self.mean_inter_arrival)))

    def _sample_disruption(self):
        dtype = int(self.rng.integers(1, 4))

        if dtype == 1:   # port closure
            return DisruptionState(True, 1,
                days_remaining  = int(self.rng.integers(5, 16)),
                lead_time_delta = float(self.rng.uniform(10, 20)),
                capacity_ratio  = 1.0,
                demand_mult     = 1.0)

        elif dtype == 2: # supplier failure: capacity down
            return DisruptionState(True, 2,
                days_remaining  = int(self.rng.integers(7, 22)),
                lead_time_delta = float(self.rng.uniform(0, 5)),
                capacity_ratio  = float(self.rng.uniform(0.2, 0.5)),
                demand_mult     = 1.0)

        else:            # demand surge
            return DisruptionState(True, 3,
                days_remaining  = int(self.rng.integers(3, 11)),
                lead_time_delta = 0.0,
                capacity_ratio  = 1.0,
                demand_mult     = float(self.rng.uniform(1.5, 3.0)))