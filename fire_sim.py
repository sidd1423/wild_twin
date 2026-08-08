import numpy as np

UNBURNED = 0
BURNING = 1
BURNED = 2


class FireSimulation:
    def __init__(self, risk, fuel, dryness, density, canopy, slope, aspect, wind_speed, wind_direction, seed_row, seed_col):
        self.risk = np.nan_to_num(risk, nan=0.0)
        self.fuel = np.nan_to_num(fuel, nan=0.0)
        self.dryness = np.nan_to_num(dryness, nan=0.0)
        self.density = np.nan_to_num(density, nan=0.0)
        self.canopy = np.nan_to_num(canopy, nan=0.0)
        self.slope = np.nan_to_num(slope, nan=0.0)
        self.aspect = np.nan_to_num(aspect, nan=0.0)
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction

        self.rows, self.cols = risk.shape
        self.state = np.zeros((self.rows, self.cols), dtype=np.uint8)

        self.burn_time = np.zeros((self.rows, self.cols),dtype=np.float32)

        if 0 <= seed_row < self.rows and 0 <= seed_col < self.cols:
            self.state[seed_row, seed_col] = BURNING

    # =====================================
    # WIND EFFECT
    # =====================================

    def wind_effect(self, row, col, neighbor_row, neighbor_col):
        dy = neighbor_row - row
        dx = neighbor_col - col

        direction = (np.degrees(np.arctan2(dx, -dy)) + 360) % 360
        difference = abs(direction - self.wind_direction)
        difference = min(difference,
360 - difference
        )

        alignment = np.cos(
            np.radians(difference)
        )

        # Convert wind speed to 0-1

        speed_factor = np.clip(
            self.wind_speed / 30.0,
            0,
            1
        )

        return np.clip(
            1.0 + alignment * speed_factor,
            0.2,
            2.0
        )

    # =====================================
    # SLOPE EFFECT
    # =====================================

    def slope_effect(
        self,
        row,
        col,
        neighbor_row,
        neighbor_col
    ):

        elevation_direction = (
            self.aspect[row, col]
        )

        dy = neighbor_row - row
        dx = neighbor_col - col

        direction = (
            np.degrees(
                np.arctan2(dx, -dy)
            )
            + 360
        ) % 360

        difference = abs(
            direction -
            elevation_direction
        )

        difference = min(
            difference,
            360 - difference
        )

        uphill_alignment = np.cos(
            np.radians(difference)
        )

        slope_strength = np.clip(
            self.slope[row, col] / 45.0,
            0,
            1
        )

        return np.clip(
            1.0 +
            uphill_alignment *
            slope_strength,
            0.3,
            2.0
        )

    # =====================================
    # IGNITION PROBABILITY
    # =====================================

    def ignition_probability(
        self,
        row,
        col,
        neighbor_row,
        neighbor_col
    ):

        fuel = self.fuel[
            neighbor_row,
            neighbor_col
        ]

        dryness = self.dryness[
            neighbor_row,
            neighbor_col
        ]

        density = self.density[
            neighbor_row,
            neighbor_col
        ]

        canopy = self.canopy[
            neighbor_row,
            neighbor_col
        ]

        risk = self.risk[
            neighbor_row,
            neighbor_col
        ]

        wind = self.wind_effect(
            row,
            col,
            neighbor_row,
            neighbor_col
        )

        slope = self.slope_effect(
            row,
            col,
            neighbor_row,
            neighbor_col
        )

        # Fuel availability

        fuel_factor = (
            0.45 * fuel +
            0.25 * density +
            0.30 * canopy
        )

        # Environmental condition

        environment = (
            0.45 * dryness +
            0.55 * risk
        )

        probability = (
            fuel_factor *
            environment *
            wind *
            slope
        )

        return np.clip(
            probability,
            0,
            1
        )

    # =====================================
    # ONE SIMULATION STEP
    # =====================================

    def step(self):

        burning_cells = np.argwhere(
            self.state == BURNING
        )

        newly_burning = []

        for row, col in burning_cells:

            # 8 neighboring cells

            for dr in (-1, 0, 1):

                for dc in (-1, 0, 1):

                    if dr == 0 and dc == 0:
                        continue

                    nr = row + dr
                    nc = col + dc

                    # Outside map

                    if nr < 0 or nr >= self.rows:
                        continue

                    if nc < 0 or nc >= self.cols:
                        continue

                    # Already burning/burned

                    if self.state[nr, nc] != UNBURNED:
                        continue

                    probability = self.ignition_probability(
                        row,
                        col,
                        nr,
                        nc
                    )

                    # Small timestep probability

                    probability *= 0.15

                    if np.random.random() < probability:

                        newly_burning.append(
                            (nr, nc)
                        )

            self.burn_time[
                row,
                col
            ] += 1

        # Ignite new cells

        for row, col in newly_burning:

            self.state[
                row,
                col
            ] = BURNING

        # Cells eventually burn out

        burn_duration = 8

        finished = (
            (self.state == BURNING)
            &
            (self.burn_time >= burn_duration)
        )

        self.state[finished] = BURNED

        return self.state

    # =====================================
    # RUN
    # =====================================

    def run(self, steps):

        history = []

        for _ in range(steps):

            state = self.step()

            history.append(
                state.copy()
            )

        return history