import numpy as np
from scipy.optimize import least_squares

from .nav_decoder import NavDataDecoder


class PseudorangeSolver:
    def __init__(self, navigation_components: list[NavDataDecoder]):
        self.nav_components = navigation_components
        self.speed_of_light = 299_792_458

        self.sat_positions = []
        self.travel_times = []
        self.corrections = []

        self.coordinate_iterations = []

    def extract_data(self):
        for nav_decoder in self.nav_components:
            sat_pos_ECEF = nav_decoder.get_sat_pos_ECEF_at_first_subframe()
            travel_time = nav_decoder.get_travel_time_first_subframe()
            clock_correction = nav_decoder.calculate_correction()

            self.sat_positions.append(sat_pos_ECEF)
            self.travel_times.append(travel_time)
            self.corrections.append(clock_correction)

        self.sat_positions = np.array(self.sat_positions)
        self.travel_times = np.array(self.travel_times)
        self.corrections = np.array(self.corrections)

    def solve(self):
        minimal_time = np.min(self.travel_times)
        relative_pseudoranges = (
            (self.travel_times - minimal_time) / 1_000 * self.speed_of_light
        )
        pseudoranges = relative_pseudoranges + self.corrections * self.speed_of_light

        def residuals(x):
            user_position = x[:3]
            clock_bias = x[3]

            self.coordinate_iterations.append(self.ecef_to_wgs84(*user_position)[:2])

            corrected_sat_positions = np.copy(self.sat_positions)
            for sat_i in range(len(corrected_sat_positions)):
                Omega_dot_earth = 7.2921151467e-5
                angle_derotation = Omega_dot_earth * (clock_bias / self.speed_of_light)
                Rotation = np.array(
                    [
                        [np.cos(angle_derotation), np.sin(angle_derotation), 0],
                        [-np.sin(angle_derotation), np.cos(angle_derotation), 0],
                        [0, 0, 1],
                    ]
                )
                corrected_sat_positions[sat_i] = Rotation @ self.sat_positions[sat_i]

            analytical_pseudoranges = (
                np.sqrt(np.sum((corrected_sat_positions - user_position) ** 2, axis=1))
                + clock_bias
            )

            return analytical_pseudoranges - pseudoranges

        initial_guess = np.zeros(4)
        result = least_squares(residuals, initial_guess)

        estimated_position = result.x[:3]
        estimated_clock_bias = result.x[3]

        self.coordinate_iterations.append(self.ecef_to_wgs84(*estimated_position)[:2])
        estimated_clock_bias_seconds = estimated_clock_bias / self.speed_of_light

        return estimated_position, estimated_clock_bias_seconds

    @staticmethod
    def ecef_to_wgs84(x, y, z):
        a = 6378137.0
        f = 1 / 298.257223563
        b = a * (1 - f)
        e2 = (a**2 - b**2) / a**2

        lon = np.arctan2(y, x)

        p = np.sqrt(x**2 + y**2)
        phi = np.arctan2(z, p * (1 - f))
        for _ in range(20):
            N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
            h = p / np.cos(phi) - N
            phi = np.arctan2(z + e2 * N * np.sin(phi), p)

        N = a / np.sqrt(1 - e2 * np.sin(phi) ** 2)
        h = p / np.cos(phi) - N

        lat = np.degrees(phi)
        lon = np.degrees(lon)

        return lat, lon, h
