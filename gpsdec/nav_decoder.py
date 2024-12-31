import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize

from .storage import EphemerisParameters, RecieverSettings
from .tracking import TrackingComponent


class NavDataDecoder:
    def __init__(
        self,
        cur_settings: RecieverSettings,
        cur_sat_tracker: TrackingComponent,
        skip_ticks_error: int,
    ):
        self.settings = cur_settings
        self.sat_tracker = cur_sat_tracker

        self.start_stable_error_ind = skip_ticks_error
        self.from_I_P: int = None
        self.data_bits: np.array = None

        self.is_preamble_found = False
        self.preamble_index = None
        self.bits_from_first_preamb: np.array = None

        self.bits_per_word = 30
        self.words_per_subframe = 10
        self.bits_per_subframe = self.bits_per_word * self.words_per_subframe

        self.first_frame_TOW = None
        self.current_ephemeris = EphemerisParameters()

    def decode_bits(self, to_show=False):
        sat_tracking_stable = np.sign(
            self.sat_tracker.I_P[self.start_stable_error_ind :]
        ).astype(np.int8)

        for ind in range(1, len(sat_tracking_stable)):
            current_data = sat_tracking_stable[ind]
            prev_data = sat_tracking_stable[ind - 1]
            if current_data * prev_data < 0:
                bit_flip_index = ind
                break

        if bit_flip_index is None:
            raise Exception("No bit flip detected in satellite tracking data.")

        self.from_I_P = (
            self.start_stable_error_ind + +bit_flip_index - bit_flip_index // 20 * 20
        )

        padded_ms_k = 21 - (ind - (ind // 20) * 20)
        padded_tracking = np.pad(
            sat_tracking_stable,
            pad_width=(padded_ms_k, 0),
            mode="constant",
            constant_values=(sat_tracking_stable[0],),
        )

        padded_tracking = padded_tracking[: len(padded_tracking) // 20 * 20]
        reshaped = padded_tracking.reshape(-1, 20)
        self.data_bits = np.sign(np.mean(reshaped, axis=1)).astype(np.int8)

        if to_show:
            to_show_length = 300
            section = self.sat_tracker.I_P[:to_show_length]
            norm_coeff = np.max(np.abs((section)))

            plt.plot(section / norm_coeff, label="BPSK decoded")
            plt.plot(
                np.hstack(
                    (
                        np.zeros(self.from_I_P - padded_ms_k + ind // 20 * 20),
                        np.repeat(self.data_bits[: to_show_length // 20], 20)[
                            bit_flip_index:
                        ],
                    )
                ),
                label="bits",
            )
            plt.xlabel("time")
            plt.ylim([-1.2, 1.5])
            plt.legend(loc="upper right")
            plt.grid()
            plt.show()

    def locate_preamble(self, check_forward_subframes_count):
        preamble_bits = [1, -1, -1, -1, 1, -1, 1, 1]

        # don't use first bit, it has may have unvalid ca_code offset
        correlation_with_preamble = np.correlate(self.data_bits[1:], preamble_bits)
        corr_positions = 1 + np.where(np.abs(correlation_with_preamble) == 8)[0]
        if len(corr_positions) == 0:
            raise Exception("No preamble detected. Try extending or filtering data.")

        # TODO Make more stricter preamble location check, based on bits 00
        bits_per_subframe = 300
        for pos in corr_positions:
            for check_ind in range(check_forward_subframes_count):
                next_preamble = pos + bits_per_subframe * check_ind
                if not np.isin(next_preamble, corr_positions):
                    break
            else:
                self.is_preamble_found = True
                self.preamble_index = pos
                break

        if not self.is_preamble_found:
            raise Exception("Preamble not found, maybe provide bigger recording")

        # cutting preamble, would check the integrity on the second one from prev
        self.bits_from_first_preamb = self.data_bits[self.preamble_index :]

        first_correlation = np.sum(
            preamble_bits * self.bits_from_first_preamb[: len(preamble_bits)]
        )
        if np.sign(first_correlation) < 0:
            self.data_bits *= -1

    def check_validity(self, number_of_words):
        is_valid_data = True
        for word_num in range(1, number_of_words):  # TODO calculate number of words
            is_valid_data *= self.check_word_validity(
                self.bits_from_first_preamb, word_num
            )

        if not is_valid_data:
            raise Exception("Data didn't pass the parity check")

    @staticmethod
    def check_word_validity(data_bits_from_preambule, word_number_to_check):
        next_parity_line = [
            1,
            1,
            1,
            0,
            1,
            1,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            0,
            0,
            1,
            1,
            0,
            1,
            0,
            0,
            1,
            0,
        ]

        matrix_result = next_parity_line
        for _ in range(3):
            next_parity_line = np.roll(next_parity_line, 1)
            matrix_result = np.vstack([matrix_result, next_parity_line])

        parity_matrix = np.vstack(
            [
                matrix_result,
                [
                    1,
                    0,
                    1,
                    0,
                    1,
                    1,
                    1,
                    0,
                    1,
                    1,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                ],
                [
                    0,
                    0,
                    1,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                    1,
                    1,
                    0,
                    1,
                    0,
                    1,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    1,
                    1,
                    1,
                ],
            ]
        )

        D29 = data_bits_from_preambule[(word_number_to_check - 1) * 30 + 28]
        D30 = data_bits_from_preambule[(word_number_to_check - 1) * 30 + 29]
        D = data_bits_from_preambule[
            30 * word_number_to_check : 30 * word_number_to_check + 24
        ] * (-1)
        d = D * ((-1) * D30)
        res_step3 = parity_matrix * d

        res_step3_no_zeros = np.where(res_step3 == 0, 1, res_step3)
        row_products = np.prod(res_step3_no_zeros, axis=1)
        validated_parity = row_products * [D29, D30, D29, D30, D30, D29]

        parity_bits = data_bits_from_preambule[
            30 * word_number_to_check + 24 : 30 * word_number_to_check + 30
        ]
        return np.all(validated_parity == parity_bits)

    @staticmethod
    def from_sym_to_uint(data_sym):
        data_bits = np.copy(data_sym)
        data_bits[np.where(data_bits == -1)[0]] = 0
        return int("".join(data_bits.astype(str)), 2)

    @staticmethod
    def from_sym_to_signed_int(data_sym):
        data_bits = np.copy(data_sym)
        data_bits[np.where(data_bits == -1)[0]] = 0

        n_bits = len(data_bits)
        decimal_value = int("".join(data_bits.astype(str)), 2)

        if data_bits[0] == 1:
            decimal_value -= 1 << n_bits

        return decimal_value

    def decode_subframe(
        self, data_bits_subframe, cur_ephemeris: EphemerisParameters, prev_byte
    ):
        if len(data_bits_subframe) != self.bits_per_subframe:
            raise Exception("Not valid amount of data in subframe")

        data_bits_subframe[:24] *= -1 * prev_byte
        for word in range(1, 10):
            data_bits_subframe[
                word * self.bits_per_word : word * self.bits_per_word + 24
            ] *= (-1 * data_bits_subframe[word * self.bits_per_word - 1])

        HOW_num = 1

        subframe_num = self.from_sym_to_uint(
            data_bits_subframe[
                HOW_num * self.bits_per_word + 19 : HOW_num * self.bits_per_word + 22
            ]
        )
        TOW_message = self.from_sym_to_uint(data_bits_subframe[30:47])

        def str_bytes(data):
            copier = np.copy(data)
            copier[np.where(copier == -1)] = 0
            return "".join(map(str, list(copier)))

        print(f"Found subframe: {subframe_num} with TOW: {TOW_message}")

        if subframe_num == 1:
            cur_ephemeris.week_number = self.from_sym_to_uint(data_bits_subframe[60:70])
            cur_ephemeris.sat_accuracy = self.from_sym_to_uint(
                data_bits_subframe[72:76]
            )
            cur_ephemeris.sat_health = self.from_sym_to_uint(data_bits_subframe[76:82])
            cur_ephemeris.issue_of_data_clock = self.from_sym_to_uint(
                np.hstack((data_bits_subframe[82:84], data_bits_subframe[210:218]))
            )
            cur_ephemeris.T_GD = (
                self.from_sym_to_signed_int(data_bits_subframe[196:204]) * 2**-31
            )
            temp_copy = np.copy(data_bits_subframe[196:204])
            temp_copy[np.where(temp_copy == -1)[0]] = 0
            cur_ephemeris.t_oc = (
                self.from_sym_to_uint(data_bits_subframe[218:234]) * 2**4
            )
            cur_ephemeris.a_f2 = (
                self.from_sym_to_signed_int(data_bits_subframe[240:248]) * 2**-55
            )
            cur_ephemeris.a_f1 = (
                self.from_sym_to_signed_int(data_bits_subframe[248:264]) * 2**-43
            )
            cur_ephemeris.a_f0 = (
                self.from_sym_to_signed_int(data_bits_subframe[270:292]) * 2**-31
            )

        elif subframe_num == 2:
            cur_ephemeris.issue_of_data_emphemeris = self.from_sym_to_uint(
                data_bits_subframe[60:68]
            )
            cur_ephemeris.c_rs = (
                self.from_sym_to_signed_int(data_bits_subframe[68:84]) * 2**-5
            )
            cur_ephemeris.delta_n = (
                self.from_sym_to_signed_int(data_bits_subframe[90:106]) * np.pi * 2**-43
            )
            cur_ephemeris.m_0 = (
                self.from_sym_to_signed_int(
                    np.hstack(
                        (data_bits_subframe[106:114], data_bits_subframe[120:144])
                    )
                )
                * np.pi
                * 2**-31
            )
            cur_ephemeris.c_uc = (
                self.from_sym_to_signed_int(data_bits_subframe[150:166]) * 2**-29
            )
            cur_ephemeris.e_s = (
                self.from_sym_to_uint(
                    np.hstack(
                        (data_bits_subframe[166:174], data_bits_subframe[180:204])
                    )
                )
                * 2**-33
            )
            cur_ephemeris.c_us = (
                self.from_sym_to_signed_int(data_bits_subframe[210:226]) * 2**-29
            )
            cur_ephemeris.sqrt_a_s = (
                self.from_sym_to_uint(
                    np.hstack(
                        (data_bits_subframe[226:234], data_bits_subframe[240:264])
                    )
                )
                * 2**-19
            )
            cur_ephemeris.t_oe = (
                self.from_sym_to_uint(data_bits_subframe[270:286]) * 2**4
            )

        elif subframe_num == 3:
            cur_ephemeris.c_ic = (
                self.from_sym_to_signed_int(data_bits_subframe[60:76]) * 2**-29
            )
            cur_ephemeris.omega_0 = (
                self.from_sym_to_signed_int(
                    np.hstack((data_bits_subframe[76:84], data_bits_subframe[90:114]))
                )
                * np.pi
                * 2**-31
            )
            cur_ephemeris.c_is = (
                self.from_sym_to_signed_int(data_bits_subframe[120:136]) * 2**-29
            )
            cur_ephemeris.i_0 = (
                self.from_sym_to_signed_int(
                    np.hstack(
                        (data_bits_subframe[136:144], data_bits_subframe[150:174])
                    )
                )
                * np.pi
                * 2**-31
            )
            cur_ephemeris.c_rc = (
                self.from_sym_to_signed_int(data_bits_subframe[180:196]) * 2**-5
            )
            cur_ephemeris.omega = (
                self.from_sym_to_signed_int(
                    np.hstack(
                        (data_bits_subframe[196:204], data_bits_subframe[210:234])
                    )
                )
                * np.pi
                * 2**-31
            )
            cur_ephemeris.omega_dot = (
                self.from_sym_to_signed_int(data_bits_subframe[240:264])
                * np.pi
                * 2**-43
            )
            cur_ephemeris.IDOE = self.from_sym_to_uint(data_bits_subframe[270:278])
            cur_ephemeris.i_dot = (
                self.from_sym_to_signed_int(data_bits_subframe[278:292])
                * np.pi
                * 2**-43
            )

        return subframe_num, TOW_message

    def decode_available_subframes(self):
        five_frames = np.copy(self.bits_from_first_preamb[:1500])
        five_frames[np.where(five_frames == -1)] = 0
        five_frames = list(five_frames)

        prev_byte = 1
        for i in range(5):
            cur_num, cur_tow = self.decode_subframe(
                self.bits_from_first_preamb[
                    self.bits_per_subframe * i : self.bits_per_subframe * (i + 1)
                ],
                self.current_ephemeris,
                prev_byte,
            )
            prev_byte = self.bits_from_first_preamb[self.bits_per_subframe * i - 1]

            if cur_num == 1:
                self.first_frame_TOW = (
                    cur_tow  # TODO because it should be without bits decoding
                )
                # TODO make variable bits_from_first_subframe

    def calculate_E(self, at_time):
        """
        Can be called with either gps time for sat position
        Or can be passed sv time for calculation of relativistic time
        """
        GM_gps = 3.986005e14  # TODO check standard
        a_semimajor = self.current_ephemeris.sqrt_a_s**2

        time_since_eph_update = at_time - self.current_ephemeris.t_oe
        n = np.sqrt(GM_gps / a_semimajor**3) + self.current_ephemeris.delta_n
        M = (self.current_ephemeris.m_0 + n * time_since_eph_update) % (2 * np.pi)

        def kepler_equation(E):
            return E - self.current_ephemeris.e_s * np.sin(E) - M

        result = optimize.root(kepler_equation, M)
        if not result.success:
            raise Exception(f"Kepler equation solver failed: {result.message}")

        E = result.x[0] % (2 * np.pi)

        return E  # TODO should i make it strictly positive?

    def calculate_week_gps_time_from_sv(self, truncated_TOW):
        # TODO use correction calculate method?
        seconds_at_subframe_start = truncated_TOW * 6

        time_diff = seconds_at_subframe_start - self.current_ephemeris.t_oc

        poly_correction = (
            self.current_ephemeris.a_f0
            + self.current_ephemeris.a_f1 * time_diff
            + self.current_ephemeris.a_f2 * time_diff**2
        )

        F = -4.442807633e-10
        E_of_TOW = self.calculate_E(seconds_at_subframe_start)
        delta_t_rel = (
            F
            * self.current_ephemeris.e_s
            * self.current_ephemeris.sqrt_a_s
            * np.sin(E_of_TOW)
        )

        delta_sv = poly_correction + delta_t_rel
        delta_sv_l1_ca = (
            delta_sv - self.current_ephemeris.T_GD
        )  # TODO should i do it? 30.3.3.3.1.1.1

        gps_time = seconds_at_subframe_start - delta_sv_l1_ca
        # TODO
        # The value of t must account for beginning or end of week
        # crossovers. That is, if the quantity t - toc is greater than 302,400 seconds, subtract 604,800 seconds from t. If the
        # quantity t - toc is less than -302,400 seconds, add 604,800 seconds to t.

        return gps_time

    def calculate_correction(self):
        seconds_at_subframe_start = (self.first_frame_TOW - 1) * 6
        time_diff = seconds_at_subframe_start - self.current_ephemeris.t_oc

        poly_correction = (
            self.current_ephemeris.a_f0
            + self.current_ephemeris.a_f1 * time_diff
            + self.current_ephemeris.a_f2 * time_diff**2
        )

        F = -4.442807633e-10
        E_of_TOW = self.calculate_E(seconds_at_subframe_start)
        delta_t_rel = (
            F
            * self.current_ephemeris.e_s
            * self.current_ephemeris.sqrt_a_s
            * np.sin(E_of_TOW)
        )

        delta_sv = poly_correction + delta_t_rel
        delta_sv_l1_ca = (
            delta_sv - self.current_ephemeris.T_GD
        )  # TODO should i do it? 30.3.3.3.1.1.1

        return delta_sv_l1_ca

    def __satellite_pos_ECEF(self, at_gps_time):
        time_gps_since_eph_update = at_gps_time - self.current_ephemeris.t_oe
        E = self.calculate_E(at_gps_time)
        a_semimajor = self.current_ephemeris.sqrt_a_s**2

        v = np.arctan2(
            np.sqrt(1 - self.current_ephemeris.e_s**2) * np.sin(E),
            np.cos(E) - self.current_ephemeris.e_s,
        )
        phi_old = v + self.current_ephemeris.omega
        r_old = a_semimajor * (1 - self.current_ephemeris.e_s * np.cos(E))

        phi = (
            phi_old
            + self.current_ephemeris.c_us * np.sin(2 * phi_old)
            + self.current_ephemeris.c_uc * np.cos(2 * phi_old)
        )
        r = (
            r_old
            + self.current_ephemeris.c_rs * np.sin(2 * phi_old)
            + self.current_ephemeris.c_rc * np.cos(2 * phi_old)
        )
        i = (
            self.current_ephemeris.i_0
            + self.current_ephemeris.i_dot * time_gps_since_eph_update
            + self.current_ephemeris.c_is * np.sin(2 * phi_old)
            + self.current_ephemeris.c_ic * np.cos(2 * phi_old)
        )

        Omega_dot_earth = 7.2921151467e-5
        omega_er = (
            self.current_ephemeris.omega_0
            + self.current_ephemeris.omega_dot * time_gps_since_eph_update
            - Omega_dot_earth * at_gps_time
        )

        r_sat_ECEF = np.array(
            [
                r * np.cos(omega_er) * np.cos(phi)
                - r * np.sin(omega_er) * np.cos(i) * np.sin(phi),
                r * np.sin(omega_er) * np.cos(phi)
                + r * np.cos(omega_er) * np.cos(i) * np.sin(phi),
                r * np.sin(i) * np.sin(phi),
            ]
        )

        return r_sat_ECEF

    def get_sat_pos_ECEF_at_first_subframe(self):
        real_first_frame_TOW = self.first_frame_TOW - 1
        gps_time = self.calculate_week_gps_time_from_sv(real_first_frame_TOW)
        sat_pos_ECEF = self.__satellite_pos_ECEF(gps_time)

        return sat_pos_ECEF

    def get_travel_time_first_subframe(self):
        # TODO probably need 20 * (self.preamble_index - 1), since relative, doesn't matter
        ms_to_first_subframe = 20 * self.preamble_index
        index_from_rec_start = self.from_I_P + ms_to_first_subframe

        samples_per_code = self.settings.sampling_frequency / (
            self.settings.code_frequency_basis / self.settings.code_length
        )
        travel_time = (
            self.sat_tracker.absolute_sample[index_from_rec_start] / samples_per_code
        )

        return travel_time
