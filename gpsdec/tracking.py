import numpy as np
from tqdm import tqdm

from .prn_generator import generate_prn
from .storage import AcquisitionInfo, RecieverSettings


class TrackingComponent:
    def __init__(
        self,
        baseband_signal,
        acquisition_info: AcquisitionInfo,
        settings: RecieverSettings,
    ):
        self.signal_iq_data = baseband_signal

        self.acquisition_info = acquisition_info
        self.settings = settings

        settings.code_periods_to_process

        # TODO thinkg about it
        self.absolute_sample = np.zeros(settings.code_periods_to_process)

        self.I_P = np.zeros(settings.code_periods_to_process)
        self.Q_P = np.zeros(settings.code_periods_to_process)
        self.early_metric = np.zeros(settings.code_periods_to_process)
        self.late_metric = np.zeros(settings.code_periods_to_process)

        self.DLL_code_frequency = np.full(settings.code_periods_to_process, np.inf)
        self.DLL_code_error = np.full(settings.code_periods_to_process, np.inf)
        self.DLL_vco_value = np.full(settings.code_periods_to_process, np.inf)

        self.PLL_carrier_frequency = np.full(settings.code_periods_to_process, np.inf)
        self.PLL_phase_errors = np.full(settings.code_periods_to_process, np.inf)
        self.PLL_vco_value = np.full(settings.code_periods_to_process, np.inf)

        self.code_ca_period = 0.001

        self.pll_tau1, self.pll_tau2 = self.locked_loop_coefficients(
            settings.PLL_noise_bandwidth, settings.PLL_damping_ration, 0.25
        )
        self.dll_tau1, self.dll_tau2 = self.locked_loop_coefficients(
            settings.PLL_noise_bandwidth, settings.PLL_damping_ration, 1.0
        )

        self.early_late_spacing = settings.DLL_correlation_spacing

        self.already_read_points = acquisition_info.code_phase - 1

        self.residual_carrier_phase = 0.0
        self.current_carrier_frequency = self.acquisition_info.acquired_frequency
        self.prev_carrier_phase_error = 0.0
        self.prev_carrier_VCO = 0.0
        self.prev_carrier_phase_mismatch = 0.0

        self.residual_ca_code_chips = 0.0
        self.current_ca_code_frequency = settings.code_frequency_basis
        self.prev_ca_code_VCO = 0.0
        self.prev_ca_code_error = 0.0

        self.code_phase_step = (
            self.settings.code_frequency_basis / self.settings.sampling_frequency
        )

        # TODO create generate prn class
        ca_code = generate_prn(self.acquisition_info.satellite_prn_number)
        self.ca_code_bits = np.concatenate([[ca_code[-1]], ca_code, [ca_code[0]]])

    def track(self):
        # TODO if already passed tracking, exception for reinitialization
        # TODO fid.seek(settings.skip_number_of_bytes + channel[channelNr]['codePhase'] - 1), 0)

        # TODO it's not true ms, this are code periods with varying time length
        for iters_ind in tqdm(range(self.settings.code_periods_to_process)):
            self.code_phase_step = (
                self.current_ca_code_frequency / self.settings.sampling_frequency
            )
            chunk_length = int(
                np.ceil(
                    (self.settings.code_length - self.residual_ca_code_chips)
                    / self.code_phase_step
                )
            )

            # TODO maybe make file reading instead of inmemory for big file processing
            signal_chunk = self.signal_iq_data[
                self.already_read_points : self.already_read_points + chunk_length
            ]
            self.already_read_points += chunk_length

            if len(signal_chunk) != chunk_length:
                # TODO return result bits I_Q
                print("Not enough samples for tracking, exiting!")
                return

            # TODO maybe implement 1 byte precision (removing DC component, but before convert to float/int array)
            # if settings['dataType'] == 'uchar': rawSignal = rawSignal - (127.0 + 127.0j)

            tcode = np.arange(
                self.residual_ca_code_chips,
                chunk_length * self.code_phase_step + self.residual_ca_code_chips,
                self.code_phase_step,
            )
            prompt_ca_code = self.ca_code_bits[np.ceil(tcode).astype(int)]

            time = np.arange(chunk_length) / self.settings.sampling_frequency
            carrier_signal_phases = self.residual_carrier_phase + (
                self.current_carrier_frequency * 2.0 * np.pi * time
            )
            carrier_signal = np.exp(1j * carrier_signal_phases)

            removed_carrier_signal = carrier_signal * signal_chunk
            self.__pll_carrier_tracking_iteration(
                removed_carrier_signal, prompt_ca_code, iters_ind
            )
            self.__dll_code_tracking_iteration(
                removed_carrier_signal, chunk_length, iters_ind
            )

            self.residual_carrier_phase = carrier_signal_phases[-1] % (2 * np.pi)
            self.residual_ca_code_chips = tcode[-1] + self.code_phase_step - 1023.0

            self.absolute_sample[iters_ind] = (
                self.already_read_points
                - self.residual_ca_code_chips / self.code_phase_step
            )

        self.finished_tracking = True

    def __pll_carrier_tracking_iteration(
        self, baseband_signal, prompt_ca_code, current_sample_ind
    ):
        qBasebandSignal = np.real(baseband_signal)
        iBasebandSignal = np.imag(baseband_signal)

        # we need to remove prompt_ca_code for summuation to work
        I_P = np.sum(prompt_ca_code * iBasebandSignal)
        Q_P = np.sum(prompt_ca_code * qBasebandSignal)

        current_phase_mismatch = np.arctan(Q_P / I_P) / (2.0 * np.pi)
        phase_error = current_phase_mismatch - self.prev_carrier_phase_mismatch

        coeff1 = (2 * self.pll_tau2 - self.code_ca_period) / (2 * self.pll_tau1)
        coeff2 = self.code_ca_period / self.pll_tau1

        proportional_component = coeff1 * phase_error
        integrated_component = (
            coeff2 * phase_error + coeff2 * self.prev_carrier_phase_mismatch
        )

        carrier_VCO_value = (
            self.prev_carrier_VCO + proportional_component + integrated_component
        )

        self.prev_carrier_VCO = carrier_VCO_value
        self.prev_carrier_phase_mismatch = current_phase_mismatch
        self.current_carrier_frequency = (
            self.acquisition_info.acquired_frequency + carrier_VCO_value
        )

        self.PLL_carrier_frequency[current_sample_ind] = self.current_carrier_frequency
        self.PLL_phase_errors[current_sample_ind] = phase_error
        self.PLL_vco_value[current_sample_ind] = carrier_VCO_value

        self.I_P[current_sample_ind] = I_P
        self.Q_P[current_sample_ind] = Q_P

    def __dll_code_tracking_iteration(
        self, baseband_signal, signal_chunk_length, current_sample_ind
    ):
        tcode = np.arange(
            self.residual_ca_code_chips - self.early_late_spacing,
            signal_chunk_length * self.code_phase_step
            + self.residual_ca_code_chips
            - self.early_late_spacing,
            self.code_phase_step,
        )
        early_ca_code = self.ca_code_bits[np.ceil(tcode).astype(int)]

        tcode = np.arange(
            self.residual_ca_code_chips + self.early_late_spacing,
            signal_chunk_length * self.code_phase_step
            + self.residual_ca_code_chips
            + self.early_late_spacing,
            self.code_phase_step,
        )
        late_ca_code = self.ca_code_bits[np.ceil(tcode).astype(int)]

        early_module = np.abs(np.sum(early_ca_code * baseband_signal))
        late_module = np.abs(np.sum(late_ca_code * baseband_signal))

        ca_code_error = (early_module - late_module) / (early_module + late_module)
        code_deviation = ca_code_error - self.prev_ca_code_error

        coeff1 = (2 * self.dll_tau2 - self.code_ca_period) / (2 * self.dll_tau1)
        coeff2 = self.code_ca_period / self.dll_tau1

        proportional_component = coeff1 * code_deviation
        integrated_component = (
            coeff2 * code_deviation + coeff2 * self.prev_ca_code_error
        )  # TODO self.prev code deviation????
        ca_code_VCO_value = (
            self.prev_ca_code_VCO + proportional_component + integrated_component
        )

        self.prev_ca_code_VCO = ca_code_VCO_value
        self.prev_ca_code_error = ca_code_error
        self.current_ca_code_frequency = (
            self.settings.code_frequency_basis - ca_code_VCO_value
        )

        self.early_metric[current_sample_ind] = early_module
        self.late_metric[current_sample_ind] = late_module

        self.DLL_code_frequency[current_sample_ind] = self.current_ca_code_frequency
        self.DLL_code_error[current_sample_ind] = ca_code_error
        self.DLL_vco_value[current_sample_ind] = ca_code_VCO_value

    @staticmethod
    def locked_loop_coefficients(bandwidth: float, damping: float, total_gain: float):
        w_n = bandwidth * 8.0 * damping / (4.0 * damping**2 + 1)
        tau1 = total_gain / w_n**2
        tau2 = 2.0 * damping / w_n
        return tau1, tau2
