import numpy as np
from tqdm import tqdm

from .prn_generator import generate_prn
from .storage import AcquisitionInfo, RecieverSettings


class AcquisitionComponent:
    def __init__(self, settings: RecieverSettings, iq_signal: np.array, ms_count=6):
        self.signal_iq_data = iq_signal
        self.ms_to_search = ms_count
        self.cur_settings = settings

    @staticmethod
    def periodic_correlation(x, y):
        """The x should be considered real for proper work of algorithm"""
        return np.fft.ifft(np.fft.fft(x).conj() * np.fft.fft(y))

    @staticmethod
    def resample_ca_code(ca_code, ms_count, samples_in_1ms):
        """For given chip bits "resamples" them to generate ca_code
        signal, that would allow multiplication for given sampling rate

        Args:
            ca_code (List[int]): chip bits of ca code
            ms_count (int): each code is raugly ???? 1ms length, so how many replicas needed
            samples_in_1ms (int): Value based on sample rate of programm

        Returns:
            List[int]: resampled version of input ca_code
        """
        samples_per_chip = samples_in_1ms / 1023

        resampled_ca = np.zeros(int(ms_count * samples_in_1ms))
        for i in range(len(resampled_ca)):
            resampled_ca[i] = ca_code[int((i + 1) / samples_per_chip) % 1023]

        return resampled_ca

    def process(self) -> dict[AcquisitionInfo]:
        results = {
            sat_prn: AcquisitionInfo(satellite_prn_number=sat_prn)
            for sat_prn in range(1, 32 + 1)
        }

        # TODO implement search for 10ms before and 10ms after
        samples_per_1ms = self.cur_settings.sampling_frequency * 1e-3
        time = np.arange(int(self.ms_to_search * samples_per_1ms))

        for sat_prn_num in tqdm(range(1, 32 + 1)):
            CA_code_sampled = self.resample_ca_code(
                generate_prn(sat_prn_num), self.ms_to_search, samples_per_1ms
            )

            max_value = 0
            max_doppler_shift, frequency_error = 10_000, 35
            for doppler_freq_shift in range(
                -max_doppler_shift, max_doppler_shift, frequency_error
            ):
                # correlation for baseband signal acts validly if first argument real signal
                # if freq wronw, correlation will give small result because of remaing `sin`

                count_samples = int(self.ms_to_search * samples_per_1ms)
                doppler_shift = np.exp(
                    1j
                    * 2
                    * np.pi
                    * (1 / self.cur_settings.sampling_frequency)
                    * doppler_freq_shift
                    * time
                )
                doppler_deshifted_signal = (
                    self.signal_iq_data[0:count_samples] * doppler_shift
                )

                corr = self.periodic_correlation(
                    CA_code_sampled, doppler_deshifted_signal
                )

                # needs normalization due to ms to correlate interval
                # Abs module compensates for const difference in phase
                corr_data = np.abs(corr) / len(CA_code_sampled)
                best_metric = np.max(corr_data)

                if best_metric > max_value:
                    max_value = best_metric
                    results[sat_prn_num].peak_correlation = best_metric

                    results[sat_prn_num].acquired_frequency = doppler_freq_shift
                    results[sat_prn_num].code_phase = np.argmax(corr_data)

        return results
