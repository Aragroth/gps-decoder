import wave

import numpy as np
from sigmf import SigMFFile, sigmffile


class SignalLoader:
    @staticmethod
    def load_gps_sym(filename, sample_rate, elements_np_type) -> tuple[int, np.array]:
        data = np.fromfile(filename, dtype=elements_np_type)
        data = data.reshape(-1, 2)

        signal_iq_data = data[:, 0].astype(np.float32) + 1j * data[:, 1].astype(
            np.float32
        )
        del data

        return sample_rate, signal_iq_data

    @staticmethod
    def sigmf_loader(filename) -> tuple[int, np.array]:
        signal = sigmffile.fromfile(filename)

        sample_rate = signal.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
        signal_iq_data = signal.read_samples()

        return sample_rate, signal_iq_data

    def iq_wav_loader(filename, elements_np_type) -> tuple[int, np.array]:
        with wave.open(filename, "rb") as wav_file:
            params = wav_file.getparams()

            num_channels = params.nchannels
            if num_channels != 2:
                raise ValueError("This file seems not to be an IQ file")

            sample_rate = params.framerate
            num_frames = params.nframes

            frames = wav_file.readframes(num_frames)
            audio_data = np.frombuffer(frames, dtype=elements_np_type)
            audio_data = audio_data.reshape((-1, 2))

            left_channel = audio_data[:, 0].astype(np.float32)
            right_channel = audio_data[:, 1].astype(np.float32)

            signal_iq_data = left_channel + 1j * right_channel

        return sample_rate, signal_iq_data
