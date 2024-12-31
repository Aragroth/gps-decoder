from dataclasses import dataclass


@dataclass(kw_only=True)
class RecieverSettings:
    number_of_channels: int = 1

    DLL_damping_ratio: float = 0.7
    DLL_noise_bandwidth: float = 5.0
    DLL_correlation_spacing: float = 0.5

    PLL_damping_ration: float = 0.7
    PLL_noise_bandwidth: float = 30

    data_adapt_coeff: int = 2
    code_frequency_basis: float = 1.023e6
    code_length: int = 1023

    code_periods_to_process: int = 1000
    intermidiate_frequency: int = 0
    file_type: int = 2

    sampling_frequency: int


@dataclass
class AcquisitionInfo:
    satellite_prn_number: int
    peak_correlation: float = 0.0
    code_phase: int = 0.0
    acquired_frequency: int = 0.0


@dataclass
class EphemerisParameters:
    # subframe 1
    week_number: int = 0
    sat_accuracy: int = 0
    sat_health: int = 0
    issue_of_data_clock: int = 0
    T_GD: float = 0.0
    t_oc: float = 0.0
    a_f2: float = 0.0
    a_f1: float = 0.0
    a_f0: float = 0.0

    # subframe 2
    issue_of_data_emphemeris: int = 0
    c_rs: float = 0.0
    delta_n: float = 0.0
    m_0: float = 0.0
    c_uc: float = 0.0
    e_s: float = 0.0
    c_us: float = 0.0
    sqrt_a_s: float = 0.0
    t_oe: float = 0.0

    # subframe 3
    c_ic: float = 0.0
    omega_0: float = 0.0
    c_is: float = 0.0
    i_0: float = 0.0
    c_rc: float = 0.0
    omega: float = 0.0
    omega_dot: float = 0.0
    IDOE: int = 0
    i_dot: float = 0.0
