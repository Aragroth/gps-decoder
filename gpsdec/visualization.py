import folium
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from .acquisition import AcquisitionComponent
from .prn_generator import generate_prn
from .solver import PseudorangeSolver
from .storage import AcquisitionInfo, RecieverSettings
from .tracking import TrackingComponent


def signal_spectre(signal_data, sample_rate, start_point=0, end_point=1000):
    to_show = signal_data[range(start_point, end_point)]

    fft_result = np.fft.fft(to_show)

    magnitude = np.abs(fft_result)
    phase = np.angle(fft_result)

    freq = np.fft.fftfreq(len(to_show), 1 / sample_rate)

    freq_shifted = np.fft.fftshift(freq)
    magnitude_shifted = np.fft.fftshift(magnitude)
    phase_shifted = np.fft.fftshift(phase)

    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.plot(freq_shifted, magnitude_shifted)
    plt.title("Magnitude")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")

    plt.subplot(122)
    plt.plot(freq_shifted, phase_shifted)
    plt.title("Phase")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Phase [radians]")

    plt.tight_layout()
    plt.show()


def acquisition_information(acqusition_results: dict[AcquisitionInfo]):
    result_peak_corr = [0] + [
        acqusition_results[i].peak_correlation for i in range(1, 32 + 1)
    ]
    threshold = np.mean(result_peak_corr)
    sat_found_indices = np.where(result_peak_corr > threshold)[0]

    colors = [
        "green" if ind in sat_found_indices else "red" for ind in range(1, 32 + 1)
    ]
    result_peak_corr_scaled = (
        10
        * (result_peak_corr - np.min(result_peak_corr))
        / (np.max(result_peak_corr) - np.min(result_peak_corr))
    )

    plt.bar(range(1, 32 + 1), result_peak_corr_scaled[1:], color=colors)

    legend_handles = [
        Line2D([0], [0], color="green", lw=4, label="Acquired satellite"),
        Line2D([0], [0], color="red", lw=4, label="Not acquired"),
    ]
    plt.legend(handles=legend_handles)
    plt.title("Acquisition results")
    plt.xlabel("PRN number")
    plt.ylabel("Metric")
    plt.grid()

    plt.show()


def show_signal_iq_plot(
    iq_signal, acq_info: AcquisitionInfo, settings: RecieverSettings, ms_to_show
):
    samples_per_1ms = settings.sampling_frequency * 1e-3
    doppler_freq_shift = acq_info.acquired_frequency
    chip_offset = acq_info.code_phase

    results_I = []
    results_Q = []
    for ms in range(1, ms_to_show):
        ca_code = AcquisitionComponent.resample_ca_code(
            generate_prn(acq_info.satellite_prn_number), 1, samples_per_1ms
        )
        shifted_ca_code = np.roll(ca_code, chip_offset)

        local_signal = iq_signal[
            int((ms - 1) * samples_per_1ms) : int(ms * samples_per_1ms)
        ]

        time = np.arange(int(1 * samples_per_1ms)) / settings.sampling_frequency
        result = (
            local_signal
            * shifted_ca_code
            * np.exp(1j * 2 * np.pi * doppler_freq_shift * time)
        )

        results_I.append(np.sum(np.real(result)))
        results_Q.append(np.sum(np.imag(result)))

    plt.figure(figsize=(6, 6))
    plt.scatter(results_I, results_Q, color="blue", s=1)
    plt.title(f"IQ Constellation Diagram for sat {acq_info.satellite_prn_number}")
    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True)
    plt.axis("equal")
    plt.show()


def sat_tracker_results(sat_tracker: TrackingComponent):
    fig, axes = plt.subplots(3, 1, figsize=(6, 10))

    axes[0].scatter(sat_tracker.I_P, sat_tracker.Q_P, color="blue", s=1)
    axes[0].set_title(
        f"IQ Constellation Diagram of satellite {sat_tracker.acquisition_info.satellite_prn_number}"
    )
    axes[0].set_xlabel("In-phase (I)")
    axes[0].set_ylabel("Quadrature (Q)")
    axes[0].grid(True)
    axes[0].axis("equal")

    to_show_ms = 400
    axes[1].plot(range(to_show_ms), sat_tracker.I_P[:to_show_ms])
    axes[1].set_title(
        f"Navigation data bits of satellite {sat_tracker.acquisition_info.satellite_prn_number}"
    )
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    axes[2].plot(sat_tracker.PLL_carrier_frequency[:to_show_ms])
    axes[2].set_title(
        f"PLL carrier frequency of satellite {sat_tracker.acquisition_info.satellite_prn_number}"
    )
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Frequency [Hz]")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()


def show_solution(position_solver: PseudorangeSolver, true_position=None):
    last_lat = position_solver.coordinate_iterations[-1][0]
    last_lon = position_solver.coordinate_iterations[-1][1]
    m = folium.Map(location=[last_lat, last_lon], zoom_start=17, max_zoom=20)

    for lat, lon in position_solver.coordinate_iterations:
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=1,
            tooltip=f"({lat:.4f}, {lon:.4f})",
        ).add_to(m)

    folium.PolyLine(
        locations=position_solver.coordinate_iterations,
        color="blue",
        weight=2.5,
        opacity=1,
        tooltip="Iterations",
    ).add_to(m)

    spacing_for_text = 0.0001
    folium.Marker(
        [last_lat + spacing_for_text, last_lon + spacing_for_text],
        color="red",
        icon=folium.DivIcon(
            html='<div style="font-weight: bold; font-size: 12pt; color: blue; width: 300px">End point of iterations</div>'
        ),
    ).add_to(m)

    if true_position is None:
        return m

    latitude_init, longitude_init = true_position
    folium.CircleMarker(
        [latitude_init, longitude_init],
        popup="Target Location",
        tooltip="Point of simulation",
        fill_opacity=0.5,
        color="red",
    ).add_to(m)

    folium.Marker(
        [latitude_init - spacing_for_text, longitude_init + spacing_for_text],
        color="red",
        icon=folium.DivIcon(
            html='<div style="font-weight: bold; font-size: 12pt; color: red; width: 300px">True position</div>'
        ),
    ).add_to(m)

    return m
