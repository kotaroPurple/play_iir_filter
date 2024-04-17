
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class WaveOption:
    is_sin: bool
    amplitude: float
    frequency: float


def generate_wave(
        sample_frequency: float, max_sec: float, wave_list: list[WaveOption]) \
        -> tuple[NDArray, NDArray]:
    times = np.linspace(0., max_sec, int(max_sec * sample_frequency + 0.5), endpoint=False)
    data = np.zeros_like(times)
    for one_wave in wave_list:
        if one_wave.is_sin:
            data += one_wave.amplitude * np.sin(2 * np.pi * one_wave.frequency * times)
        else:
            data += one_wave.amplitude * np.cos(2 * np.pi * one_wave.frequency * times)
    return data, times

# # テスト用の信号生成
# fs = 1000  # サンプリング周波数
# t = np.linspace(0, 1, fs, endpoint=False)
# data = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 50 * t)