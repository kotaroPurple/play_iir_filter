
from collections import deque

import numpy as np
from numpy.typing import NDArray
from scipy import signal


def generate_lowpass_filter(
        cutoff: float, sample_frequency: int, order: int) -> tuple[NDArray, NDArray]:
    nyquist = 0.5 * sample_frequency
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order, normal_cutoff, btype='low')
    return b, a


def apply_filter(data: NDArray, filter_b: NDArray, filter_a: NDArray) -> NDArray:
    filtered_data = np.zeros_like(data)
    nb = len(filter_b)
    na = len(filter_a)

    buffer_x = deque([0.] * nb)
    buffer_y = deque([0.] * (na - 1))

    for i in range(len(data)):
        # filter b
        buffer_x.pop()
        buffer_x.appendleft(data[i])
        filtered_data[i] = np.dot(np.array(buffer_x), filter_b)
        # filter a
        filtered_data[i] -= np.dot(np.array(buffer_y), filter_a[1:])
        buffer_y.pop()
        buffer_y.appendleft(filtered_data[i])
    return filtered_data


def apply_filter_with_chunk(
        data: NDArray, filter_b: NDArray, filter_a: NDArray, chunk: int) -> NDArray:
    number = len(data)
    filtered_data = np.zeros_like(data)
    nb = len(filter_b)
    na = len(filter_a)
    buffer_x = deque([0.] * nb)
    buffer_y = deque([0.] * (na - 1))

    for ch in range(number // chunk):
        one_filtered = _apply_short_filter(
            data[ch * chunk:(ch + 1) * chunk], filter_b, filter_a, buffer_x, buffer_y)
        filtered_data[ch * chunk:(ch + 1) * chunk] = one_filtered
    return filtered_data


def _apply_short_filter(
        data: NDArray, filter_b: NDArray, filter_a: NDArray,
        buffer_b: deque, buffer_a: deque) -> NDArray:
    filtered_data = np.zeros_like(data)
    for i in range(len(data)):
        # filter b
        buffer_b.pop()
        buffer_b.appendleft(data[i])
        filtered_data[i] = np.dot(np.array(buffer_b), filter_b)
        # filter a
        filtered_data[i] -= np.dot(np.array(buffer_a), filter_a[1:])
        buffer_a.pop()
        buffer_a.appendleft(filtered_data[i])
    return filtered_data


def apply_filter_simple(data: NDArray, filter_b: NDArray, filter_a: NDArray) -> NDArray:
    filtered_data = np.zeros_like(data)
    nb = len(filter_b)
    na = len(filter_a)

    for i in range(len(data)):
        filtered_data[i] = filter_b[0] * data[i]
        for j in range(1, min(i+1, nb)):
            filtered_data[i] += filter_b[j] * data[i - j]
        for j in range(1, min(i+1, na)):
            filtered_data[i] -= filter_a[j] * filtered_data[i - j]
    return filtered_data
