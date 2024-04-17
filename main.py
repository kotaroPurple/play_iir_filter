
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from wave_generator import generate_wave
from wave_generator import WaveOption
from iir_filter import apply_filter
from iir_filter import apply_filter_with_chunk
# from iir_filter import apply_filter_simple
from iir_filter import generate_lowpass_filter


# テスト用の信号生成
chunk = 100
n_chunk = 10
fs = 1000  # サンプリング周波数

wave_options = [WaveOption(True, 1., 5.), WaveOption(True, 0.5, 50.)]
data, times = generate_wave(fs, 1., wave_options)
# data = np.random.random(len(data))

# フィルタ適用
cutoff_freq = 20  # カットオフ周波数
order = 5  # フィルタ次数
padtype = 'odd' if order // 2 == 1 else 'even'
b, a = generate_lowpass_filter(cutoff_freq, fs, order)

filtered = apply_filter(data, b, a)
chunk_filtered = apply_filter_with_chunk(data, b, a, chunk)
# simple_filtered = apply_filter_simple(data, b, a)
scipy_filtered = signal.filtfilt(b, a, data, padlen=0, padtype=padtype)

print(f'Chunk Filtered is the same: {np.allclose(filtered, chunk_filtered)}')

# プロット
offset = 4 * max(len(a), len(b)) + order // 2  # 3 * max(len(a), len(b))
plt.figure(figsize=(12, 8))
plt.plot(times, data, label='Original Data', alpha=0.7)
plt.plot(times, filtered, label='Filtered Data', alpha=0.7)
plt.plot(times, chunk_filtered, label='Chunk', alpha=0.7)
# plt.plot(times, simple_filtered, label='Simple', alpha=0.7)
plt.plot(times[:-offset], filtered[offset:], label='offset', alpha=0.7)
plt.plot(times, scipy_filtered, label='Scipy Data')

plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.title('IIR Filter')
plt.legend()
plt.show()
