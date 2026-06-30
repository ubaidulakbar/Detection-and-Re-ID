import argparse
import polars as pl
import pynvml as nvml
import threading
import time
import torch
from ultralytics.utils.benchmarks import benchmark

DEVICE = 0 if torch.cuda.is_available() else 'cpu'


class PowerSampler(threading.Thread):
    def __init__(self, device_index: int = 0, interval_s: float = 0.05):
        super().__init__(daemon=True)
        self.device_index = device_index
        self.interval_s = interval_s
        self.samples = []
        self._running = False
        self._handle = None

    def run(self):
        nvml.nvmlInit()
        self._handle = nvml.nvmlDeviceGetHandleByIndex(self.device_index)

        self._running = True
        while self._running:
            t = time.time()
            power_mw = nvml.nvmlDeviceGetPowerUsage(self._handle)  # milliwatts
            power_w = power_mw / 1000.0                            # watts
            self.samples.append((t, power_w))
            time.sleep(self.interval_s)

        nvml.nvmlShutdown()

    def stop(self):
        self._running = False

    def _windowed_samples(self, t_start: float, t_end: float):
        return [(t, p) for (t, p) in self.samples if t_start <= t <= t_end]

    def integrate_energy(self, t_start: float, t_end: float) -> float:
        samples = self._windowed_samples(t_start, t_end)
        if not samples:
            return 0.0

        samples.sort(key=lambda x: x[0])
        energy = 0.0

        for i in range(len(samples) - 1):
            t_i, p_i = samples[i]
            t_j, _ = samples[i + 1]
            dt = max(0.0, min(t_j, t_end) - max(t_i, t_start))
            if dt > 0:
                energy += p_i * dt

        t_last, p_last = samples[-1]
        if t_last < t_end:
            dt_tail = t_end - t_last
            energy += p_last * dt_tail

        return energy


def run_benchmark(model_path: str, fmt: str, fp16: bool):
    # Reset CUDA stats and sync
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    sampler = PowerSampler(device_index=0, interval_s=0.05)
    sampler.start()

    torch.cuda.synchronize()
    t_start = time.time()

    table = benchmark(
        model=model_path,
        data='Yolo-Stamp-3Cs/data.yaml',
        imgsz=640,
        device=DEVICE,
        format=fmt,
        half=fp16
    )

    if isinstance(table, list):
        table = table[0]

    torch.cuda.synchronize()
    t_end = time.time()
    total_time = t_end - t_start

    sampler.stop()
    sampler.join()

    # Total energy in Joules over the benchmark window
    total_energy_j = sampler.integrate_energy(t_start, t_end)

    # Average power (W)
    avg_power = total_energy_j / max(1e-9, total_time)

    # Peak GPU memory (MB)
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_mem_mb = peak_bytes / (1024 ** 2)

    # FPS from benchmark table
    fps = float('nan')
    if isinstance(table, pl.DataFrame) and 'FPS' in table.columns:
        try:
            fps = float(table['FPS'].to_list()[0])
        except Exception:
            fps = float('nan')

    # Energy per frame (J/frame), using total energy and frames = fps * time
    energy_per_frame = float('nan')
    if fps and fps > 0 and total_time > 0:
        frames = fps * total_time
        energy_per_frame = total_energy_j / frames

    energy_df = pl.DataFrame(
        {
            'FPS': [fps],
            'Energy Per Frame': [energy_per_frame],
            'Average GPU Power': [avg_power],
            'Peak GPU Memory Usage (MB)': [peak_mem_mb],
        }
    )

    # First table: original benchmark
    print(table)

    # Second table: energy metrics
    print('=== Energy Metrics ===')
    print(energy_df)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run YOLO benchmark and measure energy per frame (NVML).'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the model .pt file',
    )

    parser.add_argument(
        '--format',
        type=str,
        default='-',
        help='Benchmark format (engine for TensorRT and - for PyTorch)',
    )

    parser.add_argument(
        '--fp16',
        type=bool,
        default=False,
        help='Half Precision. True or False',
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_benchmark(args.model, args.format, args.fp16)