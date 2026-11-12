"""Application exporter"""

import logging
import os
import time

import prometheus_client
import pynvml

logger = logging.getLogger(__name__)


class AppMetrics:
    """
    Representation of Prometheus metrics and loop to fetch and transform
    application metrics into Prometheus metrics.
    """

    def __init__(self, polling_interval_seconds=5):
        self.polling_interval_seconds = polling_interval_seconds
        self.labels = ["minor_number", "uuid", "name"]
        # Prometheus metrics to collect
        self.num_devices = prometheus_client.Gauge("nvidia_num_devices", "Number of GPU devices")
        self.used_memory = prometheus_client.Gauge(
            "nvidia_memory_used_bytes", "Memory used by the GPU device in bytes", self.labels
        )
        self.total_memory = prometheus_client.Gauge(
            "nvidia_memory_total_bytes", "Total memory of the GPU device in bytes", self.labels
        )
        self.duty_cycle = prometheus_client.Gauge(
            "nvidia_duty_cycle",
            "Percent of time over the past sample period during which one or more kernels were executing on the GPU device",
            self.labels,
        )
        self.power_usage = prometheus_client.Gauge(
            "nvidia_power_usage_milliwatts", "Power usage of the GPU device in milliwatts", self.labels
        )
        self.temperature = prometheus_client.Gauge(
            "nvidia_temperature_celsius", "Temperature of the GPU device in celsius", self.labels
        )
        self.fan_speed = prometheus_client.Gauge(
            "nvidia_fanspeed_percent", "Fanspeed of the GPU device as a percent of its maximum", self.labels
        )

        self.compute_process_memory = prometheus_client.Gauge(
            "nvidia_compute_process_memory",
            "Memory used by compute process",
            self.labels + ["process_id", "process_name"],
        )
        self.graphics_process_memory = prometheus_client.Gauge(
            "nvidia_graphics_process_memory",
            "Memory used by graphics process",
            self.labels + ["process_id", "process_name"],
        )
        self.MPScompute_process_memory = prometheus_client.Gauge(
            "nvidia_MPScompute_process_memory",
            "Memory used by MPS compute process",
            self.labels + ["process_id", "process_name"],
        )
        self.collect_errors = prometheus_client.Counter(
            "nvidia_collect_errors", "Number of NVML errors met while collecting metrics"
        )

    def run_metrics_loop(self):
        """Metrics fetching loop"""

        while True:
            try:
                self.fetch()
            except pynvml.NVMLError as error:
                # NVML errors are transient most of the time (process exiting,
                # driver being reloaded...): keep the exporter alive, the
                # previous values stay exposed until the next successful run.
                self.collect_errors.inc()
                logger.warning("NVML error while collecting metrics, keeping previous values: %s", error)
            time.sleep(self.polling_interval_seconds)

    @staticmethod
    def _process_name(pid):
        """
        Name of a process using the GPU, or "unknown" when NVML cannot give it:
        a process that is exiting, or a zombie still holding a GPU context, makes
        nvmlSystemGetProcessName raise NVMLError_NoData / NVMLError_NotFound.
        """
        try:
            return pynvml.nvmlSystemGetProcessName(pid)
        except pynvml.NVMLError:
            return "unknown"

    def fetch(self):
        """
        Get metrics from application and refresh Prometheus metrics with
        new values.
        """
        pynvml.nvmlInit()
        num_devices = pynvml.nvmlDeviceGetCount()
        self.num_devices.set(num_devices)
        self.compute_process_memory.clear()
        self.graphics_process_memory.clear()
        self.MPScompute_process_memory.clear()
        for i in range(num_devices):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            labels = [
                pynvml.nvmlDeviceGetMinorNumber(handle),
                pynvml.nvmlDeviceGetUUID(handle),
                pynvml.nvmlDeviceGetName(handle),
            ]
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)

            self.used_memory.labels(*labels).set(memory.used)
            self.total_memory.labels(*labels).set(memory.total)
            self.duty_cycle.labels(*labels).set(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            self.power_usage.labels(*labels).set(pynvml.nvmlDeviceGetPowerUsage(handle))
            self.temperature.labels(*labels).set(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU))
            self.fan_speed.labels(*labels).set(pynvml.nvmlDeviceGetFanSpeed(handle))

            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                p_labels = [p.pid, self._process_name(p.pid)]
                self.compute_process_memory.labels(*labels, *p_labels).set(p.usedGpuMemory or 0)

            for p in pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle):
                p_labels = [p.pid, self._process_name(p.pid)]
                self.graphics_process_memory.labels(*labels, *p_labels).set(p.usedGpuMemory or 0)

            for p in pynvml.nvmlDeviceGetMPSComputeRunningProcesses(handle):
                p_labels = [p.pid, self._process_name(p.pid)]
                self.MPScompute_process_memory.labels(*labels, *p_labels).set(p.usedGpuMemory or 0)


def main():
    """Main entry point"""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    polling_interval_seconds = int(os.getenv("POLLING_INTERVAL_SECONDS", "5"))
    exporter_port = int(os.getenv("EXPORTER_PORT", "9102"))

    app_metrics = AppMetrics(polling_interval_seconds=polling_interval_seconds)
    prometheus_client.start_http_server(exporter_port)
    app_metrics.run_metrics_loop()
