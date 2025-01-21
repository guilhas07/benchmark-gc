import glob
import os
import re
import subprocess
from typing import Optional

_dir = os.path.dirname(__file__)
_BENCHMARK_STATS_PATH = f"{_dir}/benchmark_stats"
_BENCHMARK_LOG_PATH = f"{_dir}/benchmark_logs"
_BENCHMARK_MATRIX_PATH = f"{_dir}/benchmark_matrix"

# NOTE: there is a no-op GC Epsilon
_GCS = ["G1", "Parallel", "Shenandoah", "Z"]


def get_cpu_count():
    cpu = os.cpu_count()
    assert cpu is not None, "Error: cpu_count is None"
    return cpu


def clean_stats():
    print("Cleaning logs and stats directories")
    for i in glob.glob(f"{_BENCHMARK_STATS_PATH}/**/*.json", recursive=True):
        os.remove(i)


def clean_logs():
    for i in glob.glob(f"{_BENCHMARK_LOG_PATH}/**/*.log", recursive=True):
        os.remove(i)


def get_benchmark_log_path(
    gc: str, benchmark_group: str, benchmark_name: str, heap_size: str
) -> str:
    return f"{_BENCHMARK_LOG_PATH}/{benchmark_group}_{benchmark_name}_{gc}_{heap_size}m.log"


def get_benchmark_stats_path(
    gc: str,
    benchmark_group: str,
    benchmark_name: str,
    heap_size: str,
    jdk: str,
    error: Optional[str],
) -> str:
    return f"{_BENCHMARK_STATS_PATH}/{benchmark_group}_{benchmark_name}_{gc}_{heap_size}m_{jdk}{'' if error is None else '_error'}.json"


def get_error_report_path(jdk: str, date: str) -> str:
    return f"{_BENCHMARK_STATS_PATH}/gc_stats/error_report_{jdk}_{date}.json"


def get_gc_stats_path(gc: str, jdk: str) -> str:
    return f"{_BENCHMARK_STATS_PATH}/gc_stats/{gc}_{jdk}.json"


def get_matrix_path(jdk: str, date: str) -> str:
    return f"{_BENCHMARK_MATRIX_PATH}/{jdk}_{date}.json"


def _parse_java_version() -> str | None:
    p = subprocess.run(["java", "--version"], capture_output=True, text=True)
    if p.returncode != 0:
        return None
    output = p.stdout.splitlines()

    # Parse OpenJDK based runtimes
    if "openjdk" in output[0]:
        matches = re.findall("OpenJDK Runtime Environment (\\w*)-", output[1])
        runtime = matches[0] if len(matches) else "OpenJDK"
    # Parse HotSpot based runtimes
    elif "HotSpot" in output[2]:
        matches = re.findall(
            "Java\\(TM\\) SE Runtime Environment \\w* (\\w*)",
            output[1],
        )
        runtime = matches[0] if len(matches) else "HotSpot"
    else:
        print(
            "Runtime not supported. Please open an issue to extend java parsing capabilities."
        )
        return None

    p = subprocess.run(
        ["java", "-XshowSettings:properties", "-version"],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        return None

    # NOTE: java outputs settings info to stderr
    version = re.findall("java\\.version = (\\d+\\.\\d+\\.\\d+)", p.stderr)[0]

    return f"{runtime}_{version}"


def _get_garbage_collectors() -> list[str] | None:
    # TODO: maybe merge PR to jdk that allows to get all supported Garbage Collectors
    gcs = []
    for gc in _GCS:
        p = subprocess.run(
            ["java", f"-XX:+Use{gc}GC", "-version"],
            capture_output=True,
            text=True,
        )
        if p.returncode == 0:
            gcs.append(gc)

    return gcs if len(gcs) != 0 else None


def get_garbage_collectors() -> list[str]:
    return _GCS


def get_java_env_info() -> tuple[str, list[str]] | tuple[None, None]:
    """
    Get list of available garbage collectors for your java version.

    Returns:
        Tuple of jdk: str | None, garbage_collectors: list[str] | None
    """

    version = _parse_java_version()
    if version is None:
        return None, None

    gcs = _get_garbage_collectors()
    if gcs is None:
        return None, None
    return version, gcs


def get_heap_sizes() -> list[str]:
    """Returns a list of heap sizes in megabytes"""
    return ["256", "512", "1024", "2048", "4096", "8192"]
