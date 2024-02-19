import glob
import re
import subprocess
import time
from collections import defaultdict
from enum import Enum
from os import path
from threading import Timer
from typing import Optional
import json

import numpy

import utils
from model import (
    BenchmarkReport,
    ErrorReport,
    GarbageCollectorReport,
)


class BENCHMARK_GROUP(Enum):
    RENAISSANCE = "Renaissance"
    DACAPO = "DaCapo"


_debug = False
_dir = path.dirname(__file__)
_BENCHMARK_PATH = f"{_dir}/benchmark_apps"
_BENCHMARK_CONFIG_PATH = f"{_dir}/benchmarks_config.json"
_benchmark_paths = {
    BENCHMARK_GROUP.RENAISSANCE.value: f"{_BENCHMARK_PATH}/renaissance-gpl-0.15.0.jar",
    BENCHMARK_GROUP.DACAPO.value: f"{_BENCHMARK_PATH}/dacapo-23.11-chopin.jar",
}

benchmarks_config = {}
try:
    with open(_BENCHMARK_CONFIG_PATH) as f:
        benchmarks_config = json.loads(f.read())
except Exception as e:
    print(f"Error while reading file {_BENCHMARK_CONFIG_PATH} {str(e)[:50]}")


def set_debug(value: bool):
    global _debug
    _debug = value


def _get_benchmarks(benchmark_group: BENCHMARK_GROUP) -> list[str]:
    match benchmark_group:
        case BENCHMARK_GROUP.RENAISSANCE:
            result = subprocess.run(
                [
                    "java",
                    "-jar",
                    _benchmark_paths[benchmark_group.value],
                    "--raw-list",
                ],
                capture_output=True,
                text=True,
            )
            assert result.stderr == "", result.stderr
            return result.stdout.splitlines()

        case BENCHMARK_GROUP.DACAPO:
            result = subprocess.run(
                [
                    "java",
                    "-jar",
                    _benchmark_paths[benchmark_group.value],
                    "--list-benchmarks",
                ],
                capture_output=True,
                text=True,
            )
            assert result.stderr == "", result.stderr
            return result.stdout.split()
        case _:
            raise AssertionError("Benchmark group is not supported")


def _get_benchmark_command(
    benchmark_group: BENCHMARK_GROUP,
    benchmark: str,
    gc: str,
    heap_size: str,
    iterations: int,
) -> list[str]:
    bench_path = _benchmark_paths[benchmark_group.value]
    command = [
        "java",
        f"-XX:+Use{gc}GC",
        f"-Xms{heap_size}m",
        f"-Xmx{heap_size}m",
        f"-Xlog:gc*,safepoint:file={utils.get_benchmark_log_path(gc, benchmark_group.value, benchmark, heap_size)}::filecount=0",
    ]

    settings = benchmarks_config.get(benchmark_group.value, {}).get(benchmark, {})
    command.extend(settings.get("java", []))

    command.extend(["-jar", bench_path, benchmark])

    match benchmark_group:
        case BENCHMARK_GROUP.RENAISSANCE:
            command.extend(["-r", f"{iterations}", "--no-forced-gc"])
        case BENCHMARK_GROUP.DACAPO:
            command.extend(["-n", f"{iterations}", "--no-pre-iteration-gc"])
        case _:
            raise AssertionError(f"{benchmark_group} not supported")
    return command


def run_benchmark(
    benchmark_group: BENCHMARK_GROUP,
    benchmark: str,
    gc: str,
    heap_size: str,
    iterations: int,
    jdk: str,
    timeout: int | None,
) -> BenchmarkReport:
    """
    Args:
        timeout: int -> value in seconds to interrupt benchmark
    Returns:
        BenchmarkReport
    """

    # NOTE: No-op timer and file
    class DummyTimerAndFile:
        def start(self):
            pass

        def close(self):
            pass

        def cancel(self):
            pass

    def kill_process(process: subprocess.Popen[bytes], cmd: str):
        print(
            f"Killing command: \n\t{' '.join(cmd)}\n\tdue to timeout with {timeout} seconds.\n"
        )
        # file = utils.get_benchmark_debug_path(
        #     gc, benchmark_group.value, benchmark, heap_size
        # )
        # with open(file, "wb") as f:
        #     if process.stdout is not None:
        #         print(f"Writing to {file}...")
        #         f.write(process.stdout.read())
        process.kill()
        # process.wait()

    benchmark_command = _get_benchmark_command(
        benchmark_group, benchmark, gc, heap_size, iterations
    )

    print(
        f"[{benchmark_group.value}]: Running benchmark {benchmark} with:\n"
        f"\tGC: {gc}\n"
        f"\tHeap Size: {heap_size}\n"
        f"\tIterations: {iterations}\n"
        f"\tCommand: {' '.join(benchmark_command)}"
    )

    file = DummyTimerAndFile()
    file_path = utils.get_benchmark_debug_path(
        gc, benchmark_group.value, benchmark, heap_size
    )
    if _debug:
        file = open(file_path, "w")
        process = subprocess.Popen(
            benchmark_command, stdout=file, stderr=subprocess.STDOUT
        )
    else:
        process = subprocess.Popen(
            benchmark_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    timer = (
        DummyTimerAndFile()
        if timeout is None
        else Timer(timeout, kill_process, (process, benchmark_command))
    )

    time_start = time.time_ns()
    pid = process.pid
    cpu_usage_stats = []
    cpu_time_stats = []
    io_time_stats = []

    # TODO: see better polling method than sleeping 1 seconds for throughput
    timer.start()
    while process.poll() is None:
        # subprocess.run with capture_output doesn't seem to capture the whole output when
        # using top with -1 flag
        p = subprocess.run(
            ["top", "-bn", "1", "-p", f"{pid}"], capture_output=True, text=True
        )

        lines = p.stdout.splitlines()
        # NOTE: from man top(1)
        # us, user    : time running un-niced user processes
        # wa, IO-wait : time waiting for I/O completion
        # %Cpu(s): 15.1 us,  2.2 sy,  0.0 ni, 81.2 id,  0.0 wa,  0.0 hi,  1.6 si,  0.0 st
        us, wa = map(float, re.findall("(\\d+.\\d+) us.*(\\d+.\\d+) wa", lines[2])[0])
        io_time_stats.append(wa)
        cpu_time_stats.append(us)

        lines = lines[-2:]
        assert lines[0].split()[8] == "%CPU", "Couldn't find %CPU in " + lines[0]
        cpu_usage = round(float(lines[1].split()[8]) / utils.get_cpu_count(), 1)
        cpu_usage_stats.append(cpu_usage)
        time.sleep(0.1)
    timer.cancel()
    throughput = time.time_ns() - time_start

    cpu_usage_avg = round(float(numpy.mean(cpu_usage_stats)), 1)
    cpu_time_avg = round(float(numpy.mean(cpu_time_stats)), 1)
    io_time_avg = round(float(numpy.mean(io_time_stats)), 1)
    p90_io = float(round(numpy.percentile(io_time_stats, 90), 2))

    print(
        f"{cpu_usage_avg=} {cpu_time_avg=} {io_time_avg=} {p90_io=} and {throughput=}"
    )

    if process.returncode == 0:
        print("Success")
        result = BenchmarkReport.build_benchmark_result(
            gc,
            benchmark_group.value,
            benchmark,
            heap_size,
            cpu_usage_avg,
            cpu_time_avg,
            io_time_avg,
            p90_io,
            throughput,
            jdk,
        )
    else:
        error = ""
        if _debug:
            error = f"Check {file_path} for error logs."
        elif process.stderr:
            error = process.stderr.read().decode()

        print(f"Error: {error}")
        result = BenchmarkReport.build_benchmark_error(
            gc,
            benchmark_group.value,
            benchmark,
            heap_size,
            cpu_usage_avg,
            cpu_time_avg,
            io_time_avg,
            p90_io,
            jdk,
            process.returncode,
            error,
        )
    file.close()
    result.save_to_json()
    return result


def run_benchmark_groups(
    gc: str,
    heap_size: str,
    iterations: int,
    jdk: str,
    timeout: int | None,
    benchmark_groups: Optional[list[BENCHMARK_GROUP]],
) -> list[BenchmarkReport]:
    """Run all benchmark groups or only the ones
    specified in `benchmark_groups`

    Args:
        `gc`: Garbage collector used when running the benchmarks
        `heap_size`: Heap size that the JVM will use
        `iterations`: Number of iterations to run the benchmark
        `jdk`: Java version
        `timeout`: Time in seconds to stop a benchmark from running
        `benchmark_groups`: list[BENCHMARK_GROUP] | None -> If you want to run a
        subset of the benchmarks

    Returns:
        list[BenchmarkReport]
    """
    benchmark_reports: list[BenchmarkReport] = []
    groups: list[BENCHMARK_GROUP] = benchmark_groups or [*BENCHMARK_GROUP]

    for benchmark_group in groups:
        benchmarks = _get_benchmarks(benchmark_group)

        for benchmark in benchmarks:
            # NOTE: Override with config settings
            settings = benchmarks_config.get(benchmark_group.value, {}).get(
                benchmark, {}
            )

            result = run_benchmark(
                benchmark_group,
                benchmark,
                gc,
                heap_size,
                settings.get("iterations", iterations),
                jdk,
                settings.get("timeout", timeout),
            )

            benchmark_reports.append(result)
    return benchmark_reports


def run_benchmarks(
    iterations: int,
    jdk: str,
    garbage_collectors: list[str],
    skip_benchmarks: bool,
    benchmark_groups: list[BENCHMARK_GROUP],
    timeout: int | None = None,
) -> dict[str, dict[str, list[BenchmarkReport]]]:
    """Run benchmarks

    Args:
        iterations: number of iterations to run benchmarks
        jdk: [TODO:description]
        garbage_collectors: [TODO:description]
        skip_benchmarks: [TODO:description]
        benchmark_groups: [TODO:description]
        timeout: [TODO:description]

    Returns:
        dict[gc, dict[heap_size, list[BenchmarkReport]]]
    """

    # { gc: heap_size: { list[BenchmarkReport } }
    benchmark_reports: dict[str, dict[str, list[BenchmarkReport]]] = defaultdict(
        lambda: defaultdict(list)
    )
    heap_sizes: list[str] = utils.get_heap_sizes()

    failed_benchmarks: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for gc in garbage_collectors:
        for heap_size in heap_sizes:
            if skip_benchmarks:
                benchmark_reports[gc][heap_size] = load_benchmark_reports(
                    gc, heap_size, jdk
                )
            else:
                benchmark_reports[gc][heap_size].extend(
                    run_benchmark_groups(
                        gc,
                        heap_size,
                        iterations,
                        jdk,
                        timeout,
                        benchmark_groups,
                    )
                )

            for result in benchmark_reports[gc][heap_size]:
                if not result.is_successfull():
                    failed_benchmarks[result.heap_size][result.benchmark_name].append(
                        (
                            result.garbage_collector,
                            result.error,  # type: ignore -> error is not None in case of a failed benchmark
                        )
                    )

    # NOTE: using list to avoid modifying dict while iterating
    for gc in list(benchmark_reports):
        for heap_size, results in list(benchmark_reports[gc].items()):
            valid_results = [
                el
                for el in results
                if failed_benchmarks.get(el.heap_size, {}).get(el.benchmark_name)
                is None
            ]

            if len(valid_results) == 0:
                del benchmark_reports[gc][heap_size]
                continue

            benchmark_reports[gc][heap_size] = valid_results
            assert all(
                el.is_successfull() for el in valid_results
            ), "All benchmarks should be successfull"

        if len(benchmark_reports[gc]) > 0:
            GarbageCollectorReport.build_garbage_collector_report(
                benchmark_reports[gc]
            ).save_to_json()
        else:
            f"Garbage Collector {gc} doesn't have successfull benchmarks."
            del benchmark_reports[gc]

    if len(failed_benchmarks) > 0:
        error_report = ErrorReport(jdk, failed_benchmarks)
        error_report.save_to_json()

    return benchmark_reports


def load_benchmark_reports(
    garbage_collector: str, heap_size: str, jdk: str
) -> list[BenchmarkReport]:
    return [
        BenchmarkReport.load_from_json(i)
        for i in glob.glob(
            f"{utils._BENCHMARK_STATS_PATH}/*{garbage_collector}_{heap_size}m_{jdk}*.json"
        )
    ]


def load_garbage_collector_results(jdk: str) -> list[GarbageCollectorReport]:
    return [
        GarbageCollectorReport.load_from_json(i)
        for i in glob.glob(f"{utils._BENCHMARK_STATS_PATH}/gc_stats/*{jdk}.json")
    ]
