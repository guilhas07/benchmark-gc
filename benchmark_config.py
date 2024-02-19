from __future__ import annotations
from dataclasses import dataclass
import json
import re
import numpy
from collections import defaultdict
import subprocess
from threading import Timer
import time

from typing import Any, Optional, get_args, get_type_hints
import string

import utils
from model import BenchmarkReport, ErrorReport, GarbageCollectorReport


@dataclass
class BenchmarkRunOptions:
    command: Optional[str] = None
    extra_java_options: Optional[str] = None
    post_exec_script: Optional[str] = None
    timeout: Optional[int] = None

    def __post_init__(self):
        # NOTE: due to the import of __future__ annotations the
        # hints are stored as strings and need to be resolved with get_type_hints
        for field_name, field_def in get_type_hints(self.__class__).items():
            v = getattr(self, field_name)
            assert (
                type(v) in get_args(field_def)
            ), f"Error: {field_name} with type {type(v)} should have type in {get_args(field_def)}."


@dataclass
class BenchmarkConfig:
    name: str
    run_options: BenchmarkRunOptions | None

    def __init__(self, name: str, run_options: Optional[dict] = None):
        self.name = name
        self.run_options = (
            run_options if run_options is None else BenchmarkRunOptions(**run_options)
        )
        self.__post_init__()

    def __post_init__(self):
        # NOTE: No need to validate `run_options` because it errors before in __init__
        validate_str(getattr(self, "name"), self.__class__.__name__, "name")

    def run(self):
        raise NotImplementedError()


@dataclass
class BenchmarkSuite:
    """
    Represents a suite of benchmarks to be executed

    Attributes:
        suite_name: Name used to save log/stats files
        jar_path: Path to your suite jar
        run_options: Options to run your benchmarks
        benchmarks_config: A list of individual benchmark configurations.
            If this is specified your suite will use the benchmark name like this:
            (default java options) (extra_java_options) ./jar_path **benchmark_name** (benchmark_options)
            NOTE: It is possible to override every benchmark option in BenchmarkSuite by specifying it in `BenchmarkConfig`
    """

    suite_name: str
    jar_path: str
    run_options: BenchmarkRunOptions | None
    benchmarks_config: list[BenchmarkConfig] | None

    def __init__(
        self, suite_name, jar_path, run_options=None, benchmarks_config=None
    ) -> None:
        self.suite_name = suite_name
        self.jar_path = jar_path
        self.run_options = (
            run_options if run_options is None else BenchmarkRunOptions(**run_options)
        )
        self.benchmarks_config = (
            [BenchmarkConfig(**config) for config in benchmarks_config]
            if benchmarks_config is not None
            else None
        )
        self.__post_init__()

    def __post_init__(self):
        validate_str(self.suite_name, self.__class__.__name__, "suite_name")
        validate_str(self.jar_path, self.__class__.__name__, "jar_path")

        allowed_chars = string.ascii_letters + string.digits + " -_"
        for c in self.suite_name:
            if c not in allowed_chars:
                raise ValueError(
                    f"Error: '{self.suite_name}' has invalid charactes: '{c}'. Allowed Characters: {allowed_chars}."
                )
        if self.benchmarks_config is not None and len(self.benchmarks_config) == 0:
            raise ValueError("Error: Invalid list of benchmarks_config provided.")
        names = {config.name for config in self.benchmarks_config or {}}
        if len(names) != len(self.benchmarks_config or {}):
            raise ValueError("Error: benchmark names should be unique.")

    def run(
        self, gc, heap_size, jdk, timeout
    ) -> tuple[list[BenchmarkReport], list[BenchmarkReport]]:
        """Runs a Benchmark Suite

        Args:
            gc ([TODO:parameter]): [TODO:description]
            heap_size ([TODO:parameter]): [TODO:description]
            jdk ([TODO:parameter]): [TODO:description]
            timeout ([TODO:parameter]): [TODO:description]

        Returns:
            (success_benchmarks: list[BenchmarkReport], failed_benchmarks: list[BenchmarkReport])
        """

        def handle_report(report: BenchmarkReport):
            nonlocal success_benchmarks
            nonlocal failed_benchmarks
            lst = success_benchmarks if report.is_successfull() else failed_benchmarks
            lst.append(report)

        success_benchmarks: list[BenchmarkReport] = []
        failed_benchmarks: list[BenchmarkReport] = []
        command = None
        java_opt = None
        exec_script = None
        if opt := self.run_options:
            timeout = opt.timeout if opt.timeout is not None else timeout
            command = opt.command
            java_opt = opt.extra_java_options
            exec_script = opt.post_exec_script

        if self.benchmarks_config is not None:
            for b in self.benchmarks_config:
                report = b.run()
                handle_report(report)
        else:
            report = run_benchmark(
                self.suite_name,
                self.suite_name,
                self.jar_path,
                gc,
                heap_size,
                jdk,
                timeout,
                command,
                java_opt,
                exec_script,
            )
            handle_report(report)

        return success_benchmarks, failed_benchmarks


@dataclass
class BenchmarkSuiteCollection:
    benchmark_suites: list[BenchmarkSuite]

    def __init__(self, benchmark_suites) -> None:
        self.benchmark_suites = [BenchmarkSuite(**suite) for suite in benchmark_suites]
        self.__post_init__()

    def __post_init__(self):
        if len(self.benchmark_suites) == 0:
            raise ValueError(
                "Error: Please provide at least one benchmark suite to test."
            )

    def run_benchmarks(
        self, jdk: str, gcs: list[str], timeout: int
    ) -> dict[str, dict[str, list[BenchmarkReport]]]:
        benchmark_reports: dict[str, dict[str, list[BenchmarkReport]]] = defaultdict(
            lambda: defaultdict(list)
        )
        heap_sizes: list[str] = utils.get_heap_sizes()

        failed_benchmarks: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for gc in gcs:
            for heap_size in heap_sizes:
                for suite in self.benchmark_suites:
                    success, failed = suite.run(gc, heap_size, jdk, timeout)

                    benchmark_reports[gc][heap_size].extend(success)

                    # NOTE: error is not None so ignore type checker
                    failed_benchmarks[gc][heap_size].extend(
                        [(r.garbage_collector, r.error) for r in failed]  # type: ignore
                    )
                # TODO: remove
                print("BREAKING")
                break
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

    @staticmethod
    def load_from_json(file_path: str) -> BenchmarkSuiteCollection:
        with open(file_path) as f:
            try:
                data = json.loads(f.read())
            except Exception as e:
                raise ValueError(f"Error: Couldn't read {file_path=}: {e}.")

            return BenchmarkSuiteCollection(**data)


def validate_str(value: Any, class_name: str, field_name: str):
    """Check if given value is a string with length > 0.
        Raises AssertionError with msg if it's not the case.

    Args:
        value: Value to validate
        field_name: Field being validated
    """
    err_str_msg = "Error: %s in %s should've a non null string-value, with one or more characters."
    assert value is not None and isinstance(value, str) and len(value) > 0, (
        err_str_msg % (field_name, class_name)
    )


def run_benchmark(
    suite_name: str,
    benchmark_name: str,
    jar_path: str,
    gc: str,
    heap_size: str,
    jdk: str,
    timeout: int,
    command: Optional[str],
    java_opt: Optional[str],
    exec_script: Optional[str],
) -> BenchmarkReport:
    """
    Args:
        timeout: int -> value in seconds to interrupt benchmark
    Returns:
        BenchmarkReport
    """

    def kill_process(
        process: subprocess.Popen[bytes],
        process_to_check: subprocess.Popen[bytes],
        cmd: list[str],
    ):
        print(f"Killing command: \n\t{' '.join(cmd)}\n")
        process.kill()
        process_to_check.kill()

    benchmark_command = _get_benchmark_command(
        suite_name, benchmark_name, jar_path, gc, heap_size, command, java_opt
    )

    print(
        f"[{suite_name}]: Running benchmark {benchmark_name} with:\n"
        f"\tGC: {gc}\n"
        f"\tHeap Size: {heap_size}\n"
        f"\tCommand: {' '.join(benchmark_command)}"
    )

    process = subprocess.Popen(
        benchmark_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # NOTE: When the post exec script ends we stop polling the main `process`
    process_to_check = process
    if exec_script is not None:
        print("Sleeping 10 seconds in order to give time for application to start")
        time.sleep(10)
        process_to_check = subprocess.Popen(
            exec_script.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    timer = Timer(timeout, kill_process, (process, process_to_check, benchmark_command))

    time_start = time.time_ns()
    pid = process.pid
    cpu_usage_stats = []
    cpu_time_stats = []
    io_time_stats = []

    print("Starting...")

    # TODO: see better polling method than sleeping 1 seconds for throughput
    timer.start()
    while process_to_check.poll() is None:
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
    throughput = time.time_ns() - time_start

    # NOTE: Cancel timer
    timer.cancel()

    cpu_usage_avg = round(float(numpy.mean(cpu_usage_stats)), 1)
    cpu_time_avg = round(float(numpy.mean(cpu_time_stats)), 1)
    io_time_avg = round(float(numpy.mean(io_time_stats)), 1)
    p90_io = float(round(numpy.percentile(io_time_stats, 90), 2))

    print(
        f"{cpu_usage_avg=} {cpu_time_avg=} {io_time_avg=} {p90_io=} and {throughput=}"
    )

    return_code = process_to_check.returncode
    print("Current return code", return_code)

    if process_to_check != process:
        assert (
            return_code == 0
        ), f"Script exited with code {return_code}. It should always be successfull"

    if process.poll() is not None:
        print("Main Application finished")
        return_code = process.returncode

    kill_process(process, process_to_check, benchmark_command)

    if return_code == 0:
        print("Success")
        result = BenchmarkReport.build_benchmark_result(
            gc,
            suite_name,
            benchmark_name,
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
        if process.stderr:
            error = process.stderr.read().decode()

        print(f"Error: {error}")
        result = BenchmarkReport.build_benchmark_error(
            gc,
            suite_name,
            benchmark_name,
            heap_size,
            cpu_usage_avg,
            cpu_time_avg,
            io_time_avg,
            p90_io,
            jdk,
            process.returncode,
            error,
        )
    result.save_to_json()

    return result


def _get_benchmark_command(
    suite_name: str,
    benchmark_name: str,
    jar_path: str,
    gc: str,
    heap_size: str,
    benchmark_cmd: Optional[str],
    java_opt: Optional[str],
) -> list[str]:
    command = [
        "java",
        f"-XX:+Use{gc}GC",
        f"-Xms{heap_size}m",
        f"-Xmx{heap_size}m",
        f"-Xlog:gc*,safepoint:file={utils.get_benchmark_log_path(gc, suite_name, benchmark_name, heap_size)}::filecount=0",
    ]

    if java_opt is not None:
        command.extend(java_opt.split(" "))

    command.extend(["-jar", jar_path])
    if benchmark_name is not None:
        command.append(benchmark_name)

    if benchmark_cmd is not None:
        command.extend(benchmark_cmd.split(" "))

    return command
