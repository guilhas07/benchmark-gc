from __future__ import annotations
from collections import defaultdict

import datetime
import json
import re
from dataclasses import asdict, dataclass
from typing import Optional

import numpy

import utils


CPU_THRESHOLD = 60


@dataclass
class StatsMatrix:
    @dataclass
    class GCScore:
        throughput: float
        pause_time: float

    matrix: dict[
        str, dict[str, GCScore]
    ]  # {heap_size: { gc : {throughput, pause_time }}}

    garbage_collectors: list[str]
    # { heap_size : list[garbage_collectors]}}
    benchmarks: dict[str, list[str]]

    @staticmethod
    def build_stats_matrix(
        benchmark_reports: dict[str, dict[str, list[BenchmarkReport]]],
        default_gc: str,
    ) -> StatsMatrix:
        """Builds a stats matrix

        Args:
            benchmark_reports: dict[garbage_collector, dict[heap_size, list[BenchmarkReport]]]
            default_garbage_collector: The default garbage collector to compare the others to

        Returns:
            A `StatsMatrix`
        """

        def normalize_matrix(
            matrix: dict[str, dict[str, StatsMatrix.GCScore]],
        ):
            for heap_size in matrix:
                default_throughput = matrix[heap_size][default_gc].throughput
                default_pause_time = matrix[heap_size][default_gc].pause_time
                for gc in matrix[heap_size]:
                    matrix[heap_size][gc].throughput = round(
                        matrix[heap_size][gc].throughput / default_throughput, 2
                    )
                    matrix[heap_size][gc].pause_time = round(
                        matrix[heap_size][gc].pause_time / default_pause_time, 2
                    )

        found = False
        garbage_collectors = set()
        for gc in benchmark_reports:
            assert (
                gc not in garbage_collectors
            ), "Results should be of distinct garbage collectors."
            garbage_collectors.add(gc)
            if gc == default_gc:
                found = True

        assert (
            found
        ), f"default_garbage_collector: {default_gc} was not found in benchmark_reports"

        # Populate list of benchmark names with the benchmarks present in
        # the default garbage collector
        default_gc_reports: dict[str, list[str]] = defaultdict(list)

        default_jdk = None
        for heap_size, report_list in benchmark_reports[default_gc].items():
            for report in report_list:
                if default_jdk is None:
                    default_jdk = report_list[0].jdk

                assert (
                    report.jdk == default_jdk
                ), "All benchmark reports should have the same runtime"

                assert (
                    report.benchmark_name not in default_gc_reports[heap_size]
                ), f"Default garbage collector {default_gc} shouldn't have repeated reports for the same benchmark {report.benchmark_name}"

                default_gc_reports[heap_size].append(report.benchmark_name)

        matrix: dict[str, dict[str, StatsMatrix.GCScore]] = defaultdict(
            lambda: defaultdict(lambda: StatsMatrix.GCScore(0, 0))
        )

        for gc in benchmark_reports:
            for heap_size, reports_list in benchmark_reports[gc].items():
                assert (
                    heap_size in default_gc_reports
                ), "All garbage collectors should have stats for the same heap sizes"

                assert (
                    len(reports_list) == len(default_gc_reports[heap_size])
                ), "All garbage collectors should have the same number of benchmark reports"

                gc_benchmarks = {i.benchmark_name for i in reports_list}

                assert gc_benchmarks == set(
                    default_gc_reports[heap_size]
                ), "All garbage collectors should have the same benchmarks"

                for report in reports_list:
                    assert (
                        report.jdk == default_jdk
                    ), "All benchmark reports should be from the same runtime"

                    matrix[heap_size][gc].throughput += report.throughput
                    matrix[heap_size][gc].pause_time += report.p90_pause_time

        normalize_matrix(matrix)
        return StatsMatrix(
            matrix,
            list(garbage_collectors),
            default_gc_reports,
        )

    def save_to_json(self, jdk: str):
        date = datetime.datetime.now()
        day = date.day
        month = date.month
        hour = date.hour
        minute = date.minute

        with open(
            utils.get_matrix_path(jdk, f"{day}_{month}_{hour}_{minute}"),
            "w",
        ) as f:
            f.write(json.dumps(asdict(self), indent=4))


@dataclass
class GarbageCollectorReport:
    garbage_collector: str
    jdk: str
    stats: list[GarbageCollectorStats]

    @dataclass
    class GarbageCollectorStats:
        heap_size: str
        number_of_pauses: int
        total_pause_time: int
        avg_pause_time: float
        p90_avg_pause_time: float
        avg_throughput: float
        benchmarks: list[str]

    @staticmethod
    def _build_gc_stats(
        heap_size: str,
        benchmark_reports: list[BenchmarkReport],
    ) -> GarbageCollectorReport.GarbageCollectorStats:
        """
        Args:
            heap_size: int -> size of the heap used for the list of benchmark results.
            benchmark_reports: list[BenchmarkResult] -> Valid benchmark results (meaning only successfull benchmarks), with atleast one element.
            All benchmark results should have an heap_size equal to the provided heap_size and should be from the same garbage collector.
            You can check the validity of the benchmark with benchmark_result.is_successfull().

        Returns:
            stats: GarbageCollectorStats
        """
        total_gc_pauses: int = 0
        total_gc_pause_time: int = 0
        p90_gc_pause_time = []
        gc_throughput = []
        benchmarks = []

        for result in benchmark_reports:
            total_gc_pauses += result.number_of_pauses
            total_gc_pause_time += result.total_pause_time
            p90_gc_pause_time.append(result.p90_pause_time)
            gc_throughput.append(result.throughput)

            assert (
                result.benchmark_name not in benchmarks
            ), "There shouldn't be repeated benchmarks"
            benchmarks.append(result.benchmark_name)

        benchmarks.sort()
        return GarbageCollectorReport.GarbageCollectorStats(
            heap_size,
            total_gc_pauses,
            total_gc_pause_time,
            round(total_gc_pause_time / total_gc_pauses, 2),
            float(round(numpy.mean(p90_gc_pause_time), 2)),
            float(round(numpy.mean(gc_throughput), 2)),
            benchmarks,
        )

    @staticmethod
    def build_garbage_collector_report(
        benchmarks_results: dict[str, list[BenchmarkReport]],
    ) -> GarbageCollectorReport:
        """Builds a garbage collector report, taking into account every successfull benchmark run for every heap_size.

        Args:
            benchmarks_results: Dictionary with `heap_size` as key and value as the list of `BenchmarkResult`s for that `heap_size`.
                                NOTE: Every list should have one or more elements.

        Returns:
            A `GarbageCollectorResult`
        """
        stats = []

        assert (
            len(benchmarks_results) > 0
        ), "benchmark_reports should have one or more keys"

        gc = ""
        jdk = ""
        for heap_size, results in benchmarks_results.items():
            assert (
                len(results) >= 0
            ), "List of benchmark results should have one element or more"

            if gc == "" and jdk == "":
                gc = results[0].garbage_collector
                jdk = results[0].jdk

            assert all(
                el.is_successfull() for el in results
            ), "All benchmarks should be successfull"

            assert all(
                el.heap_size == heap_size for el in results
            ), f"All benchmark results should have the same heap size {heap_size}"

            assert all(
                el.is_successfull() for el in results
            ), "All benchmark results should be successfull"

            assert all(
                el.garbage_collector == gc for el in results
            ), "All benchmark results should be from the same garbage collector"

            assert all(
                el.jdk == jdk for el in results
            ), "All benchmark results should be from the same runtime"
            stats.append(GarbageCollectorReport._build_gc_stats(heap_size, results))

        return GarbageCollectorReport(gc, jdk, stats)

    @staticmethod
    def load_from_json(file_path: str) -> GarbageCollectorReport:
        with open(file_path) as f:
            data = f.read()
            print(f"{json.loads(data)}")
            result = GarbageCollectorReport(**json.loads(data))

            # NOTE: Convert dicts to GarbageCollectorStats
            result.stats = [
                GarbageCollectorReport.GarbageCollectorStats(**el)  # type: ignore
                for el in result.stats
            ]

            return result

    def save_to_json(self):
        with open(
            utils.get_gc_stats_path(self.garbage_collector, self.jdk),
            "w",
        ) as f:
            f.write(json.dumps(asdict(self), indent=4))


@dataclass
class BenchmarkReport:
    garbage_collector: str
    jdk: str
    benchmark_group: str
    benchmark_name: str
    heap_size: str
    error: Optional[str]
    avg_cpu_usage: float
    avg_cpu_time: float
    avg_io_time: float
    p90_io: float
    number_of_pauses: int
    total_pause_time: int
    avg_pause_time: float
    pauses_per_category: dict[str, int]
    total_pause_time_per_category: dict[str, int]
    avg_pause_time_per_category: dict[str, float]
    p90_pause_time: float
    throughput: int

    @staticmethod
    def load_from_json(file_path: str) -> BenchmarkReport:
        with open(file_path) as f:
            return BenchmarkReport(**json.loads(f.read()))

    @staticmethod
    def build_benchmark_error(
        gc: str,
        benchmark_group: str,
        benchmark: str,
        heap_size: str,
        average_cpu_usage: float,
        average_cpu_time: float,
        average_io_time: float,
        p90_io: float,
        jdk: str,
        error_code: int,
        error_message: str,
    ) -> BenchmarkReport:
        def error_code_message(error_code: int, error_message: str) -> str:
            error_codes = {-9: "Process was killed due to timeout", -1: "Out of memory"}
            if error_code in error_codes:
                return f"Return code {error_code}: {error_codes[error_code]}\n{error_message}"
            return f"Error code not found: {error_code}\n{error_message}"

        return BenchmarkReport(
            gc,
            jdk,
            benchmark_group,
            benchmark,
            heap_size,
            error_code_message(error_code, error_message),
            average_cpu_usage,
            average_cpu_time,
            average_io_time,
            p90_io,
            0,
            0,
            0,
            {},
            {},
            {},
            0,
            0,
        )

    @staticmethod
    def build_benchmark_result(
        gc: str,
        benchmark_group: str,
        benchmark_name: str,
        heap_size: str,
        average_cpu_usage: float,
        average_cpu_time: float,
        average_io_time: float,
        p90_io: float,
        throughput: int,
        jdk: str,
    ) -> BenchmarkReport:
        log_file = utils.get_benchmark_log_path(
            gc, benchmark_group, benchmark_name, heap_size
        )
        number_of_pauses = 0

        total_pause_time = 0
        avg_pause_time = 0.0
        pauses_per_category = {}
        total_pause_time_per_category = {}
        avg_pause_time_per_category = {}
        pause_times = []

        with open(log_file) as f:
            for line in f:
                if "safepoint" in line:
                    # NOTE: use () to capture group
                    pause_category = re.findall('Safepoint "(.*)"', line)[0]
                    pause_time = int(re.findall("Total: (\\d+) ns", line)[0])
                    pause_times.append(pause_time)

                    if pause_category not in pauses_per_category:
                        pauses_per_category[pause_category] = 0
                        total_pause_time_per_category[pause_category] = 0

                    pauses_per_category[pause_category] += 1
                    total_pause_time_per_category[pause_category] += pause_time
                    total_pause_time += pause_time
                    number_of_pauses += 1

        p90_pause_time = float(round(numpy.percentile(pause_times, 90), 2))
        avg_pause_time = float(round(total_pause_time / number_of_pauses, 2))

        for category, pause_time in total_pause_time_per_category.items():
            avg_pause_time_per_category[category] = round(
                pause_time / pauses_per_category[category], 2
            )

        return BenchmarkReport(
            gc,
            jdk,
            benchmark_group,
            benchmark_name,
            heap_size,
            None,
            average_cpu_usage,
            average_cpu_time,
            average_io_time,
            p90_io,
            number_of_pauses,
            total_pause_time,
            avg_pause_time,
            pauses_per_category,
            total_pause_time_per_category,
            avg_pause_time_per_category,
            p90_pause_time,
            throughput,
        )

    def is_successfull(self):
        return self.error is None

    def save_to_json(self):
        with open(
            utils.get_benchmark_stats_path(
                self.garbage_collector,
                self.benchmark_group,
                self.benchmark_name,
                self.heap_size,
                self.jdk,
                self.error,
            ),
            "w",
        ) as f:
            f.write(json.dumps(asdict(self), indent=4))


@dataclass
class ErrorReport:
    jdk: str
    failed_benchmarks: dict[str, dict[str, list[tuple[str, str]]]]

    def save_to_json(self):
        date = datetime.datetime.now()
        day = date.day
        month = date.month
        hour = date.hour
        minute = date.minute
        with open(
            utils.get_error_report_path(self.jdk, f"{day}_{month}_{hour}_{minute}"),
            "w",
        ) as f:
            f.write(json.dumps(asdict(self), indent=4))
