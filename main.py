#!/usr/bin/env python

import argparse

from benchmark_config import BenchmarkConfig, BenchmarkSuiteCollection
import utils
import benchmark
from model import (
    StatsMatrix,
)
from typing import Optional


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compute throughput and average pause time for benchmarks"
    )
    parser.add_argument(
        "-d",
        "--debug",
        dest="debug",
        action="store_true",
        help="Enable benchmark debugging.",
    )
    parser.add_argument(
        "-c",
        "--clean",
        dest="clean",
        action="store_true",
        help="Clean the benchmark stats.",
    )
    parser.add_argument(
        "-s",
        "--skip-benchmarks",
        dest="skip_benchmarks",
        action="store_true",
        help="""Skip the benchmarks and compute the matrix with previously obtained garbage collector results.
        You must specify the java jdk used to obtain previous results. The jdk version is present in the name of each benchmark stat file. See `--jdk`.""",
    )

    parser.add_argument(
        "-j",
        "--jdk",
        dest="jdk",
        help="Specify the java jdk version when you wish to skip the benchmarks and only calculate the matrix.",
    )

    parser.add_argument(
        "-n",
        "--number-iterations",
        dest="iterations",
        default=10,
        type=int,
        help="Number of iterations to run benchmarks. Increase this number to achieve more reliable metrics.",
    )

    parser.add_argument(
        "-t",
        "--timeout",
        dest="timeout",
        default=600,
        type=int,
        help="Timeout in seconds for each benchmark.",
    )

    parser.add_argument(
        "-b",
        "--benchmarks",
        dest="benchmarks",
        choices=[group.value for group in benchmark.BENCHMARK_GROUP],
        help="Specify the group of benchmarks to run",
        nargs="+",
    )

    parser.add_argument(
        "--config",
        dest="config",
        help="Specify a benchmark_config file to run",
    )

    args = parser.parse_args(argv)
    parser.print_help()
    skip_benchmarks = args.skip_benchmarks
    iterations: int = args.iterations
    clean = args.clean
    jdk: Optional[str] = args.jdk
    benchmarks = args.benchmarks and [
        benchmark.BENCHMARK_GROUP(el) for el in args.benchmarks
    ]
    timeout: int = args.timeout
    debug = args.debug
    config = args.config

    benchmark.set_debug(debug)

    print(f"{debug=}")
    print(benchmarks)

    # Always clean benchmark garbage collection logs
    utils.clean_logs()

    if clean:
        utils.clean_stats()
        if skip_benchmarks:
            print("Cleaned and skipped benchmarks")
            return 0

    if skip_benchmarks:
        assert jdk is not None, (
            "Please provide the jdk in order to load previously obtained results."
        )
        garbage_collectors = utils.get_garbage_collectors()
    else:
        jdk, garbage_collectors = utils.get_java_env_info()

    assert jdk is not None and garbage_collectors is not None, (
        "Please make sure you have Java installed on your system."
    )

    if config is not None:
        c = BenchmarkSuiteCollection.load_from_json(config)
        benchmark_reports = c.run_benchmarks(jdk, garbage_collectors, timeout)
    else:
        benchmark_reports = benchmark.run_benchmarks(
            iterations,
            jdk,
            garbage_collectors,
            skip_benchmarks,
            benchmarks,
            timeout,
        )

    if len(benchmark_reports) == 0:
        print("No GarbageCollector had successful benchmarks.")
        return 0

    matrix = StatsMatrix.build_stats_matrix(benchmark_reports, "G1")
    matrix.save_to_json(jdk)
    return 0


if __name__ == "__main__":
    exit(main())
