import unittest
from unittest.mock import mock_open, patch

from benchmark import (
    BenchmarkRunOptions,
    BenchmarkSuiteCollection,
    BenchmarkConfig,
)


class TestBenchmarkConfig(unittest.TestCase):
    def test_invalid_benchmark_suites(self):
        tests = {
            "Invalid JSON": (
                "",
                ValueError,
                "Error: Couldn't read file_path='test.json': Expecting value: line 1 column 1 (char 0).",
            ),
            "Non Existing": (
                "{}",
                TypeError,
                "BenchmarkSuiteCollection.__init__() missing 1 required positional argument: 'benchmark_suites'",
            ),
            "String Value": (
                '{"benchmark_suites": "test"}',
                TypeError,
                "benchmark_config.BenchmarkSuite() argument after ** must be a mapping, not str",
            ),
            "Empty Suites": (
                '{"benchmark_suites": {}}',
                ValueError,
                "Error: Please provide at least one benchmark suite to test.",
            ),
            "Invalid Benchmark Suites": (
                '{"benchmark_suites": {"test": "test"}}',
                TypeError,
                "benchmark_config.BenchmarkSuite() argument after ** must be a mapping, not str",
            ),
            "Invalid Benchmark Suites 2": (
                '{"benchmark_suites": [{"test": "test"}]}',
                TypeError,
                "BenchmarkSuite.__init__() got an unexpected keyword argument 'test'",
            ),
            "Missing suite_name in BenchmarkSuite": (
                '{"benchmark_suites": [{"jar_path": "a"}]}',
                TypeError,
                "BenchmarkSuite.__init__() missing 1 required positional argument: 'suite_name'",
            ),
            "Empty suite_name in BenchmarkSuite": (
                '{"benchmark_suites": [{"suite_name": "", "jar_path": "a"}]}',
                AssertionError,
                "Error: suite_name in BenchmarkSuite should've a non null string-value, with one or more characters.",
            ),
            "Wrong type suite_name in BenchmarkSuite": (
                '{"benchmark_suites": [{"suite_name": 5, "jar_path": "a"}]}',
                AssertionError,
                "Error: suite_name in BenchmarkSuite should've a non null string-value, with one or more characters.",
            ),
            "Missing jar_path in BenchmarkSuite": (
                '{"benchmark_suites": [{"suite_name": "a"}]}',
                TypeError,
                "BenchmarkSuite.__init__() missing 1 required positional argument: 'jar_path'",
            ),
            "Empty jar_path in BenchmarkSuite": (
                '{"benchmark_suites": [{"jar_path": "", "suite_name": "a"}]}',
                AssertionError,
                "Error: jar_path in BenchmarkSuite should've a non null string-value, with one or more characters.",
            ),
            "Wrong type jar_path in BenchmarkSuite": (
                '{"benchmark_suites": [{"jar_path": 5, "suite_name": "a"}]}',
                AssertionError,
                "Error: jar_path in BenchmarkSuite should've a non null string-value, with one or more characters.",
            ),
        }

        self.run_tests_with_exceptions(tests)

    def test_invalid_run_options(self):
        tests = {
            "Invalid run_options": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "run_options": ""}]}',
                TypeError,
                "benchmark_config.BenchmarkRunOptions() argument after ** must be a mapping, not str",
            ),
            "Invalid run_options type command": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "run_options": {"command": 5}}]}',
                AssertionError,
                "Error: command with type <class 'int'> should have type in (<class 'str'>, <class 'NoneType'>).",
            ),
            "Invalid run_options type java options": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "run_options": {"java_options": 5}}]}',
                AssertionError,
                "Error: java_options with type <class 'int'> should have type in (<class 'str'>, <class 'NoneType'>).",
            ),
            "Invalid run_options type post exec script": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "run_options": {"post_exec_script": 5}}]}',
                AssertionError,
                "Error: post_exec_script with type <class 'int'> should have type in (<class 'str'>, <class 'NoneType'>).",
            ),
            "Invalid run_options type timeout": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "run_options": {"timeout": "5"}}]}',
                AssertionError,
                "Error: timeout with type <class 'str'> should have type in (<class 'int'>, <class 'NoneType'>).",
            ),
        }
        self.run_tests_with_exceptions(tests)

    def test_valid_run_options(self):
        tests = {
            "Empty run options": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "run_options": {}}]}',
                BenchmarkRunOptions(),
            )
        }

        for name, (data, expected) in tests.items():
            with patch("builtins.open", mock_open(read_data=data)):
                s = BenchmarkSuiteCollection.load_from_json("test.json")
                self.assertIsNotNone(s)
                b = s.benchmark_suites[0]
                self.assertEqual(
                    b.run_options,
                    expected,
                    f"Fail: Test {name} got value {b.run_options} instead of {expected}",
                )

    def test_invalid_benchmark_config(self):
        tests = {
            "Invalid benchmark config str": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "benchmarks_config": ""}]}',
                ValueError,
                "Error: Invalid list of benchmarks_config provided.",
            ),
            "Invalid benchmark config int": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "benchmarks_config": 5}]}',
                TypeError,
                "'int' object is not iterable",
            ),
            "Invalid benchmark config list with dict": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "benchmarks_config": {}}]}',
                ValueError,
                "Error: Invalid list of benchmarks_config provided.",
            ),
            "Invalid benchmark config list empty": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "benchmarks_config": []}]}',
                ValueError,
                "Error: Invalid list of benchmarks_config provided.",
            ),
            "Invalid benchmark config list with dict and values": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "benchmarks_config": {"test": "test"}}]}',
                TypeError,
                "benchmark_config.BenchmarkConfig() argument after ** must be a mapping, not str",
            ),
            "Invalid benchmark config list with some values": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "benchmarks_config": [{"test": "test"}]}]}',
                TypeError,
                "BenchmarkConfig.__init__() got an unexpected keyword argument 'test'",
            ),
            "Invalid benchmark config list with empty dict": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "benchmarks_config": [{"name": "a"}, {"name": "a"}]}]}',
                ValueError,
                "Error: benchmark names should be unique.",
            ),
        }
        self.run_tests_with_exceptions(tests)

    def test_valid_benchmark_config(self):
        tests = {
            "Benchmark Config": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "benchmarks_config": [{"name": "a"}]}]}',
                [BenchmarkConfig("a")],
            ),
            "Benchmark Configs": (
                '{"benchmark_suites": [{"jar_path": "5", "suite_name": "a", "benchmarks_config": [{"name": "a"}, {"name":"b"}]}]}',
                [BenchmarkConfig("a"), BenchmarkConfig("b")],
            ),
        }

        for name, (data, expected) in tests.items():
            with patch("builtins.open", mock_open(read_data=data)):
                s = BenchmarkSuiteCollection.load_from_json("test.json")
                self.assertIsNotNone(s)
                b = s.benchmark_suites[0]
                self.assertEqual(
                    b.benchmarks_config,
                    expected,
                    f"Fail: Test {name} got value {b.benchmarks_config} instead of {expected}",
                )

    def test_suite_name_valid_characters(self):
        data = '{"benchmark_suites": [{"suite_name": " _-azAZ19", "jar_path": "a"}]}'
        with patch("builtins.open", mock_open(read_data=data)):
            s = BenchmarkSuiteCollection.load_from_json("test.json")
            self.assertIsNotNone(s)
            b = s.benchmark_suites[0]
            self.assertEqual(b.suite_name, " _-azAZ19")

    def test_suite_name_invalid_characters(self):
        tests = {
            "Invalid characters .": (
                '{"benchmark_suites": [{"suite_name": ".", "jar_path": "a"}]}',
                ValueError,
                "Error: '.' has invalid characters: '.'. Allowed Characters: abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.",
            ),
            "Invalid characters /": (
                '{"benchmark_suites": [{"suite_name": "/", "jar_path": "a"}]}',
                ValueError,
                "Error: '/' has invalid characters: '/'. Allowed Characters: abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.",
            ),
            "Invalid characters \\": (
                '{"benchmark_suites": [{"suite_name": "\\\\", "jar_path": "a"}]}',
                ValueError,
                "Error: '\\' has invalid characters: '\\'. Allowed Characters: abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -_.",
            ),
        }

        self.run_tests_with_exceptions(tests)

    def run_tests_with_exceptions(self, tests):
        for name, (data, exception, msg_expected) in tests.items():
            with patch("builtins.open", mock_open(read_data=data)):
                with self.assertRaises(
                    exception,
                    msg=f'Failed: Test {name} should\'ve raised an {exception} error with msg="{msg_expected}"',
                ) as cm:
                    BenchmarkSuiteCollection.load_from_json("test.json")

                self.assertEqual(
                    str(cm.exception),
                    msg_expected,
                    f'Failed: Test {name} got message "{cm.exception}" instead of "{msg_expected}"',
                )


if __name__ == "__main__":
    # with patch("builtins.print"):
    unittest.main()
