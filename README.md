# BenchmarkGC

## Description

BenchmarkGC is a tool with the aim of benchmarking any given Java application, collecting metrics, and scoring all available Java Garbage Collectors
with respect to execution time and GC pause time.

## Requirements

- Python == 3.12

## How to use

1. Ensure you have Python 3.12 installed.
2. Install the dependencies required with the command:
```
pip install -r requirements.txt
```
3. Have a benchmark suite config with the applications you desire to benchmark ready. 
See [./benchmarks_config_example.json](./benchmarks_config_example.json) for an example containing
a config for [Renaissance](https://renaissance.dev/) and [DaCapo](https://www.dacapobench.org/) or [Benchmark Config](#benchmark-config) 
for more details on the Benchmark Config format.

4. Run the tool with:
```
python main.py --config benchmarks_config_example.json
```
Remember to substitute the `benchmarks_config_example.json` file with your 
actual file.


> [!NOTE]
> To see additional functionality, run:  
> ```
> python main.py --help
> ```

## Development

To develop, additionally to the application dependencies you should also install
the Development dependencies with:
```
pip install -r requirements-dev.txt
```
## Benchmark Config

A benchmark config is a `json` file with the following properties:

- **`benchmark_suites`**: An array of benchmark suite objects, each defining a collection of benchmarks to run.
  - **`suite_name`**: The name of the benchmark suite (e.g., DaCapo, Renaissance).
  - **`jar_path`**: The path to the JAR file containing the benchmark applications.
  - **`run_options`** (optional): An object containing additional runtime options for the benchmark, such as:
    - **`command`** (optional): A string specifying the general command-line options for the suite.
    - **`java_options`** (optional): Java-specific options (e.g., JVM flags).
    - **`post_exec_script`** (optional): A script to execute after the benchmark application starts. 
    This could be used to send requests to your application for example.
    - **`timeout`** (optional): Specifies the time (in seconds) allowed for the benchmark to run. 
    If the benchmark exceeds this time it will be terminated.
  - **`benchmarks_config`** (optional): An array of individual benchmark configurations within the suite, each with:
    - **`name`**: The name of the specific benchmark.
    - **`run_options`** (optional): Same as above. This will override the global benchmark suite options.

