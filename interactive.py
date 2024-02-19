from enum import Enum
from typing import Any, Callable, Iterable
import subprocess
from threading import Timer

from benchmark import (
    BENCHMARK_GROUP,
    _get_benchmarks,
    run_benchmark,
)
from utils import get_heap_sizes


class RETURN_CODE(Enum):
    SUCCESS = 0
    ERROR = 1
    BACK = 2
    QUIT = 3


Command_Return = tuple[RETURN_CODE, dict]
Command = Callable[..., Command_Return]
Command_Args = Iterable[Any]
Option = tuple[str, Command, Command_Args]
Menu = list[Option]


def print_menu(name: str, cmds: Menu):
    print(f"[Menu] {name}:")
    for i in range(len(cmds)):
        print(f"[{i}]: {cmds[i][0]}")


def is_valid_option(option: int, length: int) -> bool:
    if option < 0 or option >= length:
        print("Please choose a valid option.")
        return False
    else:
        return True


def read_option() -> int:
    return int(a) if (a := input("> ")) and a.isdigit() else -1


def timeout_menu() -> Menu:
    def get_timeout():
        timeout = read_option()
        if timeout <= 0:
            return (RETURN_CODE.ERROR, {})
        return (RETURN_CODE.SUCCESS, {"timeout": timeout})

    commands: Menu = [
        (
            "Choose Timeout",
            get_timeout,
            [],
        )
    ]

    commands.extend(
        [
            ("None", lambda: (RETURN_CODE.SUCCESS, {"timeout": None}), []),
            ("Back", lambda: (RETURN_CODE.BACK, {}), []),
            ("Quit", lambda: (RETURN_CODE.QUIT, {}), []),
        ]
    )
    return commands


def iterations_menu() -> Menu:
    def get_iterations():
        iterations = read_option()
        if iterations <= 0:
            return (RETURN_CODE.ERROR, {})
        return (RETURN_CODE.SUCCESS, {"iterations": iterations})

    menu: Menu = [
        (
            "Choose Iterations",
            get_iterations,
            [],
        )
    ]

    menu.extend(
        [
            ("Back", lambda: (RETURN_CODE.BACK, {}), []),
            ("Quit", lambda: (RETURN_CODE.QUIT, {}), []),
        ]
    )
    return menu


def aux_benchmark_group(group: BENCHMARK_GROUP) -> Command_Return:
    menu = []
    benchmarks_choosen = []

    d = {"benchmark_group": group}

    def append_benchmarks(benchmark) -> tuple[RETURN_CODE, dict]:
        nonlocal benchmarks_choosen
        benchmarks_choosen.append(benchmark)
        # NOTE: return error so it keeps choosing
        return (RETURN_CODE.ERROR, {})

    benchmarks = _get_benchmarks(group)
    for b in benchmarks:
        option: Option = (b, append_benchmarks, [b])
        menu.append(option)

    menu.extend(
        [
            (
                "Done",
                lambda: (
                    RETURN_CODE.SUCCESS,
                    {"benchmarks": benchmarks_choosen},
                ),
                [],
            ),
            (
                "All",
                lambda: (
                    RETURN_CODE.SUCCESS,
                    {"benchmarks": benchmarks},
                ),
                [],
            ),
            ("Back", lambda: (RETURN_CODE.BACK, {}), []),
            ("Quit", lambda: (RETURN_CODE.QUIT, {}), []),
        ]
    )
    while True:
        print_menu("Choose Benchmarks:", menu)
        i = read_option()
        if is_valid_option(i, len(menu)):
            _, handler, args = menu[i]
            code, kv = handler(*args)
            match code:
                case RETURN_CODE.ERROR:
                    pass
                case RETURN_CODE.SUCCESS:
                    d.update(kv)
                    return (RETURN_CODE.SUCCESS, d)
                case RETURN_CODE.BACK:
                    # to retry benchmark_group
                    return (RETURN_CODE.ERROR, {})
                case RETURN_CODE.QUIT:
                    return (RETURN_CODE.QUIT, {})


def heap_size_menu() -> Menu:
    heap_sizes_choosen = []
    heap_sizes = get_heap_sizes()

    def append_heap_size(heap_size) -> tuple[RETURN_CODE, dict]:
        nonlocal heap_sizes
        heap_sizes_choosen.append(heap_size)
        # NOTE: return error so it keeps choosing
        return (RETURN_CODE.ERROR, {})

    menu: Menu = []
    for heap_size in heap_sizes:
        option: Option = (
            f"Run with {heap_size}m",
            append_heap_size,
            [heap_size],
        )
        menu.append(option)

    menu.extend(
        [
            (
                "Done",
                lambda: (
                    RETURN_CODE.SUCCESS,
                    {"heap_sizes": heap_sizes_choosen},
                ),
                [],
            ),
            (
                "All",
                lambda: (
                    RETURN_CODE.SUCCESS,
                    {"heap_sizes": heap_sizes},
                ),
                [],
            ),
            ("Back", lambda: (RETURN_CODE.BACK, {}), []),
            ("Quit", lambda: (RETURN_CODE.QUIT, {}), []),
        ]
    )
    return menu


def garbage_collectors_menu(garbage_collectors: list[str]) -> Menu:
    gcs = []

    def append_gc(gc) -> tuple[RETURN_CODE, dict]:
        nonlocal gcs
        gcs.append(gc)
        # NOTE: return error so it keeps choosing
        return (RETURN_CODE.ERROR, {})

    menu: Menu = []
    for gc in garbage_collectors:
        option: Option = (
            f"Run with {gc}",
            append_gc,
            [gc],
        )
        menu.append(option)

    menu.extend(
        [
            (
                "Done",
                lambda: (
                    RETURN_CODE.SUCCESS,
                    {"garbage_collectors": gcs},
                ),
                [],
            ),
            (
                "All",
                lambda: (
                    RETURN_CODE.SUCCESS,
                    {"garbage_collectors": garbage_collectors},
                ),
                [],
            ),
            ("Back", lambda: (RETURN_CODE.BACK, {}), []),
            ("Quit", lambda: (RETURN_CODE.QUIT, {}), []),
        ]
    )
    return menu


def benchmark_group_menu() -> Menu:
    menu: Menu = []
    for group in [*BENCHMARK_GROUP]:
        option: Option = (
            f"Run {group.value}",
            aux_benchmark_group,
            [group],
        )
        menu.append(option)

    menu.extend(
        [
            ("Back", lambda: (RETURN_CODE.BACK, {}), []),
            ("Quit", lambda: (RETURN_CODE.QUIT, {}), []),
        ]
    )
    return menu


def benchmark_suite_cmd(jdk, garbage_collectors) -> tuple[RETURN_CODE, dict]:
    menus: list[Menu] = []
    current_menu = 0
    command_values: dict = {
        "jdk": jdk,
    }

    menu_fns = [
        [benchmark_group_menu, []],
        [iterations_menu, []],
        [timeout_menu, []],
        [garbage_collectors_menu, [garbage_collectors]],
        [heap_size_menu, []],
    ]
    for fun in menu_fns:
        menus.append(fun[0](*fun[1]))

    while True:
        commands = menus[current_menu]
        print_menu(
            f"Benchmark Suites [{current_menu+1}/{len(menus)}]", menus[current_menu]
        )

        option = read_option()
        if is_valid_option(option, len(commands)):
            _, handler, args = commands[option]
            code, kv = handler(*args)
            match code:
                # leave to parent menu
                case RETURN_CODE.BACK if current_menu == 0:
                    return RETURN_CODE.BACK, {}
                # go to previous sub-menu
                case RETURN_CODE.BACK:
                    # NOTE: resetting menu function
                    fn = menu_fns[current_menu]
                    menus[current_menu] = fn[0](*fn[1])
                    current_menu -= 1

                # leave application
                case RETURN_CODE.QUIT:
                    return RETURN_CODE.QUIT, {}
                # if last sub-menu: run command
                case RETURN_CODE.SUCCESS if current_menu == len(menus) - 1:
                    command_values.update(kv)
                    print(f"{command_values=}")
                    values = command_values.copy()
                    del values["garbage_collectors"]
                    del values["benchmarks"]
                    del values["heap_sizes"]
                    print("Starting to run")
                    p = None
                    for gc in command_values["garbage_collectors"]:
                        print(f"{gc=}")
                        for benchmark in command_values["benchmarks"]:
                            print(f"{benchmark=}")
                            for heap_size in command_values["heap_sizes"]:
                                print(f"{heap_size=}")
                                run_benchmark(
                                    **values,
                                    heap_size=heap_size,
                                    benchmark=benchmark,
                                    gc=gc,
                                )
                                if p is not None:
                                    p.kill()
                                p = subprocess.Popen(["paplay", "notification.mp3"])
                                Timer(5, p.kill).start()

                    return RETURN_CODE.SUCCESS, {}
                # go to next sub-menu
                case RETURN_CODE.SUCCESS:
                    command_values.update(kv)
                    current_menu += 1
                # Retry
                case RETURN_CODE.ERROR:
                    pass


def cmd_new_application() -> tuple[RETURN_CODE, dict]:
    return RETURN_CODE.SUCCESS, {}


def run(jdk, garbage_collectors):
    print("Welcome to Benchmark GCs interactive mode!")
    options: Menu = [
        (
            "Run Benchmark Suites (all or specific benchmark)",
            benchmark_suite_cmd,
            [jdk, garbage_collectors],
        ),
        ("Run New Application", cmd_new_application, []),
        ("Quit", lambda: (RETURN_CODE.QUIT, {}), []),
    ]

    while True:
        print_menu("Main", options)
        option = read_option()
        if is_valid_option(option, len(options)):
            _, handler, args = options[option]
            code, _ = handler(*args)
            match code:
                case RETURN_CODE.QUIT:
                    return
                case RETURN_CODE.SUCCESS | RETURN_CODE.ERROR | RETURN_CODE.BACK:
                    pass
