from typing import Union, List, Tuple, Callable, Iterable
from line_profiler import LineProfiler

def general_test(funcs: List[Callable], *args, **kwargs)->None:
    tests = LineProfiler()
    for func in funcs:
        tests.add_function(func)
        tests.runcall(func, *args, **kwargs)
    tests.print_stats() 