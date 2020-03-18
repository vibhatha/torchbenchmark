from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Type, Union, cast


def clock_cycles(m: int, n: int) -> Iterable[List[Tuple[int, int]]]:
    """Generates schedules for each clock cycle."""
    # m: number of micro-batches
    # n: number of partitions
    # i: index of micro-batch
    # j: index of partition
    # k: clock number
    #
    # k (i,j) (i,j) (i,j)
    # - ----- ----- -----
    # 0 (0,0)
    # 1 (1,0) (0,1)
    # 2 (2,0) (1,1) (0,2)
    # 3       (2,1) (1,2)
    # 4             (2,2)
    for k in range(m+n-1):
        yield [(k-j, j) for j in range(max(1+k-m, 0), min(1+k, n))]

cv = clock_cycles(4,3)

for d in cv:
    print(d)