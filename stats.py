def mean(args: list):
    """Calcul the mean"""
    if len(args) != 0:
        return sum(args) / len(args)
    return 0


def median(args: list):
    """Calcul the median"""
    after_sort = sorted(args)
    n = len(after_sort)
    if (n % 2 == 0 and n != 0):
        mid1 = n // 2 - 1
        mid2 = n // 2
        return (after_sort[mid1] + after_sort[mid2]) / 2
    elif (n != 0):
        return (after_sort[int(len(after_sort) / 2)])
    else:
        return 0


def quartile(args: list):
    lst = sorted(args)
    n = len(lst)
    if n == 0:
        return [0, 0]

    Q1_pos = 0.25 * (n - 1)
    Q3_pos = 0.75 * (n - 1)

    Q1_int = int(Q1_pos)
    Q1_frac = Q1_pos - Q1_int
    Q1 = lst[Q1_int] + Q1_frac * (lst[Q1_int + 1] - lst[Q1_int])

    Q3_int = int(Q3_pos)
    Q3_frac = Q3_pos - Q3_int
    Q3 = lst[Q3_int] + Q3_frac * (lst[Q3_int + 1] - lst[Q3_int])

    return [Q1, Q3]


def std(args: list):
    """Calcul the standard deviation"""
    return variance(args)**0.5


def variance(args: list):
    """Calcul the variance"""
    mean_value = mean(args)
    return sum((i - mean_value) ** 2 for i in args) / (len(args) - 1)


def mininum(args):
    sort = sorted(args)
    return sort[0]


def maximum(args):
    sort = sorted(args)
    return sort[-1]


def ft_statistics(*args: any, **kwargs: any) -> None:
    """Handle different type of operations"""
    operations = {
        'mean': mean,
        'median': median,
        'quartile': quartile,
        'std': std,
        'var': variance,
        'min': mininum,
        'max': maximum
    }

    result = []
    for key, value in kwargs.items():
        if value in operations and args:
            result.append(operations[value](args[0]))
        elif not args:
            print("ERROR")
    return result
