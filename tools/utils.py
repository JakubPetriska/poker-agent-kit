def flatten(*arg):
    return [item for sublist in arg for item in sublist]


def intersection(first, second):
    return set(first).intersection(set(second))