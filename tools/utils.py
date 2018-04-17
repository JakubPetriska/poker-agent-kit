import itertools


def flatten(*arg):
    return [item for sublist in arg for item in sublist]


def intersection(*args):
    result = set(args[0])
    for i in range(1, len(args)):
        result = result.intersection(set(args[i]))
    return result


def is_unique(*item_lists):
    all_items = flatten(*item_lists)
    previous = None
    for item in sorted(all_items):
        if item == previous:
            return False
        previous = item
    return True
