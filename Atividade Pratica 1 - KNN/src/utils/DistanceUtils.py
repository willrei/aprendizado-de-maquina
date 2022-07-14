from math import sqrt, fabs


def get_euclidean_dist(pairs: tuple[tuple[float, float]]) -> float:
    """ Calcula a distância euclidiana entre dois vetores de atributos """
    squared_diffs: list[float] = list(map(lambda pair: ((pair[0] - pair[1]) ** 2), pairs))
    return sqrt(sum(squared_diffs))


def get_manhattan_dist(pairs: tuple[tuple[float, float]]) -> float:
    """ Calcula a distância de manhattan entre dois vetores de atributos """
    absolute_diffs: list[float] = list(map(lambda pair: fabs(pair[0] - pair[1]), pairs))
    return sum(absolute_diffs)
