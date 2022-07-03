from enum import Enum
from math import sqrt, fabs
from functools import partial
from DataInstance import DataInstance


def euclidean_dist(pairs: tuple[tuple[float, float]]) -> float:
    """ Calcula a distância euclidiana entre dois vetores de atributos """
    squared_diffs: list[float] = list(map(lambda pair: ((pair[0] - pair[1]) ** 2), pairs))
    return sqrt(sum(squared_diffs))


def manhattan_dist(pairs: tuple[tuple[float, float]]) -> float:
    """ Calcula a distância de manhattan entre dois vetores de atributos """
    absolute_diffs: list[float] = list(map(lambda pair: fabs(pair[0] - pair[1]), pairs))
    return sum(absolute_diffs)


class DistMethod(Enum):
    EUCLIDEAN = partial(euclidean_dist)
    MANHATTAN = partial(manhattan_dist)


class KNearest:
    """ Classe que implementa o algoritmo KNN """
    def __init__(self, k: int, instances: list[DataInstance]) -> None:
        self.instances: list[DataInstance] = instances
        self.k: int = k
        return

    def train(self, instances: list[DataInstance]) -> None:
        """ Treina o algoritmo com um conjunto de dados """
        self.instances: list[DataInstance] = instances
        return

    def predict_single(self, new_instance: DataInstance, dist_method: DistMethod = DistMethod.EUCLIDEAN) -> int:
        """ Prediz a classe de uma instância """
        # cálculo das distâncias para cada instância no conjunto de dados
        for instance in self.instances:
            zipped_pairs: zip = zip(instance.attribs, new_instance.attribs)
            pairs: tuple[tuple[float, float]] = tuple(zipped_pairs)
            instance.distance = dist_method.value(pairs)

        # ordenação das instâncias pelo valor da distância
        self.instances.sort(key=lambda inst: inst.distance)

        # votação da classe com base nos k vizinhos mais próximos
        nearest: list[DataInstance] = self.instances[:self.k]
        votes: list[int] = list(map(lambda inst: inst.target, nearest))
        new_instance.target = 0 if (votes.count(0) > votes.count(1)) else 1

        # impressão dos resultados
        print(f'\nk: {self.k}\ninstance index: {new_instance.index}')
        print(f'method: {dist_method.name.lower()}\nvotes: ', end='')
        [print(vote, end=' ') for vote in votes]
        print(f'\nvotes for 0: {votes.count(0)}\nvotes for 1: {votes.count(1)}')
        print(f'prediction: {new_instance.target}')
        return new_instance.target

    def predict(self, new_instances: list[DataInstance],
                dist_method: DistMethod = DistMethod.EUCLIDEAN) -> list[tuple[int, int]]:
        """ Prediz as classes de uma lista de instâncias """
        # predição das classes para cada instância
        predictions: list[tuple[int, int]] = []
        for instance in new_instances:
            predictions.append((instance.index, self.predict_single(instance, dist_method)))

        # impressão dos resultados
        print(f'\nk: {self.k}\nmethod: {dist_method.name.lower()}\npredictions:')
        [print(f'{prediction}', end=' ') for prediction in predictions]
        print()
        return predictions
