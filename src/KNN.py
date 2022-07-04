from src.enums.DistanceEnum import DistMethod
from src.objects.DataInstance import DataInstance


class KNN:
    """ Classe que implementa o algoritmo KNN. """
    def __init__(self, instances: list[DataInstance]):
        self.instances: list[DataInstance] = instances

    def _predict(self, new_instance: DataInstance, k: int, dist_method: DistMethod) -> int:
        """ Prediz a classe de uma instância. """
        # cálculo das distâncias para cada instância no conjunto de dados
        for instance in self.instances:
            zipped_pairs: zip = zip(instance.attribs, new_instance.attribs)
            pairs: tuple[tuple[float, float]] = tuple(zipped_pairs)
            instance.distance = dist_method.value(pairs)

        # ordenação das instâncias pelo valor da distância
        sorted_instances = sorted(self.instances, key=lambda inst: inst.distance)

        # votação da classe com base nos k vizinhos mais próximos
        nearest: list[DataInstance] = sorted_instances[:k]
        votes: list[int] = list(map(lambda inst: inst.target, nearest))
        new_instance.target = 0 if (votes.count(0) > votes.count(1)) else 1

        # impressão dos resultados
        # TODO: remover
        print(f'\nk: {k}\ninstance index: {new_instance.index}')
        print(f'method: {dist_method.name.lower()}\nvotes: ', end='')
        [print(vote, end=' ') for vote in votes]
        print(f'\nvotes for 0: {votes.count(0)}\nvotes for 1: {votes.count(1)}')
        print(f'prediction: {new_instance.target}')
        return new_instance.target

    def train(self, instances: list[DataInstance]) -> None:
        """ Treina o algoritmo com um conjunto de dados. """
        self.instances: list[DataInstance] = instances
        return

    def predict(self, new_instances: list[DataInstance], k: int,
                dist_method: DistMethod = DistMethod.EUCLIDEAN) -> list[tuple[int, int]]:
        """ Prediz as classes de uma lista de instâncias. """
        predictions: list[tuple[int, int]] = []
        for instance in new_instances:
            predictions.append((instance.index, self._predict(instance, k, dist_method)))

        # impressão dos resultados
        # TODO: remover
        print(f'\nk: {k}\nmethod: {dist_method.name.lower()}\npredictions:')
        [print(f'{prediction}', end=' ') for prediction in predictions]
        print()
        return predictions
