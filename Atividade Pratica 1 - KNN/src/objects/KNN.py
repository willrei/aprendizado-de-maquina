from src.enums.DistanceEnum import DistMethod
from src.objects.DataInstance import DataInstance


class KNN:
    """ Classe que implementa o algoritmo KNN para conjuntos com duas classes. """
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
        return 0 if (votes.count(0) > votes.count(1)) else 1

    def predict(self, new_instances: list[DataInstance], k: int,
                dist_method: DistMethod = DistMethod.EUCLIDEAN) -> list[int]:
        """ Prediz as classes de uma lista de instâncias. """
        predictions: list[int] = []
        for instance in new_instances:
            predictions.append(self._predict(instance, k, dist_method))
        return predictions
