from random import randint

from src.utils import FileUtils
from src.KNN import KNN, DistMethod


class TestInstance:
    def __init__(self, filename: str, training_prop: float = 0.8, normalize: bool = True) -> None:
        self.data = FileUtils.data_from_csv(filename)
        if normalize:
            self._normalize_data()
        self._split_data(training_prop)

    def _split_data(self, training_prop: float) -> None:
        """ Divide a base de dados em duas partes, uma para treino e outra para teste com estratificação. """
        # particiona os dados com base em suas classes (targets)
        targets = set([instance.target for instance in self.data])
        partitioned_data: list[list] = []
        for target in targets:
            filtered_data = list(filter(lambda instance: instance.target == target, self.data))
            partitioned_data.append(filtered_data)

        # monta o conjunto de teste com base na proporção de treino
        self.testing_data = []
        for partition in partitioned_data:
            partition_len = len(partition)
            testing_len = int(partition_len * training_prop)
            while len(partition) > testing_len:
                self.testing_data.append(partition.pop(randint(0, len(partition) - 1)))

        # monta o conjunto de treino com as instâncias restantes
        self.training_data = []
        for partition in partitioned_data:
            self.training_data.extend(partition)
        return None

    def _normalize_data(self) -> None:
        """ Normaliza os dados da base de teste. """
        # encontra os valores máximos e mínimos de cada atributo
        attribs_arr = [instance.attribs for instance in self.data]
        zipped_attribs = tuple(zip(*attribs_arr))
        max_arr = [max(attribs) for attribs in zipped_attribs]
        min_arr = [min(attribs) for attribs in zipped_attribs]

        # normaliza os atributos utilizando min-max [0,1]
        for attribs in attribs_arr:
            for index, attrib in enumerate(attribs):
                attribs[index] = (attrib - min_arr[index]) / (max_arr[index] - min_arr[index])

        # armazena os atributos normalizados nos dados
        for index, attribs in enumerate(attribs_arr):
            self.data[index].attribs = attribs
        return None

    def run(self, k: int, dist_method: DistMethod = DistMethod.EUCLIDEAN) -> None:
        """ Executa uma rodada de teste do algoritmo. """
        KNN(self.training_data).predict(self.testing_data, k, dist_method)
        return None
