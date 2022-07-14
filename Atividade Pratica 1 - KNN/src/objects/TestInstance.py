from __future__ import annotations

from random import randint
from src.utils import FileUtils
from src.objects.KNN import KNN, DistMethod


class TestInstance:
    """ Representa uma instância de teste para o algoritmo KNN."""
    def __init__(self, filename: str, training_prop: float = 0.8):
        self.training_prop: float = training_prop
        self.data = FileUtils.data_from_csv(filename)
        self._get_min_max()
        self._split_data()
        self._normalized: bool = False

    def _get_min_max(self) -> None:
        """ Encontra os valores máximos e mínimos de cada atributo da base de dados. """
        attribs_arr = [instance.attribs for instance in self.data]
        zipped_attribs = tuple(zip(*attribs_arr))
        self.max_arr = [max(attribs) for attribs in zipped_attribs]
        self.min_arr = [min(attribs) for attribs in zipped_attribs]
        return None

    def _split_data(self) -> None:
        """ Divide a base de dados em duas partes, uma para treino e outra para teste com estratificação. """
        # particiona os dados com base em suas classes (targets)
        targets = set([instance.target for instance in self.data])
        partitioned_data: list[list] = []
        for target in targets:
            filtered_data = list(filter(lambda instance: instance.target == target, self.data))
            partitioned_data.append(filtered_data)

        # monta o conjunto de teste com base na proporção de treino para cada partição
        self.testing_data = []
        for partition in partitioned_data:
            partition_len = len(partition)
            testing_len = int(partition_len * self.training_prop)
            while len(partition) > testing_len:
                self.testing_data.append(partition.pop(randint(0, len(partition) - 1)))

        # monta o conjunto de treino com as instâncias restantes
        self.training_data = []
        for partition in partitioned_data:
            self.training_data.extend(partition)
        return None

    def _normalize(self, data: list) -> None:
        """ Normaliza os dados de uma lista de instâncias utilizando min-max [0,1]. """
        for instance in data:
            for index, attrib in enumerate(instance.attribs):
                instance.attribs[index] = (attrib - self.min_arr[index]) / (self.max_arr[index] - self.min_arr[index])
        self._normalized = True
        return None

    def normalize_data(self) -> TestInstance:
        """ Normaliza os dados da base de teste. """
        self._normalize(self.training_data)
        self._normalize(self.testing_data)
        return self

    def run(self, k: int, dist_method: DistMethod = DistMethod.EUCLIDEAN, filename: str = 'output.txt') -> float:
        """ Executa uma rodada de teste do algoritmo. """
        # execução das predições com a utilização do knn
        predictions: list[int] = KNN(self.training_data).predict(self.testing_data, k, dist_method)

        # cálculo da acurácia do algoritmo na rodada de teste
        values: list[int] = [instance.target for instance in self.testing_data]
        comparison: tuple[tuple[int, int]] = tuple(zip(predictions, values))
        right_count: int = len(list(filter(lambda pair: pair[0] == pair[1], comparison)))
        accuracy: float = right_count / len(self.testing_data)
        return accuracy

    def run_many(self, k_list: list[int], dist_method: DistMethod = DistMethod.EUCLIDEAN,
                 filename: str = 'output.txt') -> None:
        """ Executa várias rodadas de teste do algoritmo. """
        for k in k_list:
            accuracy = self.run(k, dist_method, filename)
            string: str = f'k = {k}, normalized = {self._normalized}, method = {dist_method.name}, ' \
                          f'accuracy = {accuracy} [single run]\n'
            FileUtils.write_to_text(filename, [string])
        return None

    def run_repeat(self, k: int, repeats: int, dist_method: DistMethod = DistMethod.EUCLIDEAN,
                   filename: str = 'output.txt') -> None:
        """ Executa várias rodadas com os mesmos parâmetros e diferentes conjuntos de dados. """
        # coleta da acurácia para cada rodada do algoritmo
        accuracy_list: list[float] = []
        for _ in range(repeats):
            self._split_data()
            accuracy_list.append(self.run(k, dist_method))
        accuracy_mean = sum(accuracy_list) / repeats

        # escrita da acurácia média para cada rodada do algoritmo
        string: str = f'k = {k}, normalized = {self._normalized}, method = {dist_method.name}, ' \
                      f'accuracy mean = {accuracy_mean} [{repeats} repeats]\n'
        FileUtils.write_to_text(filename, [string])
        return None

    def export_min_max(self, filename: str = 'output.txt') -> None:
        """ Exporta os valores mínimos e máximos de cada atributo da base de dados. """
        FileUtils.write_to_text(filename, [f'min: {str(self.min_arr)}\n', f'max: {str(self.max_arr)}\n'])
        return None
