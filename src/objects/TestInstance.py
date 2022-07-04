from random import randint

from src.utils import FileUtils
from src.KNN import KNN, DistMethod


class TestInstance:
    def __init__(self, filename: str, training_prop: float = 0.7, normalize: bool = True) -> None:
        self.training_data = FileUtils.data_from_csv(filename)
        if normalize:
            self._normalize_data()
        self._split_data(training_prop)

    def _split_data(self, training_prop: float) -> None:
        """ Divide a base de dados em duas partes, uma para treino e outra para teste. """
        # TODO: implementar a estratificação
        self.testing_data = []
        original_len = len(self.training_data)
        testing_len = int(original_len * training_prop)
        while len(self.training_data) > testing_len:
            self.testing_data.append(self.training_data.pop(randint(0, len(self.training_data) - 1)))
        return None

    def _normalize_data(self) -> None:
        """ Normaliza os dados da base de teste. """
        pass

    def run(self, k: int, dist_method: DistMethod = DistMethod.EUCLIDEAN) -> None:
        """ Executa uma rodada de teste do algoritmo. """
        KNN(self.training_data).predict(self.testing_data, k, dist_method)
        return None
