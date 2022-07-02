from csv import DictReader
from math import sqrt, fabs


class DataInstance:
    """ Armazena as informações de uma instância da base de dados """
    def __init__(self, instance: dict):
        self.index: int = int(instance.pop(''))
        self.target: int = int(instance.pop('target'))
        self.attribs: list[float] = [float(value) for value in instance.values()]
        self.distance: float = float('inf')


def euclidian_dist(pairs: tuple[tuple[float, float]]) -> float:
    """ Calcula a distância euclidiana entre dois vetores de atributos """
    squared_diffs: list[float] = list(map(lambda pair: ((pair[0] - pair[1]) ** 2), pairs))
    return sqrt(sum(squared_diffs))


def manhattan_dist(pairs: tuple[tuple[float, float]]) -> float:
    """ Calcula a distância de manhattan entre dois vetores de atributos """
    absolute_diffs: list[float] = list(map(lambda pair: fabs(pair[0] - pair[1]), pairs))
    return sum(absolute_diffs)


def main() -> None:

    # leitura das instâncias armazenadas no arquivo csv
    instances: list[DataInstance] = []
    with open('data.csv', newline='') as csvfile:
        datareader: DictReader = DictReader(csvfile)
        entry: dict
        for entry in datareader:
            instances.append(DataInstance(entry))

    # cálculo das distâncias entre os dados de treinamento e uma instância de teste
    # utilização da distância de manhattan
    test_instance = instances[123]
    instance: DataInstance
    for instance in instances:
        zipped_pairs: zip = zip(instance.attribs, test_instance.attribs)
        pairs: tuple[tuple[float, float]] = tuple(zipped_pairs)
        instance.distance = manhattan_dist(pairs)

    # cálculo das distâncias entre os dados de treinamento e uma instância de teste
    # utilização da distância euclidiana
    test_instance = instances[123]
    instance: DataInstance
    for instance in instances:
        zipped_pairs: zip = zip(instance.attribs, test_instance.attribs)
        pairs: tuple[tuple[float, float]] = tuple(zipped_pairs)
        instance.distance = euclidian_dist(pairs)

    # ordenação das instâncias de acordo com a distância em ordem crescente
    instances.sort(key=lambda inst: inst.distance)

    # escolha dos k vizinhos mais próximos
    k: int = 13
    nearest: list[DataInstance] = instances[:k]

    # definição da classe da instância de teste de acordo com a votação
    votes: list[int] = list(map(lambda inst: inst.target, nearest))
    test_instance.target = 0 if (votes.count(0) > votes.count(1)) else 1

    # impressão dos resultados
    print(f'\nk: {k}\nvotes: ', end='')
    [print(vote, end=' ') for vote in votes]
    print(f'\nvotes for 0: {votes.count(0)}\nvotes for 1: {votes.count(1)}')
    print(f'test instance class: {test_instance.target}')

    return


if __name__ == '__main__':
    main()
