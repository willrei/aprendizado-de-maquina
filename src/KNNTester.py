from src.KNN import DistMethod
from src.objects.TestInstance import TestInstance


def main() -> None:
    # criação de uma instância de teste a partir dos dados contidos no arquivo csv
    # a instanciação padrão considera 70% da base (70 instâncias) para treino
    test = TestInstance('smalldata.csv')

    # utilização do algoritmo com k = 13 e distância euclidiana
    test.run(13, DistMethod.EUCLIDEAN)

    # utilização do algoritmo com k = 7 e distância de manhattan
    test.run(7, DistMethod.MANHATTAN)

    # utilização do algoritmo com k = 5, distância euclidiana e 50% da base para treino
    TestInstance('smalldata.csv', 0.5).run(5, DistMethod.EUCLIDEAN)
    return None


if __name__ == '__main__':
    main()
