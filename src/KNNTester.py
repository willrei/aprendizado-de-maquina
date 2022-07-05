from src.KNN import DistMethod
from src.objects.TestInstance import TestInstance


def main() -> None:
    # criação de uma instância de teste a partir dos dados contidos no arquivo csv
    # a instanciação padrão considera 80% da base (~80 instâncias) para treino
    test = TestInstance('data.csv')
    test.export_min_max()  # exporta os valores mínimos e máximos de cada atributo para o arquivo de saída
    test.run_many([1, 3, 5, 7], DistMethod.EUCLIDEAN)  # executa o algoritmo KNN para os valores de k especificados
    test.normalize_data()  # normaliza os dados da base de teste
    test.run_many([1, 3, 5, 7], DistMethod.EUCLIDEAN)

    # criação de uma nova instância de teste com dados não normalizados
    # essa seção foi comentada por demorar demais para ser executada
    # test = TestInstance('data.csv')
    # test.run_repeat(1, 50, DistMethod.EUCLIDEAN)  # executa 50 vezes com conjuntos de treino e teste diferentes
    # test.run_repeat(3, 50, DistMethod.EUCLIDEAN)
    # test.run_repeat(5, 50, DistMethod.EUCLIDEAN)
    # test.run_repeat(7, 50, DistMethod.EUCLIDEAN)
    # test.normalize_data()
    # test.run_repeat(1, 50, DistMethod.EUCLIDEAN)
    # test.run_repeat(3, 50, DistMethod.EUCLIDEAN)
    # test.run_repeat(5, 50, DistMethod.EUCLIDEAN)
    # test.run_repeat(7, 50, DistMethod.EUCLIDEAN)
    return None


if __name__ == '__main__':
    main()
