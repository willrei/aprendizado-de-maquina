from src.objects.KNN import KNN, DistMethod
from src.utils import FileUtils


def main() -> None:
    # leitura das instâncias armazenadas no arquivo csv
    data = FileUtils.data_from_csv('smalldata.csv')

    knn: KNN = KNN(data)
    # utilização do algoritmo com k = 13 e distância euclidiana
    knn.predict([data[13]], 13, DistMethod.EUCLIDEAN)

    # utilização do algoritmo com k = 7 e distância de manhattan
    knn.predict([data[13]], 7, DistMethod.MANHATTAN)

    # utilização do algoritmo com k = 3 para várias instâncias e distância euclidiana
    knn.predict(data[:15], 3, DistMethod.EUCLIDEAN)
    return


if __name__ == '__main__':
    main()
