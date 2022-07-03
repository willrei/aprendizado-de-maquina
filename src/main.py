from csv import DictReader
from pathlib import Path
from DataInstance import DataInstance
from KNearest import KNearest, DistMethod


def main() -> None:
    # leitura das instâncias armazenadas no arquivo csv
    data: list[DataInstance] = []
    base_path = Path(__file__).parent
    file_path = (base_path / "../resources/data.csv").resolve()
    with open(file_path, newline='') as csvfile:
        datareader: DictReader = DictReader(csvfile)
        entry: dict
        for entry in datareader:
            data.append(DataInstance(entry))

    # utilização do algoritmo com k = 13 e distância euclidiana
    knn: KNearest = KNearest(13, data)
    knn.predict_single(data[13], DistMethod.EUCLIDEAN)

    # utilização do algoritmo com k = 13 e distância de manhattan
    knn.predict_single(data[13], DistMethod.MANHATTAN)

    # utilização do algoritmo com k = 13 para várias instâncias e distância euclidiana
    knn.predict(data[:15], DistMethod.EUCLIDEAN)
    return


if __name__ == '__main__':
    main()
