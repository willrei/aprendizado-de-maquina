from csv import DictReader
from pathlib import Path

from src.objects.DataInstance import DataInstance


def data_from_csv(filename: str) -> list[DataInstance]:
    """ Lê instâncias de uma base de dados armazenada em um arquivo csv. """
    # definição do caminho absoluto do arquivo
    base_path = Path(__file__).parent.parent.parent
    file_path = (base_path / f'./resources/{filename}').resolve()

    # leitura das instâncias
    data: list[DataInstance] = []
    with open(file_path, newline='') as csvfile:
        datareader: DictReader = DictReader(csvfile)
        entry: dict
        for entry in datareader:
            data.append(DataInstance(entry))
    return data
