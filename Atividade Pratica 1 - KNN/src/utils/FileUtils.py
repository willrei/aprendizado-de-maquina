from csv import DictReader
from pathlib import Path

from src.objects.DataInstance import DataInstance


basepath = Path(__file__).parent.parent.parent


def data_from_csv(filename: str) -> list[DataInstance]:
    """ Lê instâncias de uma base de dados armazenada em um arquivo csv. """
    # definição do caminho absoluto do arquivo
    filepath = (basepath / f'./resources/{filename}').resolve()

    # leitura das instâncias da base de dados
    with open(filepath, newline='') as csvfile:
        datareader: DictReader = DictReader(csvfile)
        return [DataInstance(row) for row in datareader]


def write_to_text(filename: str, data: list[str]) -> None:
    """ Escreve strings em um arquivo txt. """
    # definição do caminho absoluto do arquivo
    filepath = (basepath / f'./output/{filename}').resolve()

    # escrita das strings no arquivo
    with open(filepath, 'a') as file:
        file.writelines(data)
    return None
