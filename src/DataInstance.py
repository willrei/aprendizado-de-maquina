class DataInstance:
    """ Armazena as informações de uma instância da base de dados """
    def __init__(self, instance: dict) -> None:
        self.index: int = int(instance.pop(''))
        self.target: int = int(instance.pop('target'))
        self.attribs: list[float] = [float(value) for value in instance.values()]
        self.distance: float = float('inf')
        return
