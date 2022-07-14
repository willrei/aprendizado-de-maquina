from pandas import read_csv, DataFrame


def main() -> None:
    # leitura dos dados da base para um dataframe
    data: DataFrame = read_csv('data.csv')

    # item b do questionário
    p_acc: float = data['target'].value_counts()['acc'] / data.shape[0]  # P(target = acc)
    p_unacc: float = data['target'].value_counts()['unacc'] / data.shape[0]  # P(target = unacc)
    print(f'P(target = acc) = {p_acc}')
    print(f'P(target = unacc) = {p_unacc}')
    print(f'P(target = unacc) > P(target = acc)? {p_unacc > p_acc}', end='\n\n')

    # item c do questionário
    acc_data: DataFrame = data.query('target == "acc"')  # filtra os dados da classe acc
    pcond_price_med: float = acc_data['price'].value_counts()['med'] / acc_data.shape[0]  # P(price = med | target = acc)
    print(f'P(price = med | target = acc) = {pcond_price_med}')
    print(f'P(price = med | target = acc) == 0.3? {pcond_price_med == 0.3}', end='\n\n')

    # item d do questionário
    acc_prod: float = 1.0  # produtório de P(target = acc | x)
    unacc_prod: float = 1.0  # produtório de P(target = unacc | x)
    unacc_data: DataFrame = data.query('target == "unacc"')  # filtra os dados da classe unacc
    test_instance: dict = {'price': 'low', 'lug_boot': 'small', 'safety': 'high'}  # instância de teste x
    for attr, value in test_instance.items():
        acc_prod *= acc_data[attr].value_counts()[value] / acc_data.shape[0]  # prod * P(attr = value | target = acc)
        unacc_prod *= unacc_data[attr].value_counts()[value] / unacc_data.shape[0]  # prod * P(attr = value | target = unacc)
    ppost_acc: float = p_acc * acc_prod  # P(target = acc | x)
    ppost_unacc: float = p_unacc * unacc_prod  # P(target = unacc | x)
    print(f'P(target = acc | x) = {ppost_acc:.5f}')
    print(f'P(target = unacc | x) = {ppost_unacc:.5f}')
    print(f'P(target = acc | x) > P(target = unacc | x)? {ppost_acc > ppost_unacc}')
    return


if __name__ == '__main__':
    main()
