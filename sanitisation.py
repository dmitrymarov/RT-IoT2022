import argparse

import numpy as np
import pandas as pd


def dataframeCategorySanitisation(df):
    for column in df.columns:
        # Получаем первый элемент в столбце
        first_entry = df[column].iloc[0]
        columns_one_hot_encode = []

        # Если первый элемент - строка, выполняем one-hot encoding
        if isinstance(first_entry, str):
            if column != "Attack_type":
                columns_one_hot_encode.append(column)
            else:
                # Создаем маппинг атак к числовым классам
                attack_mapping = {
                    "MQTT_Publish": 0,
                    "Thing_Speak": 0,
                    "Wipro_bulb": 0,
                    "Amazon-Alexa": 0,
                    "DOS_SYN_Hping": 1,
                    "ARP_poisioning": 2,
                    "NMAP_UDP_SCAN": 3,
                    "NMAP_XMAS_TREE_SCAN": 4,
                    "NMAP_OS_DETECTION": 5,
                    "NMAP_TCP_scan": 6,
                    "DDOS_Slowloris": 7,
                    "Metasploit_Brute_Force_SSH": 8,
                    "NMAP_FIN_SCAN": 9,
                }
                df[column] = df[column].map(attack_mapping)
                df = df.rename(columns={column: "class"})
        # One-hot encoding для других категориальных столбцов
        if columns_one_hot_encode:
            df = pd.get_dummies(df, columns=columns_one_hot_encode)
    return df


def dataframeNumberSanitisation(df):
    for column in df.columns:
        # Получаем первый элемент в столбце
        first_entry = df[column].iloc[0]

        # Если первый элемент - число, нормализуем данные
        if isinstance(first_entry, (float, int, np.int64, np.float64)):
            if column != "class":  # Не нормализуем целевой столбец
                df[column] = (df[column] - df[column].min()) / (
                    df[column].max() - df[column].min()
                )
    return df


def main():
    desc = "Очистка данных в формате CSV перед обучением модели."

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-i", "--input", help="Входной CSV файл.")
    parser.add_argument("-o", "--output", help="Имя выходного CSV файла.")

    # Чтение аргументов командной строки
    args = parser.parse_args()
    input_file_name = args.input
    output_file_name = args.output

    # Импорт CSV файла в DataFrame
    original_data_df = pd.read_csv(input_file_name)

    # Удаление ненужных столбцов (если есть)
    if "Unnamed: 0" in original_data_df.columns:
        original_data_df = original_data_df.drop("Unnamed: 0", axis=1)

    # Нормализация числовых данных
    original_data_df = dataframeNumberSanitisation(original_data_df)

    # One-hot encoding категориальных данных
    original_data_df = dataframeCategorySanitisation(original_data_df)

    # Замена пустых значений на нули
    original_data_df.fillna(0, inplace=True)

    print(f"Количество признаков после предобработки: {original_data_df.shape[1]}")

    # Сохранение обработанных данных
    original_data_df.to_csv(output_file_name, index=False)


if __name__ == "__main__":
    main()
