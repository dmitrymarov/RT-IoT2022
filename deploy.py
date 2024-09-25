# Импорт библиотеки для работы с данными
import pandas as pd

# Импорт основной библиотеки PyTorch
import torch

# Импорт модуля нейронных сетей PyTorch
import torch.nn as nn


# Определение архитектуры нейронной сети
class NeuralNet(nn.Module):
    # Инициализация класса
    def __init__(self, input_size, hidden_size, num_classes):
        # Вызов конструктора родительского класса
        super(NeuralNet, self).__init__()
        # Определение первого полносвязного слоя
        self.layer1 = nn.Linear(input_size, hidden_size)
        # Определение функции активации ReLU
        self.relu = nn.ReLU()
        # Определение второго полносвязного слоя
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        # Определение выходного слоя
        self.layer3 = nn.Linear(hidden_size, num_classes)

    # Метод прямого распространения
    def forward(self, x):
        # Применение первого слоя
        out = self.layer1(x)
        # Применение функции активации ReLU
        out = self.relu(out)
        # Применение второго слоя
        out = self.layer2(out)
        # Применение функции активации ReLU
        out = self.relu(out)
        # Применение выходного слоя
        out = self.layer3(out)
        # Возвращение результата
        return out


# Функция для загрузки модели
def load_model(model_path, input_size, hidden_size, num_classes):
    # Создание экземпляра модели
    model = NeuralNet(input_size, hidden_size, num_classes)
    # Загрузка весов модели
    model.load_state_dict(torch.load(model_path))
    # Перевод модели в режим оценки
    model.eval()
    # Возвращение загруженной модели
    return model


# Функция для получения предсказаний
def predict(model, data):
    # Отключение вычисления градиентов
    with torch.no_grad():
        # Получение выходных данных модели
        outputs = model(data)
        # Получение индексов максимальных значений (предсказаний)
        _, predicted = torch.max(outputs.data, 1)
    # Возвращение предсказаний
    return predicted


# Основная функция
def main():
    # Загрузка новых данных для предсказания
    df = pd.read_csv(
        "RT-IoT2022-Neural-Network\data\RT_IOT2022_sanitized.csv"
    )  # Замените на ваш файл с новыми данными
    # Получение списка булевых столбцов
    bool_columns = df.select_dtypes(include=["bool"]).columns
    # Преобразование булевых столбцов в целочисленные
    df[bool_columns] = df[bool_columns].astype(int)
    # Извлечение признаков из DataFrame
    features = df.values
    # Преобразование признаков в тензор PyTorch
    features = torch.tensor(features, dtype=torch.float32)

    # Определение параметров модели
    input_size = features.shape[1]
    hidden_size = 128
    num_classes = 10  # Убедитесь, что соответствует количеству классов в вашей задаче

    # Загрузка модели
    model = load_model("trained_model.pth", input_size, hidden_size, num_classes)

    # Получение предсказаний
    predictions = predict(model, features)
    # Вывод предсказаний
    print("Предсказания:", predictions.numpy())


# Проверка, является ли скрипт основным файлом
if __name__ == "__main__":
    # Вызов основной функции
    main()
