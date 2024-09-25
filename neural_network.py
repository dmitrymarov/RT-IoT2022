# Импортирование библиотеки для создания графиков
import matplotlib.pyplot as plt

# Импортирование библиотеки для работы с данными
import pandas as pd

# Импортирование библиотеки для создания статистических графиков
import seaborn as sns

# Импортирование основной библиотеки PyTorch
import torch

# Импортирование модуля нейронных сетей PyTorch
import torch.nn as nn

# Импортирование функций для оценки модели машинного обучения
from sklearn.metrics import classification_report, confusion_matrix

# Импортирование функции для разделения данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split

# Импортирование классов для работы с данными в PyTorch
from torch.utils.data import DataLoader, Dataset

# Определение устройства для вычислений (GPU, если доступно, иначе CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------------------------
# Вывод в консоль сделан на английском так как из-за
# проблем с кодировками у меня не отображаются русские символы,
# с чем это конкретно связано я не знаю, проще все выводить на английском)
# -----------------------------------------------


# Определение пользовательского класса для работы с данными
class CustomDataset(Dataset):
    # Инициализация класса
    def __init__(self, dataframe):
        # Сохранение признаков (без столбца 'class')
        self.features = dataframe.drop("class", axis=1).values
        # Сохранение меток классов
        self.labels = dataframe["class"].values

    # Метод для получения длины датасета
    def __len__(self):
        # Возвращает количество образцов
        return len(self.labels)

    # Метод для получения отдельного элемента датасета
    def __getitem__(self, idx):
        # Преобразование признаков в тензор PyTorch
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        # Преобразование метки в тензор PyTorch
        target = torch.tensor(self.labels[idx], dtype=torch.long)
        # Возвращение пары (признаки, метка)
        return features, target


# Определение класса нейронной сети
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


# Определение основной функции программы
def main():
    # Загрузка данных из CSV файла
    df = pd.read_csv("RT-IoT2022-Neural-Network\data\RT_IOT2022_sanitize.csv")
    # Получение списка булевых столбцов
    bool_columns = df.select_dtypes(include=["bool"]).columns
    # Преобразование булевых столбцов в целочисленные
    df[bool_columns] = df[bool_columns].astype(int)

    # Разделение данных на признаки и метки
    X = df.drop("class", axis=1)
    y = df["class"]

    # Разделение данных на обучающую и тестовую выборки
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Разделение обучающей выборки на обучающую и валидационную
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=42,
        stratify=y_train_full,
    )

    # Создание объекта CustomDataset для обучающей выборки
    train_dataset = CustomDataset(pd.concat([X_train, y_train], axis=1))
    # Создание объекта CustomDataset для валидационной выборки
    val_dataset = CustomDataset(pd.concat([X_val, y_val], axis=1))
    # Создание объекта CustomDataset для тестовой выборки
    test_dataset = CustomDataset(pd.concat([X_test, y_test], axis=1))

    # Определение размера батча
    batch_size = 64

    # Создание DataLoader для обучающей выборки
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # Создание DataLoader для валидационной выборки
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # Создание DataLoader для тестовой выборки
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Определение размерности входного слоя
    input_size = X_train.shape[1]
    # Определение размера скрытого слоя
    hidden_size = 128
    # Определение количества классов
    num_classes = df["class"].nunique()

    # Создание экземпляра модели и перемещение ее на выбранное устройство
    model = NeuralNet(input_size, hidden_size, num_classes).to(device)
    # Определение функции потерь
    criterion = nn.CrossEntropyLoss()
    # Определение оптимизатора
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Определение количества эпох обучения
    num_epochs = 50

    # Инициализация списков для хранения значений потерь и точности
    training_losses = []
    validation_losses = []
    training_accuracies = []
    validation_accuracies = []

    # Цикл обучения по эпохам
    for epoch in range(num_epochs):
        # Перевод модели в режим обучения
        model.train()
        # Инициализация переменной для накопления потерь
        running_loss = 0.0
        # Инициализация счетчика правильных предсказаний
        correct_predictions = 0
        # Инициализация счетчика всех предсказаний
        total_predictions = 0

        # Цикл по батчам данных
        for features, labels in train_loader:
            # Перемещение данных на выбранное устройство
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            # Вычисление функции потерь
            loss = criterion(outputs, labels)

            # Обнуление градиентов
            optimizer.zero_grad()
            # Обратное распространение ошибки
            loss.backward()
            # Обновление весов
            optimizer.step()

            # Накопление значения функции потерь
            running_loss += loss.item() * features.size(0)
            # Получение предсказанных меток
            _, predicted = torch.max(outputs.data, 1)
            # Увеличение счетчика всех предсказаний
            total_predictions += labels.size(0)
            # Увеличение счетчика правильных предсказаний
            correct_predictions += (predicted == labels).sum().item()

        # Вычисление средней потери за эпоху
        epoch_loss = running_loss / len(train_dataset)
        # Вычисление точности за эпоху
        epoch_acc = 100 * correct_predictions / total_predictions
        # Добавление значения потери в список
        training_losses.append(epoch_loss)
        # Добавление значения точности в список
        training_accuracies.append(epoch_acc)

        # Перевод модели в режим оценки
        model.eval()
        # Инициализация переменной для накопления потерь на валидационной выборке
        val_running_loss = 0.0
        # Инициализация счетчика правильных предсказаний на валидационной выборке
        val_correct_predictions = 0
        # Инициализация счетчика всех предсказаний на валидационной выборке
        val_total_predictions = 0

        # Отключение вычисления градиентов
        with torch.no_grad():
            # Цикл по батчам валидационных данных
            for val_features, val_labels in val_loader:
                # Перемещение данных на выбранное устройство
                val_features = val_features.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_features)

                val_loss = criterion(val_outputs, val_labels)
                # Накопление значения функции потерь
                val_running_loss += val_loss.item() * val_features.size(0)
                # Получение предсказанных меток
                _, val_predicted = torch.max(val_outputs.data, 1)
                # Увеличение счетчика всех предсказаний
                val_total_predictions += val_labels.size(0)
                # Увеличение счетчика правильных предсказаний
                val_correct_predictions += (val_predicted == val_labels).sum().item()

        # Вычисление средней потери на валидационной выборке
        val_epoch_loss = val_running_loss / len(val_dataset)
        # Вычисление точности на валидационной выборке
        val_epoch_acc = 100 * val_correct_predictions / val_total_predictions
        # Добавление значения потери в список
        validation_losses.append(val_epoch_loss)
        # Добавление значения точности в список
        validation_accuracies.append(val_epoch_acc)

        # Вывод метрик текущей эпохи
        print(
            f"Epoch [{epoch+1}/{num_epochs}], "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
            f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%"
        )

    # Создание нового графика
    plt.figure(figsize=(10, 5))
    # Построение графика потерь на обучающей выборке
    plt.plot(range(1, num_epochs + 1), training_losses, label="Training Loss")
    # Построение графика потерь на валидационной выборке
    plt.plot(range(1, num_epochs + 1), validation_losses, label="Validation Loss")
    # Установка подписи оси X
    plt.xlabel("Эпохи")
    # Установка подписи оси Y
    plt.ylabel("Потери")
    # Добавление легенды
    plt.legend()
    # Установка заголовка графика
    plt.title("Потери на протяжении эпох")
    # Отображение графика
    plt.show()

    # Создание нового графика
    plt.figure(figsize=(10, 5))
    # Построение графика точности на обучающей выборке
    plt.plot(range(1, num_epochs + 1), training_accuracies, label="Training Accuracy")
    # Построение графика точности на валидационной выборке
    plt.plot(
        range(1, num_epochs + 1), validation_accuracies, label="Validation Accuracy"
    )
    # Установка подписи оси X
    plt.xlabel("Эпохи")
    # Установка подписи оси Y
    plt.ylabel("Точность (%)")
    # Добавление легенды
    plt.legend()
    # Установка заголовка графика
    plt.title("Точность на протяжении эпох")
    # Отображение графика
    plt.show()

    # Перевод модели в режим оценки
    model.eval()
    # Инициализация списка для истинных меток
    y_true = []
    # Инициализация списка для предсказанных меток
    y_pred = []
    # Отключение вычисления градиентов
    with torch.no_grad():
        # Цикл по батчам тестовых данных
        for test_features, test_labels in test_loader:
            # Перемещение данных на выбранное устройство
            test_features = test_features.to(device)
            test_labels = test_labels.to(device)
            # Получение выходных данных модели
            test_outputs = model(test_features)
            # Получение предсказанных меток
            _, predicted = torch.max(test_outputs.data, 1)
            # Добавление истинных меток в список
            y_true.extend(test_labels.cpu().numpy())
            # Добавление предсказанных меток в список
            y_pred.extend(predicted.cpu().numpy())

    # Вывод заголовка отчета о классификации
    print("Classification Report:")
    # Вывод отчета о классификации
    print(classification_report(y_true, y_pred))

    # Вычисление матрицы ошибок
    cm = confusion_matrix(y_true, y_pred)
    # Создание нового графика
    plt.figure(figsize=(10, 8))
    # Построение тепловой карты матрицы ошибок
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(num_classes),
        yticklabels=range(num_classes),
    )
    plt.xlabel("Предсказанные")
    plt.ylabel("Истинные")
    plt.title("Матрица ошибок")
    plt.show()

    # Сохранение модели
    torch.save(model.state_dict(), "trained_model.pth")
    print("Модель сохранена как trained_model.pth")


if __name__ == "__main__":
    main()
