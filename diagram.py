# Импорт модуля для рисования геометрических фигур
import matplotlib.patches as patches

# Импорт модуля для создания графиков
import matplotlib.pyplot as plt


# Определение функции для рисования нейронной сети
def draw_neural_network(ax, layer_sizes):
    # Установка вертикального расстояния между узлами
    v_spacing = 0.2
    # Установка горизонтального расстояния между слоями
    h_spacing = 0.5
    # Отключение осей графика
    ax.axis("off")

    # Инициализация списка для хранения позиций узлов
    positions = []

    # Цикл по слоям
    for i, layer_size in enumerate(layer_sizes):
        # Вычисление верхней позиции слоя
        layer_top = v_spacing * (layer_size - 1) / 2.0
        # Инициализация списка для хранения позиций узлов в текущем слое
        layer_positions = []
        # Цикл по узлам в слое
        for j in range(layer_size):
            # Вычисление позиции узла
            pos = (i * h_spacing, layer_top - j * v_spacing)
            # Добавление позиции узла в список
            layer_positions.append(pos)
            # Создание круга для представления узла
            circle = patches.Circle(pos, radius=0.1, color="skyblue", ec="black")
            # Добавление круга на график
            ax.add_patch(circle)
            # Добавление аннотации к узлу
            ax.annotate(
                f"Node {j + 1}\nLayer {i + 1}",
                (pos[0], pos[1]),
                xytext=(pos[0] - 0.25, pos[1] + 0.15),
            )
        # Добавление позиций узлов текущего слоя в общий список
        positions.append(layer_positions)

    # Рисование связей между узлами
    for i in range(len(layer_sizes) - 1):
        for j in range(layer_sizes[i]):
            for k in range(layer_sizes[i + 1]):
                # Создание линии, соединяющей узлы
                line = plt.Line2D(
                    [positions[i][j][0], positions[i + 1][k][0]],
                    [positions[i][j][1], positions[i + 1][k][1]],
                    c="black",
                )
                # Добавление линии на график
                ax.add_line(line)


# Пример: нейронная сеть с 3 слоями (входной, скрытый, выходной)
layer_sizes = [4, 5, 3]

# Создание фигуры для графика
fig = plt.figure(figsize=(30, 8))
# Добавление подграфика с равными масштабами по осям
ax = fig.add_subplot(111, aspect="equal")

# Вызов функции для рисования нейронной сети
draw_neural_network(ax, layer_sizes)
# Отображение графика
plt.show()
