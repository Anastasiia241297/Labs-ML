from __future__ import annotations
from typing import List
import numpy as np
from descents import BaseDescent, get_descent


class LinearRegression:
    """
    Класс линейной регрессии.

    Parameters
    ----------
    descent_config : dict
        Конфигурация градиентного спуска.
    tolerance : float, optional
        Критерий остановки для квадрата евклидова нормы разности весов. По умолчанию равен 1e-4.
    max_iter : int, optional
        Критерий остановки по количеству итераций. По умолчанию равен 300.
    alpha : float, optional
        Параметр скорости обучения. По умолчанию равен 0.01.

    Attributes
    ----------
    descent : BaseDescent
        Экземпляр класса, реализующего градиентный спуск.
    tolerance : float
        Критерий остановки для квадрата евклидова нормы разности весов.
    max_iter : int
        Критерий остановки по количеству итераций.
    loss_history : List[float]
        История значений функции потерь на каждой итерации.
    weights : np.ndarray
        Векторы весов модели.
    alpha : float
        Параметр скорости обучения.
    """

    def __init__(self, descent_config: dict, tolerance: float = 1e-4, max_iter: int = 300, alpha: float = 0.01):
        descent_name = descent_config.get('descent_name', 'full')
        if descent_name == 'full' and 'learning_rate' not in descent_config['kwargs']:
            descent_config['kwargs']['learning_rate'] = 1e-3
        """
        Инициализация линейной регрессии с настройкой градиентного спуска.

        Parameters
        ----------
        descent_config : dict
            Конфигурация для алгоритма градиентного спуска.
        tolerance : float
            Критерий остановки для квадрата евклидовой нормы разности весов.
        max_iter : int
            Критерий остановки по количеству итераций.
        alpha : float
            Скорость обучения для градиентного спуска.
        """
        # Проверяем наличие всех необходимых параметров
        self.descent: BaseDescent = get_descent(descent_config)
        self.alpha: float = alpha
        self.tolerance: float = tolerance
        self.max_iter: int = max_iter
        self.loss_history: List[float] = []
        self.weights: np.ndarray = None  # Вектора весов инициализируются позже

    def fit(self, x: np.ndarray, y: np.ndarray) -> LinearRegression:
        """
        Обучение модели линейной регрессии, подбор весов для наборов данных x и y.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        self : LinearRegression
            Возвращает экземпляр класса с обученными весами.
        """
        # Инициализация весов
        m, n = x.shape
        self.weights = np.zeros(n)

        # Основное обучение
        for iteration in range(self.max_iter):
            # Прогнозирование
            predictions = x @ self.weights

            # Вычисление ошибки
            error = predictions - y

            # Вычисление градиента
            gradient = (1 / m) * (x.T @ error)

            # Обновление весов с учетом alpha
            new_weights = self.weights - self.alpha * gradient

            # Запись потерь
            loss = self.calc_loss(x, y)
            self.loss_history.append(loss)

            # Условия остановки
            if np.linalg.norm(new_weights - self.weights) < self.tolerance:
                break
            if np.isnan(new_weights).any():
                print("Weights contain NaN values. Stopping training.")
                break

            self.weights = new_weights

        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Прогнозирование целевых переменных для набора данных x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        prediction : np.ndarray
            Массив прогнозируемых значений.
        """
        return x @ self.weights

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Вычисление функции потерь (среднеквадратичная ошибка).

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        float
            Значение функции потерь.
        """
        return np.mean((x @ self.weights - y) ** 2)


# Пример использования
if __name__ == "__main__":
    # Пример данных
    x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y = np.array([5, 7, 9, 11])

    # Конфигурация для градиентного спуска
    descent_config = {
        'descent_name': 'stochastic',
        'regularized': False,
        'kwargs': {
            'dimension': 2,
            'batch_size': 2
        }
    }

    # Инициализация модели линейной регрессии
    regression = LinearRegression(
        descent_config=descent_config,
        tolerance=1e-4,
        max_iter=300,
        alpha=0.01
    )

    # Обучение модели
    regression.fit(x, y)

    # Прогнозирование
    predictions = regression.predict(x)
    print("Predictions:", predictions)
    print("Loss history:", regression.loss_history)
