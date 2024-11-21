from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type
import numpy as np


@dataclass
class LearningRate:
    """
    Класс для вычисления длины шага.

    Parameters
    ----------
    lambda_ : float, optional
        Начальная скорость обучения. По умолчанию 1e-3.
    s0 : float, optional
        Параметр для вычисления скорости обучения. По умолчанию 1.
    p : float, optional
        Степенной параметр для вычисления скорости обучения. По умолчанию 0.5.
    iteration : int, optional
        Текущая итерация. По умолчанию 0.

    Methods
    -------
    __call__()
        Вычисляет скорость обучения на текущей итерации.
    """
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5
    iteration: int = 0

    def __call__(self) -> float:
        """
        Вычисляет скорость обучения на текущей итерации.

        Returns
        -------
        float
            Текущая скорость обучения.
        """
        return self.lambda_ / (self.s0 + self.iteration) ** self.p


# Пример использования
lr = LearningRate(lambda_=1e-3, s0=1, p=0.5, iteration=10)
current_learning_rate = lr()
print(current_learning_rate)


class LossFunction(Enum):
    """
    Перечисление для выбора функции потерь.

    Attributes
    ----------
    MSE : auto
        Среднеквадратическая ошибка.
    MAE : auto
        Средняя абсолютная ошибка.
    LogCosh : auto
        Логарифм гиперболического косинуса от ошибки.
    Huber : auto
        Функция потерь Хьюбера.
    """
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    Базовый класс для всех методов градиентного спуска.

    Parameters
    ----------
    dimension : int
        Размерность пространства признаков.
    lambda_ : float, optional
        Параметр скорости обучения. По умолчанию 1e-3.
    loss_function : LossFunction, optional
        Функция потерь, которая будет оптимизироваться. По умолчанию MSE.

    Attributes
    ----------
    w : np.ndarray
        Вектор весов модели.
    lr : LearningRate
        Скорость обучения.
    loss_function : LossFunction
        Функция потерь.

    Methods
    -------
    step(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Шаг градиентного спуска.
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов на основе градиента. Метод шаблон.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по весам. Метод шаблон.
    calc_loss(x: np.ndarray, y: np.ndarray) -> float
        Вычисление значения функции потерь.
    predict(x: np.ndarray) -> np.ndarray
        Вычисление прогнозов на основе признаков x.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        Инициализация базового класса для градиентного спуска.

        Parameters
        ----------
        dimension : int
            Размерность пространства признаков.
        lambda_ : float
            Параметр скорости обучения.
        loss_function : LossFunction
            Функция потерь, которая будет оптимизирована.

        Attributes
        ----------
        w : np.ndarray
            Начальный вектор весов, инициализированный случайным образом.
        lr : LearningRate
            Экземпляр класса для вычисления скорости обучения.
        loss_function : LossFunction
            Выбранная функция потерь.
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Выполнение одного шага градиентного спуска.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Разность между текущими и обновленными весами.
        """
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Шаблон функции для обновления весов. Должен быть переопределен в подклассах.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь по весам.

        Returns
        -------
        np.ndarray
            Разность между текущими и обновленными весами. Этот метод должен быть реализован в подклассах.
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Шаблон функции для вычисления градиента функции потерь по весам. Должен быть переопределен в подклассах.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам. Этот метод должен быть реализован в подклассах.
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Вычисление значения функции потерь с использованием текущих весов.

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
        raise NotImplementedError('BaseDescent calc_loss function not implemented')

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Расчет прогнозов на основе признаков x.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.

        Returns
        -------
        np.ndarray
            Прогнозируемые значения.
        """
        raise NotImplementedError('BaseDescent predict function not implemented')


class VanillaGradientDescent(BaseDescent):
    """
    Класс полного градиентного спуска.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с учетом градиента.
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по весам.
    """

    def __init__(self, dimension: int, learning_rate: float = 1e-3):
        super().__init__(dimension, lambda_=learning_rate)
        self.learning_rate = learning_rate
        self.weights = np.zeros(dimension)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return x.dot(self.weights)
        """
        Инициализация класса с заданным шагом обучения и размерностью весов.

        Parameters
        ----------
        learning_rate : float
            Шаг обучения.
        dimension : int
            Размерность весов.
        """


    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов на основе градиента.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь по весам.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        old_weights = self.weights.copy()
        self.weights -= self.learning_rate * gradient
        return self.weights - old_weights

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента функции потерь по весам.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент функции потерь по весам.
        """
        predictions = x.dot(self.weights)
        errors = predictions - y
        gradient = (2 / len(y)) * x.T.dot(errors)  # Градиент для MSE
        return gradient


class StochasticDescent(VanillaGradientDescent):
    """
    Класс стохастического градиентного спуска.

    Parameters
    ----------
    batch_size : int, optional
        Размер мини-пакета. По умолчанию 50.

    Attributes
    ----------
    batch_size : int
        Размер мини-пакета.

    Методы
    -------
    calc_gradient(x: np.ndarray, y: np.ndarray) -> np.ndarray
        Вычисление градиента функции потерь по мини-пакетам.
    """

    def __init__(self, dimension: int, batch_size: int = 50, lambda_: float = 1e-3):
        super().__init__(dimension, learning_rate=lambda_)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Вычисление градиента на основе случайного мини-пакета.

        Parameters
        ----------
        x : np.ndarray
            Массив признаков.
        y : np.ndarray
            Массив целевых переменных.

        Returns
        -------
        np.ndarray
            Градиент для стохастического градиентного спуска.
        """
        # Рандомное разбиение на мини-пакеты
        indices = np.random.choice(len(x), size=self.batch_size, replace=False)
        x_batch = x[indices]
        y_batch = y[indices]
        return super().calc_gradient(x_batch, y_batch)


class MomentumDescent(StochasticDescent):
    """
    Класс градиентного спуска с моментумом.

    Атрибуты
    ---------
    momentum : float
        Параметр моментума.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с учетом градиента и моментума.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, momentum: float = 0.9, batch_size: int = 50):
        super().__init__(dimension, batch_size=batch_size, lambda_=lambda_)
        self.momentum = momentum
        self.velocity = np.zeros(dimension)  # Инициализация скорости

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с учетом моментума.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь по весам.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        self.velocity = self.momentum * self.velocity + self.learning_rate * gradient
        old_weights = self.weights.copy()
        self.weights -= self.velocity
        return self.weights - old_weights


class Adam(MomentumDescent):
    """
    Класс Адама для градиентного спуска с адаптивными моментами.

    Методы
    -------
    update_weights(gradient: np.ndarray) -> np.ndarray
        Обновление весов с учетом градиента и адаптивных моментов.
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, momentum: float = 0.9, batch_size: int = 50):
        super().__init__(dimension, lambda_=lambda_, momentum=momentum, batch_size=batch_size)
        self.m = np.zeros(dimension)
        self.v = np.zeros(dimension)
        self.t = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Обновление весов с использованием метода Адама.

        Parameters
        ----------
        gradient : np.ndarray
            Градиент функции потерь по весам.

        Returns
        -------
        np.ndarray
            Разность весов (w_{k + 1} - w_k).
        """
        self.t += 1
        self.m = self.momentum * self.m + (1 - self.momentum) * gradient
        self.v = self.momentum * self.v + (1 - self.momentum) * gradient ** 2

        m_hat = self.m / (1 - self.momentum ** self.t)
        v_hat = self.v / (1 - self.momentum ** self.t)

        update = self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

        old_weights = self.weights.copy()
        self.weights -= update
        return self.weights - old_weights


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config['descent_name']
    regularized = descent_config.get('regularized', False)
    kwargs = descent_config['kwargs']

    if 'dimension' not in kwargs:
        raise ValueError("'dimension' must be specified in kwargs")

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f"Incorrect descent name, use one of these: {list(descent_mapping.keys())}")

    descent_class = descent_mapping[descent_name]
    print(f"Initializing {descent_class.__name__} with kwargs: {kwargs}")
    if descent_name == 'full':
        kwargs.setdefault('learning_rate', 1e-3)  # Значение по умолчанию для learning_rate
    elif descent_name in ['stochastic', 'momentum', 'adam']:
        kwargs.setdefault('batch_size', 50)  # Устанавливаем batch_size для методов SGD
        kwargs.setdefault('lambda_', 1e-3)  # Значение по умолчанию для lambda_

        # Пытаемся создать объект, обрабатываем ошибки параметров
    try:
        return descent_class(**kwargs)
    except TypeError as e:
        raise ValueError(f"Error initializing {descent_class.__name__} with kwargs {kwargs}: {e}")