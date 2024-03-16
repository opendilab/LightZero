"""
This code defines two classes, ``MinMaxStats`` and ``MinMaxStatsList``, \
for tracking and normalizing minimum and maximum values.
"""
FLOAT_MAX = 1000000.0
FLOAT_MIN = -float('inf')


class MinMaxStats:
    """
    Overview:
        Class for tracking and normalizing minimum and maximum values.
    Interfaces:
        ``__init__``,``set_delta``,``update``,``clear``,``normalize``.
    """

    def __init__(self) -> None:
        """
        Overview:
            Initializes an instance of the class.
        """
        self.clear()
        self.value_delta_max = 0

    def set_delta(self, value_delta_max: float) -> None:
        """
        Overview:
            Sets the maximum delta value.
        Arguments:
            - value_delta_max (:obj:`float`): The maximum delta value.
        """
        self.value_delta_max = value_delta_max

    def update(self, value: float) -> None:
        """
        Overview:
            Updates the minimum and maximum values.
        Arguments:
            - value (:obj:`float`): The value to update.
        """
        if value > self.maximum:
            self.maximum = value
        if value < self.minimum:
            self.minimum = value

    def clear(self) -> None:
        """
        Overview:
            Clears the minimum and maximum values.
        """
        self.minimum = FLOAT_MAX
        self.maximum = FLOAT_MIN

    def normalize(self, value: float) -> float:
        """
        Overview:
            Normalizes a value based on the minimum and maximum values.
        Arguments:
            - value (:obj:`float`): The value to normalize.
        Returns:
            - norm_value (:obj:`float`): The normalized value.
        """
        norm_value = value
        delta = self.maximum - self.minimum
        if delta > 0:
            if delta < self.value_delta_max:
                norm_value = (norm_value - self.minimum) / self.value_delta_max
            else:
                norm_value = (norm_value - self.minimum) / delta
        return norm_value


class MinMaxStatsList:
    """
    Overview:
        Class for managing a list of MinMaxStats instances.
    Interfaces:
        ``__init__``,``set_delta``.
    """

    def __init__(self, num: int) -> None:
        """
        Overview:
            Initializes a list of MinMaxStats instances.
        Arguments:
            - num (:obj:`int`): The number of MinMaxStats instances to create.
        """
        self.num = num
        self.stats_lst = [MinMaxStats() for _ in range(self.num)]

    def set_delta(self, value_delta_max: float) -> None:
        """
        Overview:
            Sets the maximum delta value for each MinMaxStats instance in the list.
        Arguments:
            - value_delta_max (:obj:`float`): The maximum delta value.
        """
        for i in range(self.num):
            self.stats_lst[i].set_delta(value_delta_max)
