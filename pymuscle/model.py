class Model(object):
    """
    Base model class from which other models should inherit
    """
    def __init__(
        self,
        motor_unit_count: int
    ):
        self.motor_unit_count = motor_unit_count

    def step(self):
        """
        Child classes must implement this method.
        """
        raise NotImplementedError
