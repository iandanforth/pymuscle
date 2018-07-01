class Model(object):
    """
    Base model class from which other models should inherit
    """

    def step(self):
        """
        Child classes must implement this method.
        """
        raise NotImplementedError
