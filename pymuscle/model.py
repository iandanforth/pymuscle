class Model(object):
    """
    Base model class from which other models should inherit
    """

    def step(self):
        raise NotImplementedError
