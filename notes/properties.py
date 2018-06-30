class P:

    def __init__(self, x):
        self.x = x  # Invokes the x.setter below

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self, x):
        if x < 0:
            self.__x = 0
        elif x > 1000:
            self.__x = 1000
        else:
            self.__x = x


if __name__ == '__main__':
    p = P(-3)
    print(p.x)
    p.x = 4
    print(p.x)
    p.x = 2000
    print(p.x)