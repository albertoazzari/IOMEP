from representation import *


class IOMEP:
    def __init__(self, data_dir):
        r = Representation(data_dir).get_representations()


if __name__ == '__main__':
    a = IOMEP('data/dcs')
