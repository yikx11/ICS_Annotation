from .dla_up import dlasegnet


def get_arch(arch, classes=19):
    arch = arch.split('_')
    assert arch[0] == 'dla'
    return dlasegnet(arch[1], classes, False, 4)
