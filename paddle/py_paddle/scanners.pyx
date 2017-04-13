import swig_paddle
import threading
from iscanner import IScanner
__all__ = ['IndexScanner']


class IndexScanner(IScanner):
    local = threading.local()

    def __init__(self, input_type, pos):
        IScanner.__init__(self, input_type, pos)
        if getattr(IndexScanner.local, '__ids__', None) is None:
            IndexScanner.local.__ids__ = [0] * 1024  # 1024 int buffer
        self.__idx__ = 0

    def scan(self, dat):
        try:
            IndexScanner.local.__ids__[self.__idx__] = dat
        except:
            IndexScanner.local.__ids__ *= 2
            IndexScanner.local.__ids__[self.__idx__] = dat
        self.__idx__ += 1

    def finish_scan(self, argument):
        ids = swig_paddle.IVector.create(
            IndexScanner.local.__ids__[:self.__idx__], self.data_in_gpu)
        assert isinstance(argument, swig_paddle.Arguments)
        argument.setSlotIds(self.pos, ids)
