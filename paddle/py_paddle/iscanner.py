import swig_paddle

__all__ = ['IScanner']


class IScanner(object):
    """
    The scanner will scan Python object two passes, then convert it to Paddle's
    argument.

    In the first pass, `pre_scan` will be invoked by every data instance, and
    then invoke `finish_pre_scan` to arguments. And the second pass do the same
    thing except the functions changed to `scan`, `finish_scan`.

    During the first pass, a scanner may count the shape of input matrix and
    allocate memory for this argument. Then fill the data into this  argument
    in second pass.
    """

    def __init__(self, input_type, pos):
        self.input_type = input_type
        if not isinstance(self.input_type, dp2.InputType):
            raise ValueError("input type should be dataprovider2.InputType")
        self.pos = pos
        # data_in_gpu is used to indicate whether to create argument on GPU
        # or not in GPU mode. Now if using one thread (trainer_count=1),
        # trainer uses NeuralNetwork which needs to create argument on GPU
        # before calling forward function. So, set data_in_gpu to True.
        # Otherwise, trainer uses MultiGradientMachine which will transfer
        # data from CPU to GPU in the forward function, set data_in_gpu to
        # False in this case.
        self.data_in_gpu = swig_paddle.isUsingGpu(
        ) and swig_paddle.getTrainerCount() == 1

    def pre_scan(self, dat):
        """
        First pass scan method. During this method, the scanner could count the
        data number, and get the total memory size this batch would use.

        :param dat: The python object.
        """
        pass

    def finish_pre_scan(self, argument):
        """
        Finish first scan pass. Allocate the memory.

        :param argument: Output arguments object.
        :type argument: swig_paddle.Arguments
        :return:
        """
        pass

    def scan(self, dat):
        """
        Second pass scan method. Copy the data to arguments.

        :param dat: The python object.
        """
        pass

    def finish_scan(self, argument):
        """
        Finish second pass. Finalize the resources, etc.

        :param argument: Output arguments object.
        :type argument: swig_paddle.Arguments
        """
        pass
