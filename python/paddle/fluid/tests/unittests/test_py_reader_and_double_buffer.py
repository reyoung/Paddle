# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import paddle.fluid as fluid
import numpy

shapes = [[64, 3, 224, 224], [64, 1]]
dtypes = ['float64', 'int64']
num_iter = 8 * 4

assert num_iter % 8 == 0


def numpy_provider():
    for i in range(num_iter):
        img = numpy.ones(shape=shapes[0], dtype=dtypes[0]) * i
        label = numpy.ones(shape=shapes[1], dtype=dtypes[1]) * i
        yield img, label


def network(is_train):
    with fluid.unique_name.guard():
        reader = fluid.layers.py_reader(
            capacity=10,
            shapes=shapes,
            dtypes=dtypes,
            name='train_reader' if is_train else 'test_reader')

        img, label = fluid.layers.read_file(reader)
        img_mean = fluid.layers.mean(img)
        label = fluid.layers.cast(label, dtype='float64')
        label_mean = fluid.layers.mean(label)
        return reader, img_mean, label_mean


class TestDoubleBufferAndPyReader(unittest.TestCase):
    def test_train(self):
        if not fluid.core.is_compiled_with_cuda():
            return

        with self.scope_guard():
            main = fluid.Program()
            startup = fluid.Program()
            with fluid.program_guard(main, startup):
                reader, img, lbl = network(True)

            fluid.Executor(place=fluid.CUDAPlace(0)).run(startup)
            pe = fluid.ParallelExecutor(
                use_cuda=True, loss_name="", main_program=main)

            reader.decorate_tensor_provider(numpy_provider)

            for i in xrange(3):
                reader.start()
                img_np = None
                lbl_np = None

                while True:
                    try:
                        print reader.queue.size()
                        img_np_, lbl_np_ = map(numpy.array,
                                               pe.run(fetch_list=[img, lbl]))
                        if img_np is None:
                            img_np = img_np_
                            lbl_np = lbl_np_
                        else:
                            img_np = numpy.concatenate([img_np, img_np_])
                            lbl_np = numpy.concatenate([lbl_np, lbl_np_])
                    except fluid.core.EOFException:
                        reader.reset()
                        break

                arr = numpy.arange(
                    start=0, stop=num_iter, step=1, dtype='float64')
                self.expect_same_in_chunk(arr, img_np)
                self.expect_same_in_chunk(arr, lbl_np)

    def test_train_and_test(self):
        if not fluid.core.is_compiled_with_cuda():
            return
        with self.scope_guard():
            main = fluid.Program()
            train_startup = fluid.Program()
            with fluid.program_guard(main, train_startup):
                train_reader, train_img, train_lbl = network(True)

            test = fluid.Program()
            test_startup = fluid.Program()

            with fluid.program_guard(test, test_startup):
                test_reader, test_img, test_lbl = network(False)

            fluid.Executor(place=fluid.CUDAPlace(0)).run(train_startup)
            fluid.Executor(place=fluid.CUDAPlace(0)).run(test_startup)
            train_pe = fluid.ParallelExecutor(
                use_cuda=True, loss_name="", main_program=main)
            test_pe = fluid.ParallelExecutor(
                use_cuda=True,
                loss_name="",
                share_vars_from=train_pe,
                main_program=test)

            train_reader.decorate_tensor_provider(numpy_provider)
            test_reader.decorate_tensor_provider(numpy_provider)

            for i in xrange(3):
                train_reader.start()
                test_reader.start()
                img_np = None
                lbl_np = None

                while True:
                    try:
                        print train_reader.queue.size()
                        img_np_, lbl_np_ = map(
                            numpy.array,
                            train_pe.run(fetch_list=[train_img, train_lbl]))
                        if img_np is None:
                            img_np = img_np_
                            lbl_np = lbl_np_
                        else:
                            img_np = numpy.concatenate([img_np, img_np_])
                            lbl_np = numpy.concatenate([lbl_np, lbl_np_])
                    except fluid.core.EOFException:
                        train_reader.reset()
                        break

                arr = numpy.arange(
                    start=0, stop=num_iter, step=1, dtype='float64')

                self.expect_same_in_chunk(arr, img_np)
                self.expect_same_in_chunk(arr, lbl_np)

                img_np = None
                lbl_np = None
                while True:
                    try:
                        print test_reader.queue.size()
                        img_np_, lbl_np_ = map(
                            numpy.array,
                            test_pe.run(fetch_list=[test_img, test_lbl]))
                        if img_np is None:
                            img_np = img_np_
                            lbl_np = lbl_np_
                        else:
                            img_np = numpy.concatenate([img_np, img_np_])
                            lbl_np = numpy.concatenate([lbl_np, lbl_np_])
                    except fluid.core.EOFException:
                        test_reader.reset()
                        break
                self.expect_same_in_chunk(arr, img_np)
                self.expect_same_in_chunk(arr, lbl_np)

    def scope_guard(self):
        return fluid.scope_guard(scope=fluid.Scope())

    def expect_same_in_chunk(self, a, b):
        a = list(a)
        b = list(b)
        chunk_size = fluid.core.get_cuda_device_count()

        for i in range(0, len(a), chunk_size):
            a_view = sorted(a[i * chunk_size:(i + 1) * chunk_size])
            b_view = sorted(b[i * chunk_size:(i + 1) * chunk_size])

            for a_elem, b_elem in zip(a_view, b_view):
                self.assertAlmostEqual(
                    a_elem, b_elem, delta=1e-3, msg="Failed at {0}".format(i))


if __name__ == '__main__':
    unittest.main()
