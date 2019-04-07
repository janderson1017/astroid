# -*- encoding=utf-8 -*-
# Copyright (c) 2017-2018 hippo91 <guillaume.peillex@gmail.com>
# Copyright (c) 2017 Claudiu Popa <pcmanticore@gmail.com>
# Copyright (c) 2018 Bryce Guinta <bryce.paul.guinta@gmail.com>

# Licensed under the LGPL: https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html
# For details: https://github.com/PyCQA/astroid/blob/master/COPYING.LESSER
import unittest
import contextlib

try:
    import numpy  # pylint: disable=unused-import

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

from astroid import builder


class SubTestWrapper(unittest.TestCase):
    """
    A class for supporting all unittest version wether or not subTest is available
    """

    def subTest(self, msg=None, **params):
        try:
            # For python versions above 3.5 this should be ok
            return super(SubTestWrapper, self).subTest(msg, **params)
        except AttributeError:
            #  For python versions below 3.5
            return subTestMock(msg)


@contextlib.contextmanager
def subTestMock(msg=None):
    """
    A mock for subTest which do nothing
    """
    yield msg


@unittest.skipUnless(HAS_NUMPY, "This test requires the numpy library.")
class NumpyBrainFunctionReturningArrayTest(SubTestWrapper):
    """
    Test that calls to numpy functions returning arrays are correctly inferred
    """
    numpy_functions = (("array", "[1, 2]"),
                       ("linspace", "1, 100"),
                       ('empty_like', "[1, 2]"),
                       ('logical_or', "[1, 2]", "[1, 2]"),
                       ('logical_xor', "[1, 2]", "[1, 2]"),
                       ('logical_and', "[1, 2]", "[1, 2]"),
                       ('dot', "[1, 2]", "[1, 2]"),
                       ('vdot', "[1, 2]", "[1, 2]"),
                       ('concatenate', "([1, 2], [1, 2])"),
                       ('inner', "[1, 2]", "[1, 2]"),
                       ('where', '[True, False]', "[1, 2]", "[2, 1]"),
                       ('sum', '[[1, 2], [2, 1]]', "axis=0")
                       )

    def _inferred_numpy_func_call(self, func_name, *func_args):
        node = builder.extract_node(
            """
        import numpy as np
        func = np.{:s}
        func({:s})
        """.format(
                func_name, ",".join(func_args)
            )
        )
        return node.infer()

    def test_numpy_function_calls_inferred_as_ndarray(self):
        """
        Test that some calls to numpy functions are inferred as numpy.ndarray
        """
        licit_array_types = ('.ndarray', 'numpy.core.records.recarray')
        for func_ in self.numpy_functions:
            with self.subTest(typ=func_):
                inferred_values = list(self._inferred_numpy_func_call(*func_))
                self.assertTrue(len(inferred_values) == 1,
                                msg="Too much inferred value for {:s}".format(func_[0]))
                self.assertTrue(inferred_values[-1].pytype() in licit_array_types,
                                msg="Illicit type for {:s} ({})".format(func_[0], inferred_values[-1].pytype()))

if __name__ == "__main__":
    unittest.main()
