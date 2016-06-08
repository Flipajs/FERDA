import unittest
from core.graph.test_chunk import TestChunk
from core.graph.test_graph_manager import TestGraphManager
from core.graph.test_solver import TestSolver


if __name__ == '__main__':
    unittest.main()

    # not very elegant but working approach how to run all tests in sub folders (all attempts using discover failed)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite((
        loader.loadTestsFromTestCase(TestChunk),
        loader.loadTestsFromTestCase(TestGraphManager),
        loader.loadTestsFromTestCase(TestSolver),
        ))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
