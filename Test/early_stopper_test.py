import unittest
from Utils.early_stopper import Early_stopper

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)  # add assertion here

    def test_no_stopping_test(self):
        checkpoint_list = [10,9,8,7,6,5,4,3]
        patience = 2
        min_improvement = 1
        early_stopper = Early_stopper(patience = patience, min_improvement=min_improvement)
        stopping_list = []

        for i in range(len(checkpoint_list)):
            early_stopper.report(checkpoint_list[i])
            stopping_list.append(early_stopper.should_stop())

        expected_stopping_list = [False for i in range(8)]
        self.assertTrue(expected_stopping_list == stopping_list)

    def test_stopping_test1(self):
        checkpoint_list = [10,9,8,7,6,5,4,3]
        patience = 2
        min_improvement = 5
        early_stopper = Early_stopper(patience = patience, min_improvement=min_improvement)
        stopping_list = []

        for i in range(len(checkpoint_list)):
            early_stopper.report(checkpoint_list[i])
            stopping_list.append(early_stopper.should_stop())

        expected_stopping_list = [False, False, False,True,True,False,False,False]
        self.assertTrue(expected_stopping_list == stopping_list)

    def test_no_stopping_test2(self):
        checkpoint_list = [1,0.9, 0.8, 0.7, 0.6]
        patience = 2
        min_improvement = 0.1
        early_stopper = Early_stopper(patience = patience, min_improvement=min_improvement)
        stopping_list = []

        for i in range(len(checkpoint_list)):
            early_stopper.report(checkpoint_list[i])
            stopping_list.append(early_stopper.should_stop())

        expected_stopping_list = [False, False, False,False,False]
        self.assertTrue(expected_stopping_list == stopping_list)

    def test_stopping_test1(self):
        checkpoint_list = [1,0.9,0.8,0.75,0.7]
        patience = 2
        min_improvement = 0.3
        early_stopper = Early_stopper(patience = patience, min_improvement=min_improvement)
        stopping_list = []

        for i in range(len(checkpoint_list)):
            early_stopper.report(checkpoint_list[i])
            stopping_list.append(early_stopper.should_stop())

        expected_stopping_list = [False, False, False,True,False]
        self.assertTrue(expected_stopping_list == stopping_list)

    def test_stopping_test3(self):
        checkpoint_list = [1,0.9, 0.92,0.91, 0.93, 0.95]
        patience = 3
        min_improvement = 0.11
        early_stopper = Early_stopper(patience = patience, min_improvement=min_improvement)
        stopping_list = []

        for i in range(len(checkpoint_list)):
            early_stopper.report(checkpoint_list[i])
            stopping_list.append(early_stopper.should_stop())

        expected_stopping_list = [False, False, False,False, True, True]
        self.assertTrue(expected_stopping_list == stopping_list)


if __name__ == '__main__':
    unittest.main()
