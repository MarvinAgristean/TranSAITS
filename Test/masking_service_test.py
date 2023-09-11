import unittest
import torch
from Utils.masking_service import make_indicating_mask, make_missing_mask


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_make_missing_mask_all_data_given_2d(self):
        m = torch.zeros(2,4)
        missing_mask_real = torch.ones_like(m)

        missing_mask_out = make_missing_mask(m)

        self.assertTrue(torch.equal(missing_mask_out, missing_mask_real))

    def test_make_missing_mask_all_data_given_3d(self):
        m = torch.zeros(2,4,5)
        missing_mask_real = torch.ones_like(m)

        missing_mask_out = make_missing_mask(m)

        self.assertTrue(torch.equal(missing_mask_out, missing_mask_real))

    def test_make_missing_mask_no_data_given_2d(self):
        m = torch.zeros(2,4)
        m = m +float('NaN')
        missing_mask_real = torch.zeros_like(m)

        missing_mask_out = make_missing_mask(m)

        self.assertTrue(torch.equal(missing_mask_out, missing_mask_real))

    def test_make_missing_mask_no_data_given_3d(self):
        m = torch.zeros(2,4,5)
        m = m +float('NaN')
        missing_mask_real = torch.zeros_like(m)

        missing_mask_out = make_missing_mask(m)

        self.assertTrue(torch.equal(missing_mask_out, missing_mask_real))

    def test_make_missing_mask_some_data_given_2d(self):
        m = torch.zeros(2,4)
        m[0,1] = float('NaN')
        m[1,3] = float('NaN')
        missing_mask_real = torch.ones_like(m)
        missing_mask_real[0,1] = 0
        missing_mask_real[1,3] = 0

        missing_mask_out = make_missing_mask(m)

        self.assertTrue(torch.equal(missing_mask_out, missing_mask_real))

    def test_make_missing_mask_some_data_given_3d(self):
        m = torch.zeros(2,4,5)
        m[0,1] = float('NaN')
        m[1,3] = float('NaN')
        missing_mask_real = torch.ones_like(m)
        missing_mask_real[0,1] = 0
        missing_mask_real[1,3] = 0

        missing_mask_out = make_missing_mask(m)

        self.assertTrue(torch.equal(missing_mask_out, missing_mask_real))


if __name__ == '__main__':
    unittest.main()
