import unittest
import torch
from Models.Baseline_Methods.last_imputation import last_imputation


class last_imputation_test(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_all_data_given_and_zero(self):
        X = torch.zeros(2, 4, 1)
        missing_mask = torch.ones_like(X)
        model = last_imputation()

        imputed_data = model.impute(X, missing_mask)

        self.assertTrue(torch.equal(imputed_data, torch.zeros_like(X)))

    def test_all_data_given_and_one(self):
        X = torch.ones(2, 4, 1)
        missing_mask = torch.ones_like(X)
        model = last_imputation()

        imputed_data = model.impute(X, missing_mask)

        self.assertTrue(torch.equal(imputed_data, torch.ones_like(X)))

    def test_no_data_given(self):
        X = torch.zeros(2, 4, 1)
        missing_mask = torch.zeros_like(X)
        model = last_imputation()

        imputed_data = model.impute(X, missing_mask)

        self.assertTrue(torch.equal(imputed_data, torch.zeros_like(X)))

    def test_one_feature_two_timeseries_time_gap_of_one(self):
        X = torch.zeros(2, 4, 1)
        X[0, 0, 0] = 4
        X[1, 0, 0] = 8
        missing_mask = torch.ones_like(X)
        missing_mask[0, 1, 0] = 0
        missing_mask[1, 2, 0] = 0
        imputation_expected = torch.clone(X)
        imputation_expected[0, 1, 0] = 4
        imputation_expected[1, 2, 0] = 0
        model = last_imputation()

        imputed_data = model.impute(X, missing_mask)

        self.assertTrue(torch.equal(imputed_data, imputation_expected))

    def test_one_feature_two_timeseries_time_gap_of_two(self):
        X = torch.zeros(2, 4, 1)
        X[0, 0, 0] = 1
        X[1, 0, 0] = -1

        missing_mask = torch.ones_like(X)
        missing_mask[0, 2, 0] = 0
        missing_mask[1, 2, 0] = 0
        missing_mask[0, 1, 0] = 0
        missing_mask[1, 1, 0] = 0
        imputation_expected = torch.clone(X)
        imputation_expected[0, 1, 0] = 1
        imputation_expected[1, 2, 0] = -1
        imputation_expected[0, 2, 0] = 1
        imputation_expected[1, 1, 0] = -1
        model = last_imputation()

        imputed_data = model.impute(X, missing_mask)

        self.assertTrue(torch.equal(imputed_data, imputation_expected))

    def test_one_feature_two_timeseries_different_time_gaps(self):
        X = torch.zeros(2, 5, 1)
        X[0, 0, 0] = 1
        X[0, 3, 0] = 2
        X[1, 0, 0] = -1
        X[1, 4, 0] = 8
        missing_mask = torch.ones_like(X)
        missing_mask[0, 1, 0] = 0
        missing_mask[0, 2, 0] = 0
        missing_mask[1, 1, 0] = 0
        imputation_expected = torch.clone(X)
        imputation_expected[0, 1, 0] = 1
        imputation_expected[0,2,0] = 1
        imputation_expected[1, 1, 0] = -1
        model = last_imputation()

        imputed_data = model.impute(X, missing_mask)

        self.assertTrue(torch.equal(imputed_data, imputation_expected))

    def test_two_features_two_timeseries_different_time_gaps(self):
        X = torch.zeros(2, 4, 2)
        X[0,0,0] = -1
        X[1,1,0] = -2
        X[0,0,1] = 3
        missing_mask = torch.ones_like(X)
        missing_mask[0, 1, 0] = 0
        missing_mask[0,2,0] = 0
        missing_mask[1, 2, 0] = 0
        missing_mask[0,1,1] = 0
        imputation_expected = torch.clone(X)
        imputation_expected[0, 1, 0] = -1
        imputation_expected[1, 2, 0] = -2
        imputation_expected[0,2,0] = -1
        imputation_expected[0, 1, 1] = 3
        model = last_imputation()

        imputed_data = model.impute(X, missing_mask)

        self.assertTrue(torch.equal(imputed_data, imputation_expected))

    def test_one_features_two_timeseries_nothing_seen_before(self):
        X = torch.zeros(2, 4, 1)
        X[0,0,0] = -1
        X[1,1,0] = -2
        missing_mask = torch.ones_like(X)
        missing_mask[0, 0, 0] = 0
        missing_mask[1, 2, 0] = 0
        imputation_expected = torch.clone(X)
        imputation_expected[0, 0, 0] = 0
        imputation_expected[1, 2, 0] = -2

        model = last_imputation()

        imputed_data = model.impute(X, missing_mask)

        self.assertTrue(torch.equal(imputed_data, imputation_expected))

if __name__ == '__main__':
    unittest.main()
