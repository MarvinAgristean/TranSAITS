import unittest
import torch
from Models.Baseline_Methods.mean_imputation import mean_imputation

class mean_imputation_test(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)  # add assertion here

    def test_all_data_given_and_zero(self):
        X = torch.zeros(2, 4, 1)
        missing_mask = torch.ones_like(X)
        model = mean_imputation(1)
        model.train(X,missing_mask)

        imputed_data = model.impute(X, missing_mask)

        self.assertTrue(torch.equal(imputed_data, torch.zeros_like(X)))

    def test_all_data_given_and_one(self):
        X = torch.ones(2, 4, 1)
        missing_mask = torch.ones_like(X)
        model = mean_imputation(1)
        model.train(X,missing_mask)

        imputed_data = model.impute(X, missing_mask)

        self.assertTrue(torch.equal(imputed_data, torch.ones_like(X)))

    def test_no_data_given(self):
        X = torch.zeros(2, 4, 1)
        missing_mask = torch.zeros_like(X)
        model = mean_imputation(1)
        model.train(X,missing_mask)

        imputed_data = model.impute(X, missing_mask)
        print(model.mean)

        self.assertTrue(torch.equal(imputed_data, torch.zeros_like(X)))

    def test_one_feature_two_timeseries_seen_at_same_time(self):
        X = torch.zeros(2, 4, 1)
        X[0, 0, 0] = 4
        X[1, 0, 0] = 8
        missing_mask = torch.ones_like(X)
        missing_mask[0, 1, 0] = 0
        missing_mask[1, 2, 0] = 0
        imputation_expected = torch.clone(X)
        imputation_expected[0, 1, 0] = 12/6
        imputation_expected[1, 2, 0] = 12/6
        model = mean_imputation(1)
        model.train(X,missing_mask)

        imputed_data = model.impute(X, missing_mask)

        self.assertTrue(torch.equal(imputed_data, imputation_expected))

    def test_one_feature_two_timeseries_seen_at_different_times(self):
        X = torch.zeros(2, 4, 1)
        X[0, 3, 0] = 4
        X[1, 0, 0] = 8
        missing_mask = torch.ones_like(X)
        missing_mask[0, 1, 0] = 0
        missing_mask[1, 2, 0] = 0
        imputation_expected = torch.clone(X)
        imputation_expected[0, 1, 0] = 12/6
        imputation_expected[1, 2, 0] = 12/6
        model = mean_imputation(1)
        model.train(X,missing_mask)

        imputed_data = model.impute(X, missing_mask)

        self.assertTrue(torch.equal(imputed_data, imputation_expected))

    def test_one_feature_two_timeseries_many_seen(self):
        X = torch.zeros(2, 4, 1)
        X[0, 0, 0] = 2
        X[0, 3, 0] = 2
        X[1, 0, 0] = 8
        missing_mask = torch.ones_like(X)
        missing_mask[0, 1, 0] = 0
        missing_mask[1, 2, 0] = 0
        imputation_expected = torch.clone(X)
        imputation_expected[0, 1, 0] = 12/6
        imputation_expected[1, 2, 0] = 12/6
        model = mean_imputation(1)
        model.train(X,missing_mask)

        imputed_data = model.impute(X, missing_mask)

        self.assertTrue(torch.equal(imputed_data, imputation_expected))

    def test_two_features_two_timeseries(self):
        X = torch.zeros(2, 4, 2)
        X[0, 3, 0] = 4
        X[1, 0, 0] = 8
        X[0, 1, 1] = 16
        missing_mask = torch.ones_like(X)
        missing_mask[0, 1, 0] = 0
        missing_mask[1, 2, 0] = 0
        missing_mask[0,0,1] = 0
        imputation_expected = torch.clone(X)
        imputation_expected[0, 1, 0] = 12/6
        imputation_expected[1, 2, 0] = 12/6
        imputation_expected[0, 0, 1] = 16/7
        model = mean_imputation(2)
        model.train(X,missing_mask)

        imputed_data = model.impute(X, missing_mask)

        self.assertTrue(torch.equal(imputed_data, imputation_expected))


if __name__ == '__main__':
    unittest.main()
