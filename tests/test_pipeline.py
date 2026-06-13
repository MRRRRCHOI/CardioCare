import unittest
import joblib
import pandas as pd
import numpy as np

from src.preprocessing import validate_input_ranges


class TestCardioCarePipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = joblib.load("model.pkl")
        cls.df = pd.read_csv("data/heart.csv")
        cls.X = cls.df.drop(columns=["target"]).head(10)

    def test_prediction_shape(self):
        pred = self.model.predict(self.X)
        self.assertEqual(len(pred), len(self.X))

    def test_prediction_probability_range(self):
        proba = self.model.predict_proba(self.X)

        self.assertTrue(np.all(proba >= 0))
        self.assertTrue(np.all(proba <= 1))

        row_sums = proba.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0))

    def test_clinical_input_range_validation(self):
        validate_input_ranges(self.X)

    def test_deterministic_prediction(self):
        pred1 = self.model.predict(self.X)
        pred2 = self.model.predict(self.X)

        self.assertTrue(np.array_equal(pred1, pred2))


if __name__ == "__main__":
    unittest.main()