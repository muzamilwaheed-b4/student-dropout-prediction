import os
import sys
import unittest
import numpy as np

# Ensure project directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from app import app, BUNDLE_PATH, bundle, model, scaler, FEATURES, LABEL_NAMES

class SystemTestSuite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_01_model_loaded(self):
        """Verify model bundle and scikit components are properly loaded."""
        self.assertIsNotNone(bundle, "Bundle should not be None")
        self.assertIsNotNone(model, "Model should be loaded")
        self.assertIsNotNone(scaler, "Scaler should be loaded")
        self.assertEqual(len(FEATURES), 10, "Should use exactly 10 features")

    def test_02_all_get_routes(self):
        """Test GET requests on all pages: Home, Predict, Results, About."""
        routes = ['/', '/prediction', '/results', '/about']
        for route in routes:
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, f"Route {route} failed with status {response.status_code}")
            # Ensure no python crash or unhandled exception rendered in HTML
            self.assertNotIn(b"Traceback", response.data)
            self.assertNotIn(b"Internal Server Error", response.data)

    def test_03_prediction_high_risk_dropout(self):
        """Test Case: A student with failing grades, 0 approved units, and fees not paid -> Expect Dropout."""
        data = {
            "cu1_approved": "0",
            "cu1_grade": "4.0",
            "cu2_approved": "0",
            "cu2_grade": "0.0",
            "admission_grade": "95.0",
            "tuition_fees": "0",   # Fees pending
            "scholarship": "0",
            "debtor": "1",         # In debt
            "gender": "1",
            "age": "26"
        }
        response = self.client.post('/prediction', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"JSON serializable", response.data)
        self.assertNotIn(b"Error:", response.data)
        # Should predict Dropout
        self.assertIn(b"Dropout", response.data)
        self.assertIn(b"High Risk", response.data)

    def test_04_prediction_high_performing_graduate(self):
        """Test Case: A student with top grades, all units approved, scholarship, and paid fees -> Expect Graduate."""
        data = {
            "cu1_approved": "6",
            "cu1_grade": "16.5",
            "cu2_approved": "6",
            "cu2_grade": "17.0",
            "admission_grade": "160.0",
            "tuition_fees": "1",
            "scholarship": "1",
            "debtor": "0",
            "gender": "0",
            "age": "19"
        }
        response = self.client.post('/prediction', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"Error:", response.data)
        # Should predict Graduate
        self.assertIn(b"Graduate", response.data)
        self.assertIn(b"Low Risk", response.data)

    def test_05_prediction_average_student(self):
        """Test Case: Average student hovering near threshold -> Valid response & probabilities."""
        data = {
            "cu1_approved": "4",
            "cu1_grade": "11.0",
            "cu2_approved": "4",
            "cu2_grade": "11.5",
            "admission_grade": "120.0",
            "tuition_fees": "1",
            "scholarship": "0",
            "debtor": "0",
            "gender": "1",
            "age": "20"
        }
        response = self.client.post('/prediction', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"Error:", response.data)
        self.assertIn(b"Model Evaluation Output", response.data)
        self.assertIn(b"Indicator", response.data)

    def test_06_edge_case_missing_fields(self):
        """Test Case: Incomplete form submission handled gracefully using defaults without 500 error."""
        data = {
            "cu1_approved": "3"
            # rest are missing
        }
        response = self.client.post('/prediction', data=data)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b"Internal Server Error", response.data)

if __name__ == '__main__':
    unittest.main()
