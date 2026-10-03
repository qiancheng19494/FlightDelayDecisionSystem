"""未接入预测前先封堵后验及未知字段。"""
import unittest
from core.feature_policy import validate_prediction_fields, POST_EVENT_FORBIDDEN_FIELDS

class FeaturePolicyTests(unittest.TestCase):
    def test_allowed(self):
        self.assertEqual(validate_prediction_fields(["FL_DATE","CRS_DEP_TIME"]),["flight_date","scheduled_departure"])
    def test_forbidden(self):
        for field in POST_EVENT_FORBIDDEN_FIELDS | {"DEP_DELAY","ARR_DELAY","DEP_TIME","ArrTime"}:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_prediction_fields([field])
    def test_unknown_rejected(self):
        with self.assertRaises(ValueError): validate_prediction_fields(["unapproved_feature"])
