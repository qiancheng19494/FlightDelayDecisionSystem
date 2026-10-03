"""别名映射与冲突边界。"""
import unittest
from core.field_mapper import FieldMapper, DatasetType, ALIASES

class FieldMapperTests(unittest.TestCase):
    def test_standard(self):
        result = FieldMapper().map_fields(["FL_DATE", "OP_UNIQUE_CARRIER", "ORIGIN", "DEST", "DEP_DELAY"])
        self.assertTrue(result.is_valid)
        self.assertEqual(result.dataset_type, DatasetType.HISTORICAL_LABELED)
    def test_aliases(self):
        for target, aliases in ALIASES.items():
            for alias in aliases + [target]:
                with self.subTest(alias=alias):
                    self.assertEqual(FieldMapper().canonical(alias), target)
    def test_case_insensitive(self):
        result = FieldMapper().map_fields(["fL_dAtE", "reporting_AIRLINE", "Origin", "destination"])
        self.assertTrue(result.is_valid)
        self.assertEqual(result.dataset_type, DatasetType.PREDICTION_INPUT)
    def test_missing(self):
        result = FieldMapper().map_fields(["FL_DATE"])
        self.assertEqual(result.missing_required_fields, ["airline", "origin", "destination"])
        self.assertEqual(result.dataset_type, DatasetType.UNKNOWN)
    def test_unknown(self):
        result = FieldMapper().map_fields(["FL_DATE", "AIRLINE", "ORIGIN", "DEST", "noise"])
        self.assertEqual(result.unknown_fields, ["noise"])
    def test_conflict(self):
        self.assertFalse(FieldMapper().map_fields(["FL_DATE", "AIRLINE", "OP_CARRIER", "ORIGIN", "DEST"]).is_valid)
    def test_no_fuzzy_match(self):
        self.assertIsNone(FieldMapper().canonical("DEP_DELAY_NEW"))
