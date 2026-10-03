"""阈值边界、缺失与取消规则。"""
import unittest
import pandas as pd
from core.delay_label import build_delay_label

class LabelTests(unittest.TestCase):
    def test_boundary(self):
        self.assertEqual(build_delay_label(pd.Series([14.9,15,15.1])).tolist(), [0,1,1])
    def test_missing(self):
        self.assertTrue(build_delay_label(pd.Series([float("nan"), "bad", float("inf")])).isna().all())
    def test_cancelled_excluded(self):
        labels = build_delay_label(pd.Series([0,30,20]), pd.Series([1,1,0]))
        self.assertTrue(labels.iloc[:2].isna().all())
        self.assertEqual(labels.iloc[2],1)
    def test_unknown_cancelled_excluded(self):
        self.assertTrue(build_delay_label(pd.Series([0,30,20]), pd.Series([None,3,"bad"])).isna().all())
