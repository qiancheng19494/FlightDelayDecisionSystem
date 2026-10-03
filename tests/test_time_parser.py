"""HHMM 及显式日期格式。"""
import unittest
import pandas as pd
from core.time_parser import parse_hhmm, parse_dates

class TimeTests(unittest.TestCase):
    def test_valid_times(self):
        for raw, expected in [(5,"00:05"),(30,"00:30"),(730,"07:30"),(1530,"15:30"),(2359,"23:59"),(0,"00:00")]:
            self.assertEqual(parse_hhmm(raw).text, expected)
    def test_midnight_rollover(self):
        self.assertEqual((parse_hhmm(2400).text, parse_hhmm(2400).day_offset), ("00:00",1))
    def test_invalid(self):
        for raw in [None, float("nan"), "", "bad", -1, 2360, 2401, 2500, 12.3, float("inf")]:
            with self.subTest(raw=raw):
                self.assertIsNone(parse_hhmm(raw).text)
    def test_dates(self):
        result = parse_dates(pd.Series(["2024-01-02", "01/02/2024", "2024/01/02", "2024-02-30", "bad", ""]))
        self.assertEqual(result.notna().sum(), 3)
        self.assertEqual(result.iloc[:3].nunique(), 1)
