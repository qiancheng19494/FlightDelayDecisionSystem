"""增量统计与跨分块一致性。"""
from tests.data_helpers import DataTestCase, HEADER, ROWS
from core.data_loader import FlightDataLoader

class ValidatorTests(DataTestCase):
    def test_label_statistics(self):
        report=FlightDataLoader(self.csv()).scan_metadata()
        self.assertEqual((report["valid_labeled_count"],report["delay_count"],report["cancelled_count"]),(2,1,1))
        self.assertEqual(report["delay_rate"],.5)
    def test_duplicates_across_chunks(self):
        report=FlightDataLoader(self.csv(ROWS+[ROWS[0]]),1).scan_metadata()
        self.assertEqual(report["duplicate_rows"],1)
        self.assertAlmostEqual(report["delay_statistics"]["median"],14.95)
    def test_missing_and_invalid_dates(self):
        report=FlightDataLoader(self.csv([["bad","","","JFK","bad","3"]])).scan_metadata()
        self.assertEqual(report["invalid_dates"],1)
        self.assertEqual(report["fields"][1]["missing"],1)
        self.assertEqual(report["issues"]["取消状态无效或缺失"],1)
        self.assertIsNone(report["delay_rate"])
    def test_numeric_anomalies(self):
        report=FlightDataLoader(self.csv([ROWS[0]+[-1,0,2361]],HEADER+["DISTANCE","CRS_ELAPSED_TIME","CRS_DEP_TIME"])).scan_metadata()
        self.assertEqual(report["issues"]["距离为负"],1)
        self.assertEqual(report["issues"]["计划时长非正"],1)
        self.assertEqual(report["issues"]["scheduled_departure 无效时间"],1)
    def test_unlabeled(self):
        report=FlightDataLoader(self.csv([r[:4] for r in ROWS],HEADER[:4])).scan_metadata()
        self.assertEqual(report["dataset_type"],"PREDICTION_INPUT")
        self.assertIsNone(report["delay_rate"])
        self.assertEqual(report["valid_labeled_count"],0)
    def test_all_delay_missing(self):
        report=FlightDataLoader(self.csv([ROWS[0][:4]+["","0"]])).scan_metadata()
        self.assertEqual(report["delay_statistics"]["count"],0)
        self.assertTrue(report["warnings"])
    def test_missing_cancelled_warning(self):
        report=FlightDataLoader(self.csv([ROWS[0][:5]],HEADER[:5])).scan_metadata()
        self.assertEqual(report["valid_labeled_count"],1)
        self.assertIsNone(report["cancelled_rate"])
        self.assertIn("取消",report["warnings"][0])
    def test_monthly_counts(self):
        report=FlightDataLoader(self.csv()).scan_metadata()
        self.assertEqual(report["monthly_summary"]["2024-01"], {"rows":2,"labeled":2,"delayed":1})
    def test_na_airline_preserved(self):
        row=ROWS[0].copy();row[1]="NA"
        report=FlightDataLoader(self.csv([row])).scan_metadata()
        self.assertEqual(report["airline_count"],1)
        self.assertEqual(report["issues"]["航司缺失"],0)
    def test_chunk_invariance(self):
        path=self.csv(ROWS*11)
        first=FlightDataLoader(path,1).scan_metadata()
        second=FlightDataLoader(path,20).scan_metadata()
        for key in ["row_count","duplicate_rows","delay_statistics","fields","monthly_summary"]:
            self.assertEqual(first[key],second[key])

    def test_mixed_type_chunk_invariance(self):
        rows=[r.copy() for r in ROWS]
        rows[2][4]="bad"
        path=self.csv(rows)
        self.assertEqual(FlightDataLoader(path,1).scan_metadata()["fields"], FlightDataLoader(path,100).scan_metadata()["fields"])

    def test_source_mutation_during_scan(self):
        path=self.csv()
        def mutate(progress,rows):
            if progress==94:
                with path.open("a",encoding="utf-8") as stream:
                    stream.write("\n")
        with self.assertRaisesRegex(ValueError,"发生变化"):
            FlightDataLoader(path).scan_metadata(mutate)
