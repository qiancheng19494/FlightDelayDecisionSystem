"""CSV 结构、编码、分块和记录定位。"""
from tests.data_helpers import DataTestCase, ROWS, HEADER
from core.data_loader import FlightDataLoader, DataImportError

class LoaderTests(DataTestCase):
    def test_normal_csv(self):
        report = FlightDataLoader(self.csv()).scan_metadata()
        self.assertEqual(report["row_count"],3)
        self.assertEqual(report["start_date"],"2024-01-01")
    def test_chunking(self):
        chunks = list(FlightDataLoader(self.csv(),2).iter_chunks())
        self.assertEqual([len(c) for c in chunks],[2,1])
    def test_preview(self):
        loader = FlightDataLoader(self.csv(),2)
        list(loader.iter_chunks())
        self.assertEqual(loader.read_preview(2,2,loader.checkpoints).iloc[0,0], ROWS[2][0])
    def test_utf8_bom(self):
        loader=FlightDataLoader(self.csv(encoding="utf-8-sig"))
        self.assertEqual(loader.columns[0],"FL_DATE")
    def test_gb18030(self):
        rows=[ROWS[0]+["中文城市"]]
        loader=FlightDataLoader(self.csv(rows,HEADER+["备注"],encoding="gb18030"))
        self.assertEqual(loader.encoding,"gb18030")
        self.assertEqual(loader.read_preview().iloc[0,-1],"中文城市")
    def test_windows_encoding(self):
        path=self.csv([ROWS[0]+["€ café"]],HEADER+["note"],encoding="cp1252")
        self.assertEqual(FlightDataLoader(path).read_preview().iloc[0,-1],"€ café")
    def test_empty_file(self):
        path=self.root/"empty.csv";path.touch()
        with self.assertRaises(DataImportError): FlightDataLoader(path)
    def test_header_only(self):
        report=FlightDataLoader(self.csv([])).scan_metadata()
        self.assertEqual(report["row_count"],0)
        self.assertTrue(report["warnings"])
    def test_missing_file(self):
        with self.assertRaises(DataImportError): FlightDataLoader(self.root/"missing.csv")
    def test_wrong_extension(self):
        with self.assertRaises(DataImportError): FlightDataLoader(self.csv(name="wrong.txt"))
    def test_missing_fields(self):
        with self.assertRaisesRegex(DataImportError,"airline"):
            FlightDataLoader(self.csv([["2024-01-01"]],["FL_DATE"])).scan_metadata()
    def test_parser_error(self):
        path=self.csv();path.write_text(','.join(HEADER)+'\n"unterminated',encoding="utf-8")
        with self.assertRaises(DataImportError): list(FlightDataLoader(path).iter_chunks())
    def test_bad_width(self):
        with self.assertRaises(DataImportError): list(FlightDataLoader(self.csv([["only"]])).iter_chunks())
    def test_empty_lines_preserved(self):
        path=self.csv();
        with path.open("a",encoding="utf-8") as stream: stream.write("\n")
        self.assertEqual(FlightDataLoader(path).scan_metadata()["empty_rows"],1)
    def test_quoted_newline_index(self):
        rows=[ROWS[0]+["line1\nline2"], ROWS[1]+["中文"], ROWS[2]+["end"]]
        loader=FlightDataLoader(self.csv(rows,HEADER+["note"]),1)
        list(loader.iter_chunks())
        self.assertEqual(loader.read_preview(2,1,loader.checkpoints).iloc[0,-1],"中文")
    def test_cancel_scan(self):
        with self.assertRaisesRegex(DataImportError,"取消"):
            list(FlightDataLoader(self.csv()).iter_chunks(cancelled=lambda:True))
