"""Stage5.2 synthetic real-format regressions; these fixtures are never claimed real BTS."""
import copy,json,tempfile,unittest,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from core.bts_compatibility import verify,quality_frame
from core.data_loader import FlightDataLoader,DataImportError
from core.research_protocol_auditor import ResearchProtocolAuditor,cache_protocol,parse_combinations,literal_assignment
from core.real_bts_validation import read_bounded,validate_frame,run,MemoryMonitor
from core.historical_features import HistoricalFeatureEngine
from core.ml.training_service import TrainingService,load_model_config
from core.ml.model_manager import ModelManager
from core.history_cache import file_fingerprint
ROOT=Path(__file__).resolve().parents[1];FIXTURE=ROOT/"tests/fixtures/bts_schema_realistic.csv"
class BTSFormatTests(unittest.TestCase):
    def setUp(self):self.r=verify(FIXTURE,100);self.raw=FlightDataLoader(FIXTURE).read_preview(page_size=100)
    def test_actual_header_aliases(self):self.assertEqual(self.r["mapping"]["mapped_fields"]["Reporting_Airline"],"airline");self.assertFalse(self.r["missing_bts_fields"])
    def test_alternative_fl_date_aliases(self):
        f=self.raw.rename(columns={"FlightDate":"FL_DATE","Reporting_Airline":"OP_UNIQUE_CARRIER","DepDelay":"DEP_DELAY"});self.assertTrue(quality_frame(f)["compatible"])
    def test_unknown_fields_reported(self):self.assertIn("Unknown_BTS_Column",self.r["unknown_fields"])
    def test_required_missing_rejected(self):self.assertFalse(quality_frame(self.raw.drop(columns="Origin"))["compatible"])
    def test_alias_conflict_rejected(self):
        f=self.raw.copy();f["FL_DATE"]=f.FlightDate;self.assertFalse(quality_frame(f)["compatible"])
    def test_bts_empty_trailer_preserved(self):self.assertTrue(self.r["bts_empty_trailing_column"]);self.assertIn("__BTS_TRAILING_EMPTY__",self.raw)
    def test_nonempty_bts_trailer_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"bad.csv";lines=FIXTURE.read_text(encoding="utf-8").splitlines();lines[1]+="valuable";p.write_text("\n".join(lines),encoding="utf-8")
            with self.assertRaises(DataImportError):FlightDataLoader(p).read_preview()
    def test_arbitrary_empty_header_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"bad.csv";p.write_text("a,b,\n1,2,\n")
            with self.assertRaises(DataImportError):FlightDataLoader(p)
    def test_time_formats_and_invalid_count(self):
        self.assertEqual(self.r["scheduled_times"]["scheduled_departure"]["invalid_nonempty"],10);self.assertEqual(self.r["scheduled_times"]["scheduled_departure"]["next_day_2400"],10)
    def test_cancellation_semantics(self):
        c=self.r["cancelled_check"];self.assertEqual([c["zero"],c["one"],c["missing"],c["other"]],[50,10,10,10])
    def test_negative_delay_retained(self):
        self.assertGreater(self.r["departure_delay"]["negative"],0);self.assertGreater(self.r["labels"]["normal"],0)
    def test_label_boundary_matches_published(self):self.assertEqual(self.r["published_label_check"]["mismatches"],0)
    def test_posterior_not_prediction_allowed(self):self.assertIn("departure_delay",self.r["posterior_fields"]);self.assertNotIn("departure_delay",self.r["prediction_allowed_fields"])
    def test_quality_is_not_pipeline_pass(self):self.assertEqual(self.r["feature_pipeline_compatibility"],"SCHEMA_READY_NOT_EXECUTED")
    def test_bounded_rows_and_cap(self):
        self.assertEqual(len(read_bounded(FIXTURE,7)),7)
        with self.assertRaises(ValueError):read_bounded(FIXTURE,100001)
    def test_final_period_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"final.csv";f=self.raw.copy();f.FlightDate="2025-01-01";f.to_csv(p,index=False)
            with self.assertRaisesRegex(ValueError,"锁定"):read_bounded(p,10)
    def test_source_unchanged(self):
        before=FIXTURE.read_bytes();verify(FIXTURE,10);self.assertEqual(FIXTURE.read_bytes(),before)
    def test_fixture_cannot_be_real_model(self):
        with self.assertRaisesRegex(ValueError,"DEMO/fixture"):run(ROOT,[FIXTURE],80,False,"synthetic")
class ProtocolAuditTests(unittest.TestCase):
    def test_no_assets_unresolved(self):
        with tempfile.TemporaryDirectory() as d:self.assertEqual(ResearchProtocolAuditor().audit(d)["protocol"],"UNRESOLVED")
    def test_cache_proves_rolling(self):self.assertEqual(cache_protocol(pd.DataFrame(dict(FlightDate=["2024-12-15"],Max_Source_Date=["2024-12-14"]))),"ROLLING_ASOF")
    def test_no_future_observation_does_not_prove_frozen(self):self.assertEqual(cache_protocol(pd.DataFrame(dict(FlightDate=["2024-12-15"],Max_Source_Date=["2024-11-30"]))),"UNRESOLVED")
    def test_same_day_evidence_rejected(self):self.assertEqual(cache_protocol(pd.DataFrame(dict(FlightDate=["2024-12-15"],Max_Source_Date=["2024-12-15"]))),"INVALID_EVIDENCE")
    def test_combo_cross_product_not_16_subsets(self):
        groups=["A","A+T","A+H","A+C","A+T+H","A+T+C","A+H+C","A+T+H+C"]
        table=pd.DataFrame([(m,g) for m in ("RF","XGB") for g in groups],columns=["Model","Feature_Group"]);r=parse_combinations(groups,table)
        self.assertEqual((r["feature_combination_count"],r["experiment_count"]),(8,16));self.assertFalse(r["matches_software_16_subsets"])
        self.assertEqual(parse_combinations(groups,table.iloc[:-1])["status"],"UNRESOLVED")
    def test_config_not_executed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"config.py";p.write_text("raise RuntimeError('never execute')\nGROUPS=['A','A+H']")
            self.assertEqual(literal_assignment(p,"GROUPS"),["A","A+H"])
    def test_text_alone_not_frozen(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"outputs/paper";p.mkdir(parents=True);(p/"paper_v1_3_final.md").write_text("strict chronological frozen",encoding="utf-8")
            self.assertEqual(ResearchProtocolAuditor().audit(d)["confidence"],"UNRESOLVED")
class RealLikePipelineTests(unittest.TestCase):
    def test_full_real_like_invariance_audits(self):
        with tempfile.TemporaryDirectory() as d:
            h=HistoricalFeatureEngine(source_path=FIXTURE,cache_root=Path(d)/"cache");h.build_history_cache(1)
            frame=FlightDataLoader(FIXTURE).read_preview(page_size=100);r=validate_frame(frame,h,d)
            self.assertTrue(r["frozen_invariance"] and r["rolling_same_day_invariance"] and r["rolling_later_change"] and r["C_posterior_independence"])
            self.assertTrue(all(a["passed"] for a in r["audits"].values()));self.assertEqual(r["matrix_columns"],80)
    def test_real_smoke_identity_and_serialization(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);frame=FlightDataLoader(FIXTURE).read_preview(page_size=100);cfg=load_model_config(ROOT);cfg["rf"]["n_estimators"]=3
            r=TrainingService(root).train_frame(frame,"rf","COMBO_A",is_demo=False,formal_requested=True,dataset_id=1,fingerprints={str(FIXTURE):file_fingerprint(FIXTURE)},config=cfg,training_purpose="REAL_SMOKE")
            m=r["metadata"];self.assertFalse(m["is_demo"]);self.assertEqual(m["status"],"REAL_SMOKE");self.assertTrue(r["serialization_equal"])
            manager=ModelManager(root);self.assertIsNone(manager.registry.read()["active_model_id"])
            bad=copy.deepcopy(m);bad["training_purpose"]=None
            with self.assertRaises(ValueError):manager.validate_metadata(bad)
    def test_real_smoke_cannot_claim_demo(self):
        with tempfile.TemporaryDirectory() as d,self.assertRaises(ValueError):TrainingService(d).train_frame(FlightDataLoader(FIXTURE).read_preview(),"rf","COMBO_A",training_purpose="REAL_SMOKE")
    def test_rss_recorded_on_windows(self):
        import os
        with MemoryMonitor() as monitor:value=monitor.sample()
        if os.name=="nt":self.assertGreater(value,0)
class Stage52GuiTests(unittest.TestCase):
    def test_protocol_readonly_and_compatibility_button(self):
        from PySide6.QtWidgets import QApplication
        from core.bootstrap import initialize
        from app.pages.data_manager_page import DataManagerPage
        from app.pages.model_training_page import ModelManagementPage
        app=QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/"config").mkdir();shutil.copy2(ROOT/"config/default_config.json",root/"config/default_config.json")
            context=initialize(root);data=DataManagerPage(context);models=ModelManagementPage(context)
            self.assertEqual(data.compatibility_button.text(),"验证 BTS 兼容性");self.assertTrue(data.compatibility_details.isReadOnly());self.assertTrue(models.protocol_audit_view.isReadOnly())
            data.set_busy(True);self.assertFalse(data.compatibility_button.isEnabled())
            data.close();models.close();data.deleteLater();models.deleteLater();app.processEvents()
            for h in list(context.logger.handlers):h.close();context.logger.removeHandler(h)
