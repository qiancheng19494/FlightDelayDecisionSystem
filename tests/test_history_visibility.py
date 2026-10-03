"""Stage 5.1: changed source outcomes, cutoff provenance and service/GUI contract."""
import copy,json,tempfile,unittest,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from core.history_visibility import (HistoryVisibilityMode,EvaluationProtocol,protocol_metadata,history_description)
from core.historical_features import HistoricalFeatureEngine
from core.feature_pipeline import FeaturePipeline
from core.feature_service import audit_result
from core.feature_groups import HISTORY_FEATURES
from core.ml.training_service import TrainingService,load_model_config
from core.ml.model_manager import ModelManager
from core.ml.inference_service import InferenceService
ROOT=Path(__file__).resolve().parents[1]

class VisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        cls.frame=pd.DataFrame(dict(FL_DATE=pd.date_range("2024-03-01",periods=30).strftime("%Y-%m-%d"),OP_UNIQUE_CARRIER="AA",ORIGIN="AAA",DEST="BBB",CRS_DEP_TIME=1000,DEP_DELAY=[0,30]*15,CANCELLED=0))
        cls.engines={}
        changed=cls.frame.copy();changed.loc[changed.FL_DATE.between("2024-03-13","2024-03-29"),"DEP_DELAY"]=999
        one=cls.frame.copy();one.loc[one.FL_DATE.eq("2024-03-13"),"DEP_DELAY"]=999
        future=cls.frame.iloc[:10].copy();future.FL_DATE="2024-04-10";future.DEP_DELAY=999
        many=pd.concat([changed.iloc[12:29]]*10,ignore_index=True)
        for key,frame in dict(base=cls.frame,changed=changed,one=one,future=pd.concat([cls.frame,future]),many=pd.concat([cls.frame,many])).items():
            path=cls.root/(key+".csv");frame.to_csv(path,index=False)
            h=HistoricalFeatureEngine(source_path=path,cache_root=cls.root/key);h.build_history_cache(1);cls.engines[key]=h
        cls.targets=cls.frame.iloc[12:].copy()
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def build(self,key="base",mode="FROZEN_CUTOFF",cutoff="2024-03-12",targets=None,combo="COMBO_H"):
        return FeaturePipeline(self.engines[key]).build_features(self.targets if targets is None else targets,combo,mode="VALIDATION",history_visibility_mode=mode,history_cutoff_date=cutoff)
    def test_frozen_changed_validation_labels_exact(self):
        pd.testing.assert_frame_equal(self.build().X,self.build("changed").X,check_exact=True)
    def test_rolling_prior_day_changes_later_only(self):
        a=self.build(mode="ROLLING_ASOF",cutoff=None).X;b=self.build("one",mode="ROLLING_ASOF",cutoff=None).X
        pd.testing.assert_frame_equal(a.iloc[:1],b.iloc[:1],check_exact=True)
        self.assertTrue((a.iloc[1:].hist_global_mean_dep_delay_raw!=b.iloc[1:].hist_global_mean_dep_delay_raw).all())
    def test_future_append_no_effect(self):pd.testing.assert_frame_equal(self.build().X,self.build("future").X,check_exact=True)
    def test_many_post_cutoff_outcomes_no_effect(self):pd.testing.assert_frame_equal(self.build().X,self.build("many").X,check_exact=True)
    def test_cutoff_inclusive_explainable_count(self):
        a=self.build(cutoff="2024-03-10");b=self.build()
        self.assertTrue(a.X.hist_global_count.eq(10).all());self.assertTrue(b.X.hist_global_count.eq(12).all())
    def test_cutoff_equal_or_after_target_rejected(self):
        for cutoff in ["2024-03-13","2024-03-20"]:
            with self.subTest(cutoff=cutoff),self.assertRaises(ValueError):self.build(cutoff=cutoff)
    def test_invalid_and_missing_cutoff_rejected(self):
        for cutoff in [None,"bad","NaT","2024-03-12 12:00","2024-03-12T00:00:00Z"]:
            with self.subTest(cutoff=cutoff),self.assertRaises(ValueError):self.build(cutoff=cutoff)
    def test_rolling_cutoff_rejected(self):
        with self.assertRaises(ValueError):self.build(mode="ROLLING_ASOF")
    def test_validation_default_frozen(self):
        r=FeaturePipeline(self.engines["base"]).build_features(self.targets,"COMBO_H",mode="VALIDATION",history_cutoff_date="2024-03-12")
        self.assertEqual(r.metadata["history_visibility_mode"],"FROZEN_CUTOFF")
    def test_train_expanding_unchanged(self):
        a=FeaturePipeline(self.engines["base"]).build_features(self.frame,"COMBO_H",mode="TRAIN")
        b=self.engines["base"].transform_chunk(self.frame)
        pd.testing.assert_frame_equal(a.X,b.features[list(a.X)],check_exact=True)
    def test_train_cannot_freeze(self):
        with self.assertRaises(ValueError):FeaturePipeline(self.engines["base"]).build_features(self.frame,"COMBO_H",mode="TRAIN",history_visibility_mode="FROZEN_CUTOFF",history_cutoff_date="2024-02-29")
    def test_fallback_and_raw_source_dates_frozen(self):
        target=self.targets.copy();target.OP_UNIQUE_CARRIER="UNKNOWN";target.ORIGIN="UNK";r=self.build(targets=target)
        self.assertTrue(audit_result(r)["passed"])
        for name in r.history_metadata:
            if name.endswith("max_source_date"):self.assertTrue(pd.to_datetime(r.history_metadata[name]).dropna().le(pd.Timestamp("2024-03-12")).all())
    def test_frozen_audit_rejects_post_cutoff_provenance(self):
        r=self.build();r.history_metadata["history_max_source_date"]=pd.Timestamp("2024-03-13")
        self.assertFalse(audit_result(r)["passed"])
    def test_actual_target_date_preserved(self):
        self.assertEqual(self.build().history_metadata.target_date.dt.strftime("%Y-%m-%d").tolist(),self.targets.FL_DATE.tolist())
    def test_only_H_date_is_frozen(self):
        a=self.build(combo="COMBO_ATHC");b=self.build(mode="ROLLING_ASOF",cutoff=None,combo="COMBO_ATHC")
        fields=[c for c in a.X if c not in HISTORY_FEATURES];pd.testing.assert_frame_equal(a.X[fields],b.X[fields],check_exact=True)
    def test_final_test_policy_interface_only(self):
        p=protocol_metadata("RESEARCH_FROZEN","2024-11-30","2024-12-31")
        self.assertEqual(p["test_history_cutoff"],"2024-12-31")
        target=self.targets.iloc[:1].copy();target.FL_DATE="2025-01-15"
        r=FeaturePipeline(self.engines["base"]).build_features(target,"COMBO_H",mode="TEST",history_cutoff_date=p["test_history_cutoff"])
        self.assertIsNone(r.y);self.assertEqual(r.metadata["history_visibility_mode"],"FROZEN_CUTOFF")
    def test_service_default_and_explicit_protocol(self):
        config=load_model_config(ROOT);config["rf"]["n_estimators"]=3
        ranges=dict(train_start="2024-03-01",train_end="2024-03-12",validation_start="2024-03-13",validation_end="2024-03-30")
        results=[]
        for protocol in [None,"ONLINE_ROLLING"]:
            args={} if protocol is None else dict(evaluation_protocol=protocol)
            r=TrainingService(self.root).train_frame(self.frame,"rf","COMBO_ATHC",history=self.engines["base"],schedule_context=self.frame,config=config,date_ranges=ranges,**args)
            m=r["metadata"];results.append(m);self.assertTrue(r["serialization_equal"])
            self.assertEqual(m["train_history_policy"],"ROLLING_ASOF")
            self.assertEqual(m["evaluation_protocol"],protocol or "RESEARCH_FROZEN")
            self.assertEqual(m["validation_history_cutoff"],None if protocol else "2024-03-12")
            manager=ModelManager(self.root);bundle=manager.load_model(m["model_id"])
            check=FeaturePipeline(self.engines["base"]).build_features(self.targets,"COMBO_ATHC",mode="VALIDATION",schedule_context=self.frame,history_visibility_mode=m["validation_history_policy"],history_cutoff_date=m["validation_history_cutoff"])
            from core.ml.metrics import MetricsService
            measured=MetricsService.evaluate(check.y.to_numpy(dtype=int),manager.predict_proba(bundle,check))
            self.assertEqual(m["metrics"]["confusion_matrix"],measured["confusion_matrix"])
            if protocol is None:
                wrong=FeaturePipeline(self.engines["base"]).build_features(self.targets,"COMBO_ATHC",schedule_context=self.frame)
                with self.assertRaises(ValueError):manager.predict_proba(bundle,wrong)
            for field in ["evaluation_protocol","history_visibility_mode","history_cutoff_date","train_history_policy","validation_history_policy","validation_history_cutoff","test_history_policy","test_history_cutoff"]:
                broken=copy.deepcopy(m);del broken[field]
                with self.subTest(field=field),self.assertRaises(ValueError):manager.validate_metadata(broken)
            broken=copy.deepcopy(m);broken["validation_history_cutoff"]="2024-03-14"
            with self.assertRaises(ValueError):manager.validate_metadata(broken)
        self.assertNotEqual(results[0]["metrics"],results[1]["metrics"])
    def test_legacy_description_does_not_mutate(self):
        m={"model_version":1};before=copy.deepcopy(m);self.assertIn("legacy_rolling_validation",history_description(m));self.assertEqual(m,before)
    def test_final_inference_remains_locked(self):
        class Context:root=self.root
        target=self.frame.iloc[:1].copy();target.FL_DATE="2025-01-01"
        with self.assertRaisesRegex(ValueError,"锁定"):InferenceService(Context()).predict(target)

class ProtocolGuiTests(unittest.TestCase):
    def test_default_selection_boundaries_and_legacy_display(self):
        from PySide6.QtWidgets import QApplication
        from app.pages.model_training_page import ModelTrainingPage,ModelManagementPage
        from core.bootstrap import initialize
        app=QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/"config").mkdir();shutil.copy2(ROOT/"config/default_config.json",root/"config/default_config.json")
            context=initialize(root);page=ModelTrainingPage(context,lambda:None)
            self.assertEqual(page.protocol.currentData(),"RESEARCH_FROZEN")
            self.assertIn("不会吸收",page.policy_label.text())
            page.estimated(dict(training_start_date="2024-01-01",training_end_date="2024-03-12",validation_start_date="2024-03-13",validation_end_date="2024-03-30",train_rows_before_label_filter=12,validation_rows_before_label_filter=18))
            self.assertIn("2024-03-12",page.policy_label.text())
            page.protocol.setCurrentIndex(1);self.assertIn("前一验证日",page.policy_label.text())
            page.busy(True);self.assertFalse(page.protocol.isEnabled());page.busy(False)
            page.close();page.deleteLater();app.processEvents()
            for handler in list(context.logger.handlers):handler.close();context.logger.removeHandler(handler)
