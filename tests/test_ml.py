"""模型工程安全、训练分区、序列化和 DEMO 语义的实际回归。"""
import copy,json,shutil,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn import metrics as sm
from core.feature_pipeline import FeaturePipeline
from core.ml.preprocessor import ModelPreprocessor
from core.ml.time_split import TemporalDataSplitter
from core.ml.metrics import MetricsService
from core.ml.training_service import TrainingService,load_model_config,train_dataset
from core.ml.model_manager import ModelManager
from core.ml.model_registry import ModelRegistry,atomic_json
from core.ml.inference_service import InferenceService
from core.ml.trainers import xgboost_status
from core.historical_features import HistoricalFeatureEngine
from core.bootstrap import initialize
from scripts.create_historical_demo import create_demo
ROOT=Path(__file__).resolve().parents[1]
def basic():
    return pd.DataFrame(dict(flight_date=["2024-01-01"]*3,airline=["AA","BB",None],origin=["AAA"]*3,destination=["BBB"]*3,scheduled_departure=[1000]*3,scheduled_arrival=[1200]*3,scheduled_elapsed_time=[120]*3,distance=[0,10,20]))
class PreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.X=FeaturePipeline().build_features(basic(),"COMBO_A").X;self.p=ModelPreprocessor(tuple(self.X))
    def test_median_training_only(self):
        self.p.fit(self.X);v=self.X.iloc[:1].copy();v.distance=1000;self.p.transform(v);self.assertEqual(self.p.transformer_.named_transformers_["numeric"].statistics_[-1],10)
    def test_unknown_not_added(self):
        self.p.fit(self.X);v=self.X.iloc[:1].copy();v.airline="ZZ";self.p.transform(v);categories=self.p.transformer_.named_transformers_["categorical"]["onehot"].categories_[0];self.assertNotIn("ZZ",categories)
    def test_unknown_shape(self):
        self.p.fit(self.X);v=self.X.copy();v.airline="ZZ";self.assertEqual(self.p.transform(v).shape,self.p.transform(self.X).shape)
    def test_missing_category_token(self):
        self.p.fit(self.X);self.assertIn("__MISSING__",self.p.transformer_.named_transformers_["categorical"]["onehot"].categories_[0])
    def test_empty_numeric_retained(self):
        self.X.distance=np.nan;self.p.fit(self.X);self.assertIn("distance",self.p.empty_numeric_columns_);self.assertEqual(self.p.transformer_.named_transformers_["numeric"].statistics_[-1],0)
    def test_numeric_imputation_finite(self):
        self.X.loc[0,"distance"]=np.nan;y=self.p.fit_transform(self.X);self.assertTrue(np.isfinite(y.data).all())
    def test_wrong_order(self):
        self.p.fit(self.X)
        with self.assertRaises(ValueError):self.p.transform(self.X[self.X.columns[::-1]])
    def test_empty_features(self):
        with self.assertRaises(ValueError):ModelPreprocessor(()).fit(pd.DataFrame(index=[0]))
    def test_categorical_only(self):self.assertGreater(ModelPreprocessor(("airline",)).fit_transform(self.X[["airline"]]).shape[1],0)
    def test_numeric_only(self):self.assertEqual(ModelPreprocessor(("distance",)).fit_transform(self.X[["distance"]]).shape,(3,1))
    def test_input_not_mutated(self):
        before=self.X.copy(deep=True);self.p.fit_transform(self.X);pd.testing.assert_frame_equal(before,self.X)
    def test_y_unused(self):
        a=self.p.fit_transform(self.X,[0,0,1]);b=ModelPreprocessor(tuple(self.X)).fit_transform(self.X,[1,1,0]);np.testing.assert_array_equal(a.toarray(),b.toarray())
class SplitTests(unittest.TestCase):
    def setUp(self):self.s=TemporalDataSplitter();self.d=pd.Series(pd.date_range("2024-01-01",periods=10).repeat(2))
    def test_strict_boundary(self):
        r=self.s.split(self.d,is_demo=True);self.assertLess(self.d.iloc[r.train_positions].max(),self.d.iloc[r.validation_positions].min())
    def test_unique_date_80_20(self):r=self.s.split(self.d,is_demo=True);self.assertEqual(len(r.train_positions),16);self.assertEqual(len(r.validation_positions),4)
    def test_same_day_no_overlap(self):
        r=self.s.split(self.d,is_demo=True);self.assertFalse(set(self.d.iloc[r.train_positions])&set(self.d.iloc[r.validation_positions]))
    def test_shuffle_stable(self):
        a=self.s.split(self.d,is_demo=True);f=self.d.sample(frac=1,random_state=4).reset_index(drop=True);b=self.s.split(f,is_demo=True);self.assertEqual(set(self.d.iloc[a.train_positions]),set(f.iloc[b.train_positions]))
    def test_final_lock(self):
        with self.assertRaises(ValueError):self.s.split(pd.Series(["2024-01-01","2025-01-01"]),is_demo=True)
    def test_invalid_date(self):
        with self.assertRaises(ValueError):self.s.split(pd.Series(["bad","2024-01-01"]),is_demo=True)
    def test_single_date(self):
        with self.assertRaises(ValueError):self.s.split(self.d.iloc[:2],is_demo=True)
    def test_formal_window(self):
        r=self.s.split(pd.Series(["2024-01-01","2024-11-30","2024-12-31"]));self.assertEqual(r.metadata["training_end_date"],"2024-11-30")
    def test_formal_override_rejected(self):
        with self.assertRaises(ValueError):self.s.split(self.d,train_start="2024-01-01",train_end="2024-01-08",validation_start="2024-01-09",validation_end="2024-01-10")
    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):self.s.split(self.d,is_demo=True,train_start="2024-01-01",train_end="2024-01-09",validation_start="2024-01-09",validation_end="2024-01-10")
    def test_partial_range_rejected(self):
        with self.assertRaises(ValueError):self.s.split(self.d,is_demo=True,train_start="2024-01-01")
class MetricsTests(unittest.TestCase):
    def setUp(self):self.y=np.array([0,1,0,1,1]);self.p=np.array([.1,.7,.6,.2,.8]);self.r=MetricsService.evaluate(self.y,self.p)
    def test_accuracy(self):self.assertEqual(self.r["accuracy"],sm.accuracy_score(self.y,self.p>=.5))
    def test_precision(self):self.assertEqual(self.r["precision"],sm.precision_score(self.y,self.p>=.5))
    def test_recall(self):self.assertEqual(self.r["recall"],sm.recall_score(self.y,self.p>=.5))
    def test_f1(self):self.assertEqual(self.r["f1"],sm.f1_score(self.y,self.p>=.5))
    def test_auc(self):self.assertEqual(self.r["roc_auc"],sm.roc_auc_score(self.y,self.p))
    def test_ap(self):self.assertEqual(self.r["average_precision"],sm.average_precision_score(self.y,self.p))
    def test_confusion(self):self.assertEqual(self.r["confusion_matrix"],sm.confusion_matrix(self.y,self.p>=.5).tolist())
    def test_single_class(self):r=MetricsService.evaluate([0,0],[.1,.2]);self.assertIsNone(r["roc_auc"]);self.assertIsNone(r["average_precision"]);self.assertTrue(r["warnings"])
    def test_zero_precision(self):r=MetricsService.evaluate([0,1],[.1,.2]);self.assertEqual(r["precision"],0);self.assertTrue(r["warnings"])
    def test_threshold_boundary(self):self.assertEqual(MetricsService.evaluate([0,1],[.49,.5])["accuracy"],1)
    def test_invalid_probability(self):
        with self.assertRaises(ValueError):MetricsService.evaluate([0,1],[.1,2])
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name);(cls.root/"config").mkdir();shutil.copy2(ROOT/"config/default_config.json",cls.root/"config/default_config.json")
        cls.context=initialize(cls.root);source=cls.root/"demo.csv";create_demo(source);cls.record=cls.context.datasets.import_dataset(source);cls.frame=pd.read_csv(source,keep_default_na=False)
        cls.h=HistoricalFeatureEngine(cls.context);cls.h.build_history_cache(cls.record["id"]);cls.cfg=load_model_config(ROOT);cls.cfg["rf"]["n_estimators"]=12;cls.cfg["xgboost"]["n_estimators"]=12
        cls.rf=TrainingService(cls.root).train_frame(cls.frame,"rf","COMBO_A",is_demo=True,config=cls.cfg,dataset_id=cls.record["id"])
        cls.xgb=TrainingService(cls.root).train_frame(cls.frame,"xgboost","COMBO_ATHC",history=cls.h,schedule_context=cls.frame,is_demo=True,config=cls.cfg,dataset_id=cls.record["id"])
    @classmethod
    def tearDownClass(cls):
        for h in list(cls.context.logger.handlers):h.close();cls.context.logger.removeHandler(h)
        cls.temp.cleanup()
    def setUp(self):self.manager=ModelManager(self.root)
    def test_rf_serialization(self):self.assertTrue(self.rf["serialization_equal"])
    def test_xgb_serialization(self):self.assertTrue(self.xgb["serialization_equal"])
    def test_both_demo_marked(self):
        for r in [self.rf,self.xgb]:self.assertTrue(r["metadata"]["is_demo"]);self.assertEqual(r["metadata"]["status"],"DEMO");self.assertEqual(r["metadata"]["metrics"]["label"],"DEMO VALIDATION METRICS")
    def test_xgb_weight_train_only(self):
        m=self.xgb["metadata"];self.assertEqual(m["training_config"]["scale_pos_weight"],m["negative_count_train"]/m["positive_count_train"])
    def test_model_has_preprocessor(self):self.assertIsNotNone(self.manager.load_model(self.rf["metadata"]["model_id"]).preprocessor)
    def test_metadata_complete(self):self.manager.validate_metadata(self.xgb["metadata"]);self.assertTrue(self.xgb["metadata"]["final_test_locked"])
    def mismatch(self,field,value):
        m=copy.deepcopy(self.xgb["metadata"]);m[field]=value
        with self.assertRaises(ValueError):self.manager.validate_metadata(m)
    def test_schema_mismatch(self):self.mismatch("feature_schema_version",2)
    def test_names_mismatch(self):self.mismatch("feature_names",["distance"])
    def test_history_mismatch(self):self.mismatch("historical_schema_version",99)
    def test_combination_metadata_mismatch(self):self.mismatch("combination_id","COMBO_A")
    def test_version_mismatch(self):self.mismatch("library_versions",{})
    def test_threshold_mismatch(self):self.mismatch("threshold",.7)
    def test_demo_verified_rejected(self):self.mismatch("status","VERIFIED")
    def test_missing_metadata_rejected(self):
        m=copy.deepcopy(self.xgb["metadata"]);del m["training_config"]
        with self.assertRaises(ValueError):self.manager.validate_metadata(m)
    def test_predict_combo_mismatch(self):
        b=self.manager.load_model(self.xgb["metadata"]["model_id"]);r=FeaturePipeline().build_features(self.frame.iloc[:2],"COMBO_A")
        with self.assertRaises(ValueError):self.manager.predict_proba(b,r)
    def test_active_model(self):
        identity=self.rf["metadata"]["model_id"];self.manager.registry.set_active_model(identity);self.assertEqual(self.manager.registry.read()["active_model_id"],identity)
    def test_inference_range_demo(self):
        r=InferenceService(self.context).predict(self.frame.iloc[-3:],model_id=self.rf["metadata"]["model_id"]);self.assertTrue(r.is_demo);self.assertEqual(len(r.probability_delay),3);self.assertTrue(all(0<=p<=1 for p in r.probability_delay))
    def test_inference_H(self):self.assertTrue(InferenceService(self.context).predict(self.frame.iloc[-2:],model_id=self.xgb["metadata"]["model_id"],schedule_context=self.frame).is_demo)
    def test_inference_final_lock(self):
        f=self.frame.iloc[:1].copy();f.FL_DATE="2025-01-02"
        with self.assertRaises(ValueError):InferenceService(self.context).predict(f,model_id=self.rf["metadata"]["model_id"])
    def test_empty_training_rejected(self):
        with self.assertRaises(ValueError):TrainingService(self.root).train_frame(self.frame,"rf","COMBO_EMPTY",config=self.cfg)
    def test_formal_needs_user(self):
        with self.assertRaises(ValueError):TrainingService(self.root).train_frame(self.frame,"rf","COMBO_A",is_demo=False,config=self.cfg)
    def test_training_final_lock(self):
        f=self.frame.copy();f.loc[0,"FL_DATE"]="2025-01-01"
        with self.assertRaises(ValueError):TrainingService(self.root).train_frame(f,"rf","COMBO_A",config=self.cfg)
    def test_failure_record(self):
        f=self.frame.copy();f.DEP_DELAY=0
        with self.assertRaises(ValueError):TrainingService(self.root).train_frame(f,"rf","COMBO_A",config=self.cfg)
        records=[json.loads(p.read_text(encoding="utf-8")) for p in (self.root/"output/experiments").glob("*.json")];self.assertTrue(any(r["status"]=="FAILED" for r in records))
    def test_registry_no_untrusted_load(self):
        with patch("core.ml.model_manager.joblib.load",side_effect=AssertionError("must not deserialize")):
            with self.assertRaises(ValueError):self.manager.load_model("unknown_model")
    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):self.manager.load_model("../outside")
    def test_tampered_file_before_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shutil.copytree(self.root/"models",root/"models");identity=self.rf["metadata"]["model_id"]
            (root/"models/bundles"/identity/"metadata.json").write_text("{}",encoding="utf-8")
            with patch("core.ml.model_manager.joblib.load",side_effect=AssertionError("must not deserialize")):
                with self.assertRaises(ValueError):ModelManager(root).load_model(identity)
    def test_archive_keeps_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);shutil.copytree(self.root/"models",root/"models");manager=ModelManager(root);identity=self.rf["metadata"]["model_id"];manager.registry.set_active_model(identity);manager.registry.archive_model(identity)
            self.assertIsNone(manager.registry.read()["active_model_id"]);self.assertTrue((root/"models/bundles"/identity/"model.joblib").exists())
            with self.assertRaises(ValueError):manager.load_model(identity)
    def test_corrupt_registry_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/"models").mkdir();path=root/"models/registry.json";path.write_text("broken")
            with self.assertRaises(ValueError):ModelRegistry(root).list_models()
            self.assertEqual(path.read_text(),"broken")
    def test_experiment_demo_status(self):self.assertEqual(self.rf["experiment"]["status"],"COMPLETED");self.assertTrue(self.rf["experiment"]["is_demo"])
    def test_xgb_dependency_available(self):self.assertTrue(xgboost_status()[0])
    def test_feature_audit(self):self.assertTrue(self.xgb["metadata"]["feature_audit"]["passed"])
    def test_no_validation_fit(self):
        self.assertEqual(self.xgb["metadata"]["preprocessor_fit_scope"],"TRAIN_ONLY");self.assertLess(self.rf["metadata"]["train_max_date"],self.rf["metadata"]["validation_min_date"])

    def test_validation_labels_do_not_fit_model(self):
        changed=self.frame.copy();dates=pd.to_datetime(changed.FL_DATE);changed.loc[dates.ge("2024-03-13"),"DEP_DELAY"]=0
        other=TrainingService(self.root).train_frame(changed,"rf","COMBO_A",is_demo=True,config=self.cfg)
        a=self.manager.load_model(self.rf["metadata"]["model_id"]);b=self.manager.load_model(other["metadata"]["model_id"])
        X=FeaturePipeline().build_features(self.frame.iloc[-8:],"COMBO_A")
        # RF 并行求和可能有约 1e-16 的舍入差；每棵树必须完全一致。
        for left,right in zip(a.estimator.estimators_,b.estimator.estimators_):
            np.testing.assert_array_equal(left.tree_.threshold,right.tree_.threshold)
            np.testing.assert_array_equal(left.tree_.value,right.tree_.value)
        np.testing.assert_allclose(self.manager.predict_proba(a,X),self.manager.predict_proba(b,X),rtol=0,atol=1e-12)
    def test_inference_target_outcomes_ignored(self):
        a=self.frame.iloc[-3:].copy();b=a.copy();b.DEP_DELAY=999;b.CANCELLED=1
        service=InferenceService(self.context);identity=self.xgb["metadata"]["model_id"]
        self.assertEqual(service.predict(a,model_id=identity,schedule_context=self.frame).probability_delay,service.predict(b,model_id=identity,schedule_context=self.frame).probability_delay)
    def test_locked_metadata_rejected(self):self.mismatch("validation_end_date","2025-01-31")
    def test_dependency_unavailable_explicit(self):
        from core.ml.trainers import XGBoostTrainer
        trainer=XGBoostTrainer(self.cfg);trainer.parameters={}
        with patch("core.ml.trainers.xgboost_status",return_value=(False,"missing")):
            with self.assertRaisesRegex(ValueError,"依赖不可用"):trainer.make_estimator()
    def test_fractional_label_rejected(self):
        with self.assertRaises(ValueError):MetricsService.evaluate([0,.9],[.1,.8])
    def test_no_active_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            class Context:root=Path(directory)
            with self.assertRaisesRegex(ValueError,"当前模型"):InferenceService(Context()).predict(self.frame.iloc[:1])

    def test_failed_smoke_archives_model(self):
        before={m["model_id"] for m in self.manager.registry.list_models()}
        with patch("core.ml.training_service.np.allclose",return_value=False):
            with self.assertRaisesRegex(ValueError,"概率不一致"):TrainingService(self.root).train_frame(self.frame,"rf","COMBO_A",config=self.cfg)
        added=[m for m in self.manager.registry.list_models() if m["model_id"] not in before]
        self.assertEqual(len(added),1);self.assertEqual(added[0]["status"],"ARCHIVED")
    def test_invalid_inference_date_rejected(self):
        frame=self.frame.iloc[:1].copy();frame.FL_DATE="bad"
        with self.assertRaisesRegex(ValueError,"日期无效"):InferenceService(self.context).predict(frame,model_id=self.rf["metadata"]["model_id"])
