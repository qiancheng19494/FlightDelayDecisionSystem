"""实际服务、X/y 样本与审计失败状态，验证不覆盖历史缓存。"""
import json
from pathlib import Path
import pandas as pd
from core.feature_service import build_feature_sample,export_feature_sample,audit_result
from core.historical_features import HistoricalFeatureEngine
from core.feature_groups import FeatureCombinationRegistry
from tests.data_helpers import DataTestCase
class FeatureServiceTests(DataTestCase):
    def setUp(self):
        super().setUp()
        self.record=self.context.datasets.import_dataset(Path(__file__).parent/"fixtures/schedule_tiny.csv")
        self.h=HistoricalFeatureEngine(self.context);self.h.build_history_cache(self.record["id"])
    def build(self,**kwargs):return build_feature_sample(self.context,self.record["id"],self.record["id"],"COMBO_ATHC",self.record["id"],**kwargs)
    def test_sample_uses_full_schedule(self):
        r=self.build(limit=1);self.assertEqual(len(r.X),1);self.assertEqual(r.X.origin_hour_scheduled_departures[0],3);self.assertEqual(r.metadata["schedule_context_rows"],8)
    def test_export_separate(self):
        r=self.build();p=export_feature_sample(r,self.root/"export");x=pd.read_csv(p/"X_features.csv");y=pd.read_csv(p/"y_labels.csv");self.assertEqual(x.shape,(8,80));self.assertEqual(len(y),8);self.assertNotIn("delay_label",x);self.assertTrue((p/"metadata.json").exists())
    def test_cache_unchanged(self):
        p=self.h.folder/"metadata.json";before=p.read_bytes();self.build();self.assertEqual(p.read_bytes(),before)
    def test_no_cache_rejected(self):
        with self.assertRaises(ValueError):build_feature_sample(self.context,self.record["id"],None,"COMBO_H")
    def test_limit_rejected(self):
        with self.assertRaises(ValueError):self.build(limit=5001)
    def test_export_bad_lineage_rejected(self):
        r=self.build();r.metadata["lineage"]["distance"]["source_fields"]=("arrival_delay",)
        with self.assertRaises(ValueError):export_feature_sample(r,self.root/"export")
    def test_actual_H_metadata_audited(self):
        r=self.build();r.history_metadata.loc[4,"hist_global_max_source_date"]=pd.Timestamp("2025-01-01");self.assertFalse(audit_result(r)["passed"])
    def test_y_alignment_rejected(self):
        r=self.build();r.y=r.y.iloc[:1];self.assertFalse(audit_result(r)["passed"])
