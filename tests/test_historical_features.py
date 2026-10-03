"""日级严格 date < D 测试；手算真值与变形测试共同验证，非仅检查不报错。"""
import json
import logging
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from pandas.testing import assert_frame_equal
from core.historical_features import HistoricalFeatureEngine
from core.history_schema import ENTITIES, HistoricalBatchResult, history_config
from core.leakage_auditor import TemporalLeakageAuditor
from core.feature_policy import validate_model_feature_list, PRE_FLIGHT_ALLOWED_FIELDS
from scripts.verify_bts_compatibility import verify

FIXTURE=Path(__file__).parent/"fixtures/historical_tiny.csv"

class HistoricalTests(unittest.TestCase):
    def setUp(self):
        self.log_handler=logging.NullHandler()
        logging.getLogger("flight_delay").addHandler(self.log_handler)
        self.temporary=tempfile.TemporaryDirectory(prefix="history_test_")
        self.root=Path(self.temporary.name)
        self.rows=pd.read_csv(FIXTURE,keep_default_na=False,dtype=str)
        self.counter=0
        self.engine=self.build(self.rows)

    def tearDown(self):
        logging.getLogger("flight_delay").removeHandler(self.log_handler)
        self.temporary.cleanup()

    def build(self,rows,config=None,chunksize=4):
        self.counter+=1
        source=self.root/f"source_{self.counter}.csv"
        rows.to_csv(source,index=False)
        engine=HistoricalFeatureEngine(source_path=source,cache_root=self.root,config=config,chunksize=chunksize)
        engine.build_history_cache(self.counter)
        return engine

    def query(self,date="2024-01-03",airline="AA",origin="AAA",destination="BBB",engine=None):
        return (engine or self.engine).transform_single(dict(flight_date=date,airline=airline,origin=origin,destination=destination))

    def test_first_day_no_history(self):
        result=self.query("2024-01-01")
        for e in ENTITIES:
            self.assertEqual(result.features[f"hist_{e}_count"],0)
            self.assertTrue(math.isnan(result.features[f"hist_{e}_delay_rate"]))
        self.assertEqual(result.features["cold_start_any"],1)
    def test_second_day_only_first(self):
        result=self.query("2024-01-02").features
        self.assertEqual(result["hist_global_count"],2)
        self.assertEqual(result["hist_global_delay_rate"],.5)
    def test_third_day_exact_values(self):
        result=self.query().features
        self.assertEqual(result["hist_global_count"],4)
        self.assertEqual(result["hist_route_count"],3)
        self.assertEqual(result["hist_route_delay_rate"],2/3)
        self.assertEqual(result["hist_airline_count"],3)
    def test_same_day_label_mutation(self):
        changed=self.rows.copy()
        changed.loc[changed.FL_DATE.eq("2024-01-03"),"DEP_DELAY"]="9999"
        other=self.build(changed)
        target=self.rows.loc[self.rows.FL_DATE.eq("2024-01-03")]
        assert_frame_equal(self.engine.transform_chunk(target).features,other.transform_chunk(target).features,check_exact=True)
    def test_future_labels_mutation(self):
        changed=self.rows.copy()
        changed.loc[changed.FL_DATE.ge("2024-01-10"),"DEP_DELAY"]="-1000"
        other=self.build(changed)
        target=self.rows.loc[self.rows.FL_DATE.le("2024-01-10")]
        assert_frame_equal(self.engine.transform_chunk(target).features,other.transform_chunk(target).features,check_exact=True)
    def test_append_future_month(self):
        future=self.rows.copy();future["FL_DATE"]="2024-02-01";future["DEP_DELAY"]="500"
        other=self.build(pd.concat([self.rows,future],ignore_index=True))
        assert_frame_equal(self.engine.transform_chunk(self.rows).features,other.transform_chunk(self.rows).features,check_exact=True)
    def test_shuffled_rows_exactly_equal(self):
        other=self.build(self.rows.sample(frac=1,random_state=72).reset_index(drop=True),chunksize=3)
        assert_frame_equal(self.engine.transform_chunk(self.rows).features,other.transform_chunk(self.rows).features,check_exact=True)
    def test_chunk_size_invariant(self):
        other=self.build(self.rows,chunksize=1)
        assert_frame_equal(self.engine.transform_chunk(self.rows).features,other.transform_chunk(self.rows).features,check_exact=True)
    def test_airline_isolation(self):
        self.assertEqual(self.query(airline="BB").features["hist_airline_count"],1)
        self.assertEqual(self.query(airline="BB").features["hist_airline_delay_rate_raw"],0)
    def test_origin_isolation(self):
        self.assertEqual(self.query(origin="CCC").features["hist_origin_count"],1)
    def test_destination_isolation(self):
        self.assertEqual(self.query(destination="DDD").features["hist_destination_count"],1)
    def test_directional_route(self):
        result=self.query(origin="BBB",destination="AAA")
        self.assertEqual(result.features["hist_route_count"],0)
        self.assertEqual(result.metadata["hist_route_fallback_source"],"GLOBAL")
    def test_airline_origin_isolation(self):
        result=self.query(airline="BB",origin="AAA")
        self.assertEqual(result.features["hist_airline_origin_count"],0)
        self.assertEqual(result.metadata["hist_airline_origin_fallback_source"],"AIRLINE_ORIGIN_MEAN")
        self.assertEqual(result.features["hist_airline_origin_delay_rate"],1/3)
    def test_cancelled_excluded(self):
        self.assertEqual(self.query("2024-01-04").features["hist_global_count"],7)
        self.assertAlmostEqual(self.query("2024-01-04").features["hist_global_mean_dep_delay"],95/7)
    def test_missing_delay_excluded(self):
        changed=self.rows.copy();changed.loc[8,"DEP_DELAY"]="NaN"
        other=self.build(changed)
        self.assertEqual(self.query("2024-01-04",engine=other).features["hist_global_count"],7)
    def test_boundary_labels(self):
        result=self.query("2024-01-04").features
        self.assertEqual(result["hist_global_delay_count"],4)
        self.assertEqual(result["hist_global_delay_rate"],4/7)
    def test_weighted_average_not_daily_average(self):
        result=self.query("2024-01-04").features
        self.assertAlmostEqual(result["hist_global_mean_dep_delay"],95/7)
        self.assertNotAlmostEqual(result["hist_global_mean_dep_delay"],(10+15+15)/3)
    def test_cold_route_keeps_zero_count(self):
        result=self.query(destination="DDD")
        self.assertEqual(result.features["hist_route_count"],0)
        self.assertEqual(result.features["hist_route_available"],0)
        self.assertEqual(result.features["cold_start_route"],1)
        self.assertTrue(math.isnan(result.features["hist_route_delay_rate_raw"]))
        self.assertEqual(result.features["hist_route_delay_rate"],1/3)
        self.assertEqual(result.metadata["hist_route_fallback_source"],"ORIGIN_DESTINATION")
    def test_only_origin_fallback(self):
        result=self.query(destination="NEW")
        self.assertEqual(result.metadata["hist_route_fallback_source"],"ORIGIN")
        self.assertEqual(result.features["hist_route_delay_rate"],2/3)
    def test_only_destination_fallback(self):
        result=self.query(origin="NEW")
        self.assertEqual(result.metadata["hist_route_fallback_source"],"DESTINATION")
    def test_global_fallback_no_future(self):
        result=self.query("2024-01-02",airline="NEW",origin="NEW",destination="NEW")
        self.assertEqual(result.features["hist_route_delay_rate"],.5)
        self.assertEqual(result.metadata["hist_route_fallback_source"],"GLOBAL")
        self.assertEqual(result.metadata["hist_route_effective_max_source_date"],pd.Timestamp("2024-01-01"))
    def test_before_source_cold(self):
        self.assertEqual(self.query("2023-12-01").features["hist_global_count"],0)
    def test_after_source_uses_all_past(self):
        self.assertEqual(self.query("2025-01-10").features["hist_global_count"],12)
    def test_sparse_asof(self):
        self.assertEqual(self.query("2024-01-09").features["hist_global_count"],8)
        self.assertEqual(self.query("2024-01-09").metadata["history_max_source_date"],pd.Timestamp("2024-01-05"))
    def test_missing_key_not_nan_entity(self):
        for key in [None,"", "nan",float("nan")," NULL "]:
            result=self.query(airline=key)
            self.assertEqual(result.features["hist_airline_count"],0)
            self.assertEqual(result.features["cold_start_airline"],1)
    def test_key_case_and_strip(self):
        self.assertEqual(self.query(airline=" aa ",origin=" aaa ").features["hist_airline_origin_count"],3)
    def test_min_history_status(self):
        other=self.build(self.rows,config={"min_entity_history":4})
        result=self.query(engine=other)
        self.assertEqual(result.features["hist_route_count"],3)
        self.assertEqual(result.metadata["hist_route_status"],"INSUFFICIENT_HISTORY")
        self.assertEqual(result.metadata["hist_route_fallback_source"],"GLOBAL")
        self.assertEqual(result.features["hist_route_delay_rate"],.5)
    def test_no_cancel_column_warning(self):
        other=self.build(self.rows.drop(columns="CANCELLED"))
        self.assertIn("取消状态字段缺失",other.metadata["warnings"][0])
    def test_duplicate_not_removed(self):
        other=self.build(pd.concat([self.rows,self.rows.iloc[[0]]],ignore_index=True))
        self.assertEqual(self.query("2024-01-02",engine=other).features["hist_global_count"],3)
    def test_invalid_date_excluded(self):
        changed=self.rows.copy();changed.loc[0,"FL_DATE"]="bad"
        other=self.build(changed)
        self.assertEqual(self.query("2024-01-02",engine=other).features["hist_global_count"],1)
        self.assertEqual(other.metadata["validation"]["invalid_target_dates_skipped"],1)
    def test_target_date_invalid_rejected(self):
        with self.assertRaises(ValueError): self.query("bad")
    def test_target_outcomes_ignored(self):
        target=dict(flight_date="2024-01-03",airline="AA",origin="AAA",destination="BBB",departure_delay=999999,arrival_delay=999999,cancelled=1)
        self.assertEqual(self.query().features,self.engine.transform_single(target).features)
    def test_single_batch_consistency(self):
        target=self.rows.iloc[[2]]
        single=self.engine.transform_single(target.iloc[0].to_dict())
        for field,value in self.engine.transform_chunk(target).features.iloc[0].items():
            self.assertEqual(single.features[field],value)
    def test_asof_cannot_pass_target(self):
        with self.assertRaisesRegex(ValueError,"不能晚于"):
            self.engine.transform_single(dict(flight_date="2024-01-02",airline="AA",origin="AAA",destination="BBB"),"2024-01-03")
    def test_index_preserved(self):
        target=self.rows.iloc[:3].copy();target.index=[99,7,99]
        self.assertEqual(self.engine.transform_chunk(target).features.index.tolist(),[99,7,99])
    def test_transform_dataset_streams(self):
        batches=list(self.engine.transform_dataset())
        self.assertEqual(sum(len(b.features) for b in batches),14)
        self.assertGreater(len(batches),1)
    def test_unlabeled_rejected(self):
        with self.assertRaisesRegex(ValueError,"没有真实历史标签"):
            self.build(self.rows.drop(columns="DEP_DELAY"))
    def test_header_only_global_nan(self):
        other=self.build(self.rows.iloc[:0])
        self.assertEqual(self.query(engine=other).features["hist_global_count"],0)
    def test_cache_hit_no_csv_scan(self):
        other=HistoricalFeatureEngine(source_path=self.engine.source_path,cache_root=self.root,chunksize=4)
        with patch("core.data_loader.FlightDataLoader.iter_chunks",side_effect=AssertionError("no scan allowed")):
            self.assertTrue(other.load_cache(self.engine.dataset_id))
            self.assertEqual(self.query(engine=other).features["hist_global_count"],4)
    def test_source_mutation_invalidates(self):
        with self.engine.source_path.open("a") as stream: stream.write("\n")
        with self.assertRaisesRegex(ValueError,"失效"): self.query()
        self.assertFalse(self.engine.load_cache(self.engine.dataset_id))
    def test_config_invalidates(self):
        other=HistoricalFeatureEngine(source_path=self.engine.source_path,cache_root=self.root,config={"min_entity_history":5})
        self.assertFalse(other.load_cache(self.engine.dataset_id))
    def test_schema_version_invalidates(self):
        path=self.engine.folder/"metadata.json"
        data=json.loads(path.read_text(encoding="utf-8"));data["historical_feature_schema_version"]=999
        path.write_text(json.dumps(data),encoding="utf-8")
        self.assertFalse(self.engine.load_cache(self.engine.dataset_id))
    def test_engine_version_invalidates(self):
        path=self.engine.folder/"metadata.json"
        data=json.loads(path.read_text(encoding="utf-8"));data["engine_version"]="future"
        path.write_text(json.dumps(data),encoding="utf-8")
        self.assertFalse(self.engine.load_cache(self.engine.dataset_id))
    def test_cache_file_corruption_invalidates(self):
        with (self.engine.folder/"global_daily.csv").open("a") as stream: stream.write("broken")
        self.assertFalse(self.engine.load_cache(self.engine.dataset_id))
    def test_cancel_preserves_previous_cache(self):
        old=self.engine.folder
        with self.assertRaises(ValueError):
            self.engine.build_history_cache(self.engine.dataset_id,force=True,cancelled=lambda:True)
        self.assertTrue(self.engine.load_cache(self.engine.dataset_id))
        self.assertEqual(self.engine.folder,old)
    def test_auditor_rejects_same_day(self):
        result=self.query();result.metadata["hist_route_max_source_date"]=pd.Timestamp("2024-01-03")
        self.assertFalse(TemporalLeakageAuditor().audit_history_features(result)["passed"])
    def test_auditor_rejects_future_fallback(self):
        result=self.query();result.metadata["hist_route_effective_max_source_date"]=pd.Timestamp("2025-01-01")
        self.assertFalse(TemporalLeakageAuditor().audit_history_features(result)["passed"])
    def test_auditor_rejects_raw_fields(self):
        result=self.query();result.features["DEP_DELAY"]=0
        self.assertFalse(TemporalLeakageAuditor().audit_history_features(result)["passed"])
    def test_auditor_rejects_missing_provenance(self):
        result=self.query();result.metadata["hist_route_max_source_date"]=pd.NaT
        self.assertFalse(TemporalLeakageAuditor().audit_history_features(result)["passed"])
    def test_model_feature_policy(self):
        self.assertEqual(validate_model_feature_list(["FL_DATE","hist_route_delay_rate"]),["flight_date","hist_route_delay_rate"])
        self.assertNotIn("departure_delay",PRE_FLIGHT_ALLOWED_FIELDS)
        for field in ["DEP_DELAY","ARR_DELAY","DEP_TIME","hist_route_fallback_source"]:
            with self.assertRaises(ValueError): validate_model_feature_list([field])
    def test_configuration_rejects_unimplemented(self):
        for config in [{"window_mode":"ROLLING"},{"drop_exact_duplicates":True},{"min_entity_history":0}]:
            with self.assertRaises(ValueError): history_config(config)
    def test_sampled_audit_configuration(self):
        other=self.build(self.rows,config={"enable_leakage_audit":False})
        self.assertEqual(other.metadata["validation"]["audit"]["mode"],"SAMPLED_PER_CHUNK")
    def test_compatibility_script(self):
        report=verify(FIXTURE,10)
        self.assertTrue(report["compatible"])
        self.assertEqual(report["sampled_rows"],10)
        self.assertEqual(report["labels"]["valid"],8)
    def test_entity_summary_count_filter(self):
        table=self.engine.entity_summary("airline",min_count=2)
        self.assertTrue(table.history_count.ge(2).all())
        self.assertEqual(self.engine.entity_trend("airline",'["AA"]').daily_count.sum(),7)

    def test_datetime_input_supported(self):
        target=pd.DataFrame([dict(flight_date=pd.Timestamp("2024-01-03"),airline="AA",origin="AAA",destination="BBB")])
        self.assertEqual(self.engine.transform_chunk(target).features.hist_global_count.iloc[0],4)
    def test_single_flags_are_integers(self):
        result=self.query()
        self.assertIsInstance(result.features["hist_route_count"],int)
        self.assertIsInstance(result.features["cold_start_route"],int)
    def test_airline_origin_single_parent_fallback(self):
        result=self.query(airline="NEW",origin="AAA")
        self.assertEqual(result.metadata["hist_airline_origin_fallback_source"],"ORIGIN")
        result=self.query(airline="AA",origin="NEW")
        self.assertEqual(result.metadata["hist_airline_origin_fallback_source"],"AIRLINE")
    def test_independent_datasets_never_merge(self):
        other=self.build(self.rows.iloc[[3]].copy())
        self.assertEqual(self.query("2024-01-03",engine=other).features["hist_global_count"],1)
        self.assertEqual(self.query("2024-01-03").features["hist_global_count"],4)
