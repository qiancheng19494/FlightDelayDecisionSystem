"""阶段四边界、结果独立性、注册来源与统一接口回归。"""
import copy
import tempfile
import unittest
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal,assert_series_equal
from core.temporal_features import TemporalFeatureEngineer
from core.congestion_features import CongestionFeatureEngineer
from core.feature_groups import *
from core.feature_pipeline import FeaturePipeline
from core.feature_validator import FeatureValidator
from core.historical_features import HistoricalFeatureEngine
FIXTURE=Path(__file__).parent/"fixtures/schedule_tiny.csv"
def data():return pd.read_csv(FIXTURE,keep_default_na=False)
def record(time=0,date="2024-01-01"):
    return pd.DataFrame([dict(flight_date=date,airline="AA",origin="AAA",destination="BBB",scheduled_departure=time,scheduled_arrival=115,scheduled_elapsed_time=105,distance=400)])
class TemporalTests(unittest.TestCase):
    def test_boundaries(self):
        for time,hour,period in [(0,0,"NIGHT"),(5,0,"NIGHT"),(559,5,"NIGHT"),(600,6,"MORNING"),(1159,11,"MORNING"),(1200,12,"AFTERNOON"),(1759,17,"AFTERNOON"),(1800,18,"EVENING"),(2359,23,"EVENING")]:
            with self.subTest(time=time):
                x=TemporalFeatureEngineer().transform(record(time));self.assertEqual(x.dep_hour[0],hour);self.assertEqual(x.time_of_day[0],period)
    def test_2400_cycle(self):
        a=TemporalFeatureEngineer().transform(record(0));b=TemporalFeatureEngineer().transform(record(2400));assert_frame_equal(a,b)
    def test_sunday_monday(self):
        a=TemporalFeatureEngineer().transform(record(date="2024-01-07"));b=TemporalFeatureEngineer().transform(record(date="2024-01-08"))
        self.assertEqual(a.day_of_week[0],6);self.assertEqual(b.day_of_week[0],0);self.assertAlmostEqual(a.day_of_week_sin[0],np.sin(12*np.pi/7))
    def test_december_january(self):
        x=TemporalFeatureEngineer().transform(record(date="2024-12-31"));self.assertAlmostEqual(x.month_sin[0],-.5);self.assertEqual(TemporalFeatureEngineer().transform(record()).month_cos[0],1)
    def test_invalid_departure(self):
        x=TemporalFeatureEngineer().transform(record(2460));self.assertTrue(pd.isna(x.dep_hour[0]));self.assertEqual(x.dep_time_missing[0],1)
    def test_invalid_arrival(self):
        f=record();f["scheduled_arrival"]="bad";x=TemporalFeatureEngineer().transform(f);self.assertTrue(pd.isna(x.arr_hour[0]));self.assertEqual(x.arr_time_missing[0],1)
    def test_invalid_date(self):
        x=TemporalFeatureEngineer().transform(record(date="bad"));self.assertTrue(pd.isna(x.is_weekend[0]));self.assertEqual(x.flight_date_missing[0],1)
    def test_datetime_input(self):
        f=record();f.flight_date=pd.to_datetime(f.flight_date);self.assertEqual(TemporalFeatureEngineer().transform(f).month[0],1)
    def test_weekend(self):self.assertEqual(TemporalFeatureEngineer().transform(record(date="2024-01-06")).is_weekend[0],1)
    def test_cross_midnight_preserves_duration(self):
        f=record(2330);x=FeaturePipeline().build_features(f,"COMBO_AT").X;self.assertEqual(x.scheduled_elapsed_time[0],105);self.assertEqual(x.arr_hour[0],1)
    def test_missing_optional(self):
        f=record().drop(columns=["scheduled_departure","scheduled_arrival"]);x=TemporalFeatureEngineer().transform(f);self.assertEqual(x.dep_time_missing[0],1)
    def test_raw_input_unchanged(self):
        f=data();before=f.copy(deep=True);TemporalFeatureEngineer().transform(f);assert_frame_equal(f,before)
class CongestionTests(unittest.TestCase):
    def setUp(self):self.f=data();self.engine=CongestionFeatureEngineer().fit(self.f);self.x=self.engine.transform(self.f)
    def test_origin_hour(self):self.assertEqual(self.x.origin_hour_scheduled_departures[0],3)
    def test_destination_hour(self):self.assertEqual(self.x.destination_hour_scheduled_arrivals[0],2)
    def test_airline_hour(self):self.assertEqual(self.x.airline_hour_scheduled_departures[0],3)
    def test_route_direction(self):self.assertEqual(self.x.route_day_scheduled_flights[0],2);self.assertEqual(self.x.route_day_scheduled_flights[3],1)
    def test_date_isolation(self):self.assertEqual(self.x.origin_day_scheduled_departures[4],1)
    def test_airport_isolation(self):self.assertEqual(self.x.origin_hour_scheduled_departures[3],1)
    def test_airline_isolation(self):self.assertEqual(self.x.airline_hour_scheduled_departures[2],1)
    def test_destination_day(self):self.assertEqual(self.x.destination_day_scheduled_arrivals[0],2)
    def test_cancelled_included(self):self.assertEqual(self.x.route_day_scheduled_flights[1],2)
    def test_missing_time_day_retained(self):self.assertTrue(pd.isna(self.x.origin_hour_scheduled_departures[7]));self.assertEqual(self.x.origin_day_scheduled_departures[7],2)
    def test_missing_context(self):
        x=CongestionFeatureEngineer().transform(self.f);self.assertTrue(x[list(CONGESTION_KEYS)].isna().all().all());self.assertTrue(x.congestion_context_available.eq(0).all())
    def test_uncovered_date(self):self.assertTrue(self.engine.transform(record(date="2030-01-01"))[list(CONGESTION_KEYS)].isna().all().all())
    def test_empty_context(self):self.assertTrue(CongestionFeatureEngineer().fit(self.f.iloc[:0]).transform(self.f).congestion_context_available.eq(0).all())
    def test_chunks_equal(self):assert_frame_equal(self.x,CongestionFeatureEngineer().fit_chunks([self.f.iloc[:3],self.f.iloc[3:]]).transform(self.f))
    def test_shuffle_equal(self):assert_frame_equal(self.x,CongestionFeatureEngineer().fit(self.f.sample(frac=1,random_state=9)).transform(self.f))
    def test_unknown_entity_zero_in_known_context(self):
        f=record(1000);f.origin="ZZZ";self.assertEqual(self.engine.transform(f).origin_hour_scheduled_departures[0],0)
    def test_plan_change_changes_counts(self):
        f=self.f.copy();f.loc[0,"CRS_DEP_TIME"]=1400;x=CongestionFeatureEngineer().fit(f).transform(f);self.assertEqual(x.origin_hour_scheduled_departures[0],1)
    def test_whitespace_keys(self):
        f=record(1000);f.origin=" aaa ";self.assertEqual(self.engine.transform(f).origin_hour_scheduled_departures[0],3)
    def test_invalid_date_unavailable(self):self.assertEqual(self.engine.transform(record(date="bad")).congestion_context_available[0],0)
def independence_test(column):
    def test(self):
        f=self.f.copy();f[column]=99999
        assert_frame_equal(self.x,CongestionFeatureEngineer().fit(f).transform(f))
    return test
for column in ["DEP_DELAY","ARR_DELAY","CANCELLED","DEP_TIME","ARR_TIME","WHEELS_OFF","WHEELS_ON","TAXI_OUT","AIR_TIME"]:setattr(CongestionTests,"test_independent_"+column.lower(),independence_test(column))
class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.h=HistoricalFeatureEngine(source_path=FIXTURE,cache_root=cls.temp.name);cls.h.build_history_cache();cls.p=FeaturePipeline(cls.h)
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()
    def test_all_sixteen_unions(self):
        registry=FeatureCombinationRegistry();self.assertEqual(len(registry.list_combinations()),16)
        for c in registry.list_combinations():
            with self.subTest(combo=c.name):
                x=self.p.build_features(data(),c,schedule_context=data()).X;self.assertEqual(list(x),[n for g in c.groups for n in GROUP_FIELDS[g]]);self.assertEqual(len(x.columns),len(set(x)))
    def test_determinism(self):assert_frame_equal(self.p.build_features(data(),schedule_context=data()).X,self.p.build_features(data(),schedule_context=data()).X)
    def test_single_batch(self):
        f=data();a=self.p.build_features(f,schedule_context=f).X.iloc[:1];b=self.p.transform_single(f.iloc[0].to_dict(),schedule_context=f).X;assert_frame_equal(a,b)
    def test_train_inference_same_X(self):assert_frame_equal(self.p.build_features(data(),mode="TRAIN").X,self.p.build_features(data(),mode="INFERENCE").X)
    def test_label_separate(self):
        r=self.p.build_features(data(),mode="TRAIN");self.assertEqual(r.y.name,"delay_label");self.assertNotIn("delay_label",r.X);self.assertTrue(pd.isna(r.y[1]))
    def test_train_requires_label(self):
        with self.assertRaises(ValueError):self.p.build_features(record(),"COMBO_A",mode="TRAIN")
    def test_inference_has_no_y(self):self.assertIsNone(self.p.build_features(data()).y)
    def test_H_missing_cache_rejected(self):
        with self.assertRaises(ValueError):FeaturePipeline().build_features(data(),"COMBO_H")
    def test_invalid_H_date_rejected(self):
        with self.assertRaises(ValueError):self.p.build_features(record(date="bad"),"COMBO_H")
    def test_without_H_invalid_date_retained(self):self.assertTrue(self.p.build_features(record(date="bad"),"COMBO_T").X.month.isna().all())
    def test_without_context_not_one(self):
        r=self.p.transform_single(record().iloc[0].to_dict());self.assertTrue(r.X[list(CONGESTION_KEYS)].isna().all().all());self.assertTrue(r.warnings)
    def test_duplicate_index(self):
        f=data();f.index=[1]*len(f);self.assertEqual(list(self.p.build_features(f).X.index),list(f.index))
    def test_target_outcomes_do_not_change_features(self):
        a=data();b=a.copy();b.DEP_DELAY=999;b.CANCELLED=1;assert_frame_equal(self.p.build_features(a,schedule_context=a).X,self.p.build_features(b,schedule_context=b).X)
    def test_plan_change_does_not_change_H(self):
        a=data();b=a.copy();b.CRS_DEP_TIME=999;assert_frame_equal(self.p.build_features(a,"COMBO_H").X,self.p.build_features(b,"COMBO_H").X)
    def test_airline_lookup_changes(self):
        a=record(date="2024-01-02");b=a.copy();b.airline="ZZ";self.assertGreater(self.p.build_features(a,"COMBO_H").X.hist_airline_count[0],self.p.build_features(b,"COMBO_H").X.hist_airline_count[0])
    def test_current_H_unchanged_future_H_changed(self):
        with tempfile.TemporaryDirectory() as directory:
            f=data();f.loc[0,"DEP_DELAY"]=100;p=Path(directory)/"changed.csv";f.to_csv(p,index=False)
            h=HistoricalFeatureEngine(source_path=p,cache_root=directory);h.build_history_cache();a=self.p.build_features(data(),schedule_context=data());b=FeaturePipeline(h).build_features(data(),schedule_context=data())
            assert_frame_equal(a.X.iloc[:4],b.X.iloc[:4]);assert_frame_equal(a.X[list(BASE_FEATURES_A+TEMPORAL_FEATURES+CONGESTION_FEATURES)],b.X[list(BASE_FEATURES_A+TEMPORAL_FEATURES+CONGESTION_FEATURES)])
            self.assertNotEqual(a.X.hist_global_mean_dep_delay[4],b.X.hist_global_mean_dep_delay[4])
    def test_base_time_minutes(self):self.assertEqual(self.p.build_features(record(2400),"COMBO_A").X.scheduled_departure[0],1440)
    def test_tail_flight_number_excluded(self):self.assertNotIn("tail_number",BASE_FEATURES_A);self.assertNotIn("flight_number",BASE_FEATURES_A)
    def test_bad_mode(self):
        with self.assertRaises(ValueError):self.p.build_features(data(),mode="FIT")
    def test_metadata(self):
        r=self.p.build_features(data());self.assertEqual(r.metadata["feature_count"],80);self.assertEqual(r.metadata["feature_schema_version"],1);self.assertTrue(r.metadata["history_audit"]["passed"]);self.assertFalse(r.metadata["paper_combinations_confirmed"])
class ValidatorTests(unittest.TestCase):
    def setUp(self):self.r=FeaturePipeline().build_features(record(),"COMBO_ATC");self.v=FeatureValidator()
    def invalid(self,x=None,**kwargs):
        with self.assertRaises(ValueError):self.v.validate(self.r.X if x is None else x,"COMBO_ATC",**kwargs)
    def test_forbidden_alias(self):self.invalid(self.r.X.assign(DEP_DELAY=1))
    def test_unknown_renaming(self):self.invalid(self.r.X.assign(dep_delay_raw=1))
    def test_duplicate_columns(self):self.invalid(pd.concat([self.r.X,self.r.X[["distance"]]],axis=1))
    def test_unexpected_object(self):
        x=self.r.X.copy();x.distance="bad";self.invalid(x)
    def test_schema_mismatch(self):self.invalid(schema_version=99)
    def test_changed_lineage(self):
        lineage=copy.deepcopy(self.r.metadata["lineage"]);lineage["distance"]["source_fields"]=("departure_delay",);self.invalid(lineage=lineage)
    def test_C_lineage_cancelled(self):
        lineage=copy.deepcopy(self.r.metadata["lineage"]);lineage["origin_day_scheduled_departures"]["source_fields"]=("cancelled",);self.invalid(lineage=lineage)
    def test_y_not_X(self):self.invalid(self.r.X.assign(y=1))
    def test_field_order(self):self.invalid(self.r.X[self.r.X.columns[::-1]])
    def test_combo_tamper(self):
        c=FeatureCombinationRegistry().get_combination("COMBO_A")
        with self.assertRaises(ValueError):FeatureCombinationRegistry().get_combination(replace(c,feature_names=("departure_delay",)))
    def test_report_false(self):self.assertFalse(self.v.validate(self.r.X.assign(y=1),"COMBO_ATC",raise_on_error=False)["passed"])
