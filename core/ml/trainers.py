"""RF/XGB 共用预处理与预测接口；不接收任意 CSV 字段。"""
import time
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from core.ml.preprocessor import ModelPreprocessor
from core.ml.metrics import MetricsService
from core.feature_service import audit_result

def xgboost_status():
    try:
        import xgboost
        return True,xgboost.__version__
    except (ImportError,OSError) as exc:return False,str(exc)
class BaseModelTrainer:
    model_type=None
    def __init__(self,config):self.config=config;self.pipeline=None;self.timings={}
    def fit(self,result,train_positions,stage=None):
        if not audit_result(result)["passed"]:raise ValueError("Feature audit 未通过")
        if result.X.shape[1]==0:raise ValueError("当前组合不包含任何模型输入特征。")
        if result.y is None:raise ValueError("需要单独的训练标签")
        X=result.X.iloc[train_positions];y=result.y.iloc[train_positions].to_numpy(dtype=int)
        if len(np.unique(y))!=2:raise ValueError("训练集必须包含两个类别")
        self.positive_count=int(y.sum());self.negative_count=int(len(y)-y.sum());self.parameters=self.make_parameters(y)
        pre=ModelPreprocessor(tuple(X.columns));start=time.perf_counter()
        if stage:stage("拟合预处理（仅训练集）")
        transformed=pre.fit_transform(X);self.timings["preprocessing_seconds"]=time.perf_counter()-start
        model=self.make_estimator();start=time.perf_counter()
        if stage:stage("训练模型（不搜索阈值）")
        model.fit(transformed,y);self.timings["fit_seconds"]=time.perf_counter()-start
        self.pipeline=Pipeline([("preprocessor",pre),("estimator",model)])
        return self
    def predict_proba(self,X):return self.pipeline.predict_proba(X)[:,1]
    def predict(self,X):return (self.predict_proba(X)>=.5).astype(int)
    def evaluate(self,X,y):return MetricsService.evaluate(y,self.predict_proba(X))
    def save_bundle(self,bundle,manager):return manager.save_bundle(bundle)
    def make_parameters(self,y):
        parameters=dict(self.config[self.model_type]);parameters.update(random_state=self.config["random_state"],n_jobs=self.config["n_jobs"])
        return parameters
class RandomForestTrainer(BaseModelTrainer):
    model_type="rf"
    def make_estimator(self):return RandomForestClassifier(**self.parameters)
class XGBoostTrainer(BaseModelTrainer):
    model_type="xgboost"
    def make_parameters(self,y):
        p=super().make_parameters(y)
        if p.get("scale_pos_weight")=="auto":p["scale_pos_weight"]=self.negative_count/self.positive_count
        p.update(objective="binary:logistic",eval_metric="logloss",tree_method="hist")
        return p
    def make_estimator(self):
        ok,reason=xgboost_status()
        if not ok:raise ValueError("XGBoost 依赖不可用："+reason)
        from xgboost import XGBClassifier
        return XGBClassifier(**self.parameters)
