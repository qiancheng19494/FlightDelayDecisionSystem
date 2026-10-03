"""仅在训练分区 fit；类别未知忽略，训练期全空数值显式补 0。"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from core.feature_groups import FEATURE_DEFINITIONS
class ModelPreprocessor(TransformerMixin,BaseEstimator):
    def __init__(self,feature_names):self.feature_names=feature_names
    def _clean(self,X):
        if not isinstance(X,pd.DataFrame) or list(X)!=list(self.feature_names):raise ValueError("预处理输入字段或顺序不匹配")
        out=X.copy()
        for n in self.feature_names:
            if FEATURE_DEFINITIONS[n].dtype=="categorical":out[n]=out[n].astype(object).where(out[n].notna(),np.nan)
            else:out[n]=pd.to_numeric(out[n],errors="raise").astype(float).replace([np.inf,-np.inf],np.nan)
        return out
    def fit(self,X,y=None):
        if not len(self.feature_names):raise ValueError("当前组合不包含任何模型输入特征。")
        self.numeric_columns_=[n for n in self.feature_names if FEATURE_DEFINITIONS[n].dtype=="numeric"]
        self.categorical_columns_=[n for n in self.feature_names if FEATURE_DEFINITIONS[n].dtype=="categorical"]
        clean=self._clean(X);self.empty_numeric_columns_=[n for n in self.numeric_columns_ if clean[n].isna().all()]
        steps=[]
        if self.numeric_columns_:steps.append(("numeric",SimpleImputer(strategy="median",keep_empty_features=True),self.numeric_columns_))
        if self.categorical_columns_:steps.append(("categorical",Pipeline([("missing",SimpleImputer(strategy="constant",fill_value="__MISSING__",keep_empty_features=True)),("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=True))]),self.categorical_columns_))
        self.transformer_=ColumnTransformer(steps,sparse_threshold=1.0)
        self.transformer_.fit(clean);self.feature_names_in_=np.asarray(self.feature_names,dtype=object);self.n_features_in_=len(self.feature_names)
        return self
    def transform(self,X):return self.transformer_.transform(self._clean(X))
    def get_feature_names_out(self,input_features=None):return self.transformer_.get_feature_names_out()
