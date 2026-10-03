"""可追踪的预处理/估计器联合资产。"""
from dataclasses import dataclass
@dataclass
class ModelBundle:
    metadata: dict
    pipeline: object
    @property
    def model_id(self):return self.metadata["model_id"]
    @property
    def preprocessor(self):return self.pipeline.named_steps["preprocessor"]
    @property
    def estimator(self):return self.pipeline.named_steps["estimator"]
