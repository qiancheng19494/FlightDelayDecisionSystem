"""本机 JSON 模型目录，原子更新；归档不删除文件。"""
import json,re,threading
from pathlib import Path
from uuid import uuid4
_LOCK=threading.RLock()
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+"."+uuid4().hex+".tmp")
    try:tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8");tmp.replace(path)
    finally:tmp.unlink(missing_ok=True)
def validate_id(identity):
    if not isinstance(identity,str) or not re.fullmatch(r"[a-z0-9_]{8,100}",identity):raise ValueError("模型 ID 不合法")
    return identity
class ModelRegistry:
    def __init__(self,root):self.root=Path(root);self.path=self.root/"models/registry.json"
    def read(self):
        if not self.path.exists():return {"schema_version":1,"active_model_id":None,"models":{}}
        try:
            r=json.loads(self.path.read_text(encoding="utf-8"))
            if r["schema_version"]!=1 or not isinstance(r["models"],dict):raise ValueError()
            return r
        except (ValueError,KeyError,TypeError):raise ValueError("模型注册表损坏，保留原文件并拒绝覆盖") from None
    def register_model(self,metadata):
        identity=validate_id(metadata["model_id"])
        with _LOCK:
            r=self.read()
            if identity in r["models"]:raise ValueError("模型 ID 已存在")
            r["models"][identity]=metadata;atomic_json(self.path,r)
    def list_models(self):return list(self.read()["models"].values())
    def get_model(self,identity):
        validate_id(identity)
        try:return self.read()["models"][identity]
        except KeyError:raise ValueError("模型未登记，不自动加载外部 joblib") from None
    def set_active_model(self,identity):
        # 必须通过 Manager 的完整兼容检查，不允许直接改 active 绕过检查。
        from core.ml.model_manager import ModelManager
        ModelManager(self.root).load_model(identity)
        with _LOCK:
            r=self.read();r["active_model_id"]=identity;atomic_json(self.path,r)
    def archive_model(self,identity):
        with _LOCK:
            self.get_model(identity);r=self.read();r["models"][identity]["status"]="ARCHIVED"
            if r["active_model_id"]==identity:r["active_model_id"]=None
            atomic_json(self.path,r)
