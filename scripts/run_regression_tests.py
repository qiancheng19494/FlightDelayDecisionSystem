"""执行全量测试，保存真实通过/失败/跳过/warning 计数和代码指纹。"""
import json
import sys
import unittest
import warnings
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from core.validation_evidence import source_digest

class EvidenceResult(unittest.TextTestResult):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.statuses={}
    def addSuccess(self,test):
        super().addSuccess(test);self.statuses[test.id()]="PASS"
    def addFailure(self,test,error):
        super().addFailure(test,error);self.statuses[test.id()]="FAIL"
    def addError(self,test,error):
        super().addError(test,error);self.statuses[test.id()]="ERROR"
    def addSkip(self,test,reason):
        super().addSkip(test,reason);self.statuses[test.id()]="SKIP"

if __name__=="__main__":
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),top_level_dir=str(ROOT))
        result=unittest.TextTestRunner(verbosity=1,resultclass=EvidenceResult).run(suite)
    evidence={"status":"COMPLETED","generated_at":datetime.now(timezone.utc).isoformat(),"total":result.testsRun,"passed":sum(v=="PASS" for v in result.statuses.values()),"failed":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),"warning_count":len(recorded),"warnings":[str(w.message) for w in recorded],"tests":result.statuses,"source_digest":source_digest(ROOT)}
    path=ROOT/"output/reports/stage3_test_results.json"
    path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in evidence.items() if k not in ["tests","warnings"]},ensure_ascii=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
