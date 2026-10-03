"""保存第五阶段真实全量测试结果及源码指纹，不覆盖旧阶段证据。"""
import sys,json,unittest,warnings
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_regression_tests import EvidenceResult
from core.validation_evidence import source_digest
if __name__=="__main__":
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        suite=unittest.defaultTestLoader.discover(str(ROOT/"tests"),top_level_dir=str(ROOT))
        r=unittest.TextTestRunner(verbosity=1,resultclass=EvidenceResult).run(suite)
    report={"status":"COMPLETED","generated_at":datetime.now(timezone.utc).isoformat(),"total":r.testsRun,"passed":sum(v=="PASS" for v in r.statuses.values()),"failed":len(r.failures),"errors":len(r.errors),"skipped":len(r.skipped),"warning_count":len(captured),"warnings":[str(w.message) for w in captured],"tests":r.statuses,"source_digest":source_digest(ROOT)}
    (ROOT/"output/reports/stage5_1_test_results.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k not in ("tests","warnings")},ensure_ascii=False));raise SystemExit(0 if r.wasSuccessful() else 1)
