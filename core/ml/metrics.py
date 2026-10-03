"""唯一验证指标入口，阈值固定 0.5，不提供搜索。"""
import numpy as np
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score,confusion_matrix
class MetricsService:
    @staticmethod
    def evaluate(y,probabilities):
        y=np.asarray(y);p=np.asarray(probabilities,dtype=float)
        if y.ndim!=1 or p.shape!=y.shape or not len(y) or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError("标签或概率无效")
        pred=(p>=.5).astype(int);warnings=[];two=len(np.unique(y))==2
        if not two:warnings.append("验证集只有一个类别，ROC-AUC 与 AP 显示 N/A。")
        if pred.sum()==0:warnings.append("无预测正类，Precision 分母为 0，按 zero_division=0 返回 0。")
        if y.sum()==0:warnings.append("无真实正类，Recall 按 zero_division=0 返回 0。")
        return {"threshold":.5,"accuracy":float(accuracy_score(y,pred)),"precision":float(precision_score(y,pred,zero_division=0)),"recall":float(recall_score(y,pred,zero_division=0)),"f1":float(f1_score(y,pred,zero_division=0)),"roc_auc":float(roc_auc_score(y,p)) if two else None,"average_precision":float(average_precision_score(y,p)) if two else None,"confusion_matrix":confusion_matrix(y,pred,labels=[0,1]).tolist(),"warnings":warnings}
