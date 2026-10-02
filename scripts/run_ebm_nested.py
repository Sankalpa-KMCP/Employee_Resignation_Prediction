"""Run bounded strict-nested EBM tuning on the frozen development folds.

This script is intentionally separate from saved historical EBM results. It requires
`interpret` and does not access the formal holdout. Execute only in a compatible
project environment, then review the generated evidence before updating the model
comparison registry/status.
"""
from pathlib import Path
import json, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from interpret.glassbox import ExplainableBoostingClassifier

ROOT=Path(__file__).resolve().parents[1]
data=pd.read_csv(ROOT/'data/raw/Extended_Employee_Performance_and_Productivity_Data.csv')
reg=pd.read_csv(ROOT/'data/processed/split_registry.csv')
dev=data.merge(reg[['Employee_ID','partition','cv_fold']],on='Employee_ID',validate='one_to_one')
dev=dev[dev.partition=='development'].copy()
features=[c for c in data.columns if c not in ['Employee_ID','Hire_Date','Resigned']]
cat=['Department','Gender','Job_Title','Education_Level']
feature_types=['nominal' if c in cat else 'continuous' for c in features]
for c in cat: dev[c]=dev[c].astype(str)

CANDIDATES=[
 ('EBM01',dict(learning_rate=.015,max_leaves=3,min_samples_leaf=4,interactions=5)),
 ('EBM02',dict(learning_rate=.005,max_leaves=3,min_samples_leaf=4,interactions=5)),
 ('EBM03',dict(learning_rate=.03,max_leaves=3,min_samples_leaf=4,interactions=5)),
 ('EBM04',dict(learning_rate=.015,max_leaves=2,min_samples_leaf=4,interactions=5)),
 ('EBM05',dict(learning_rate=.015,max_leaves=3,min_samples_leaf=2,interactions=5)),
 ('EBM06',dict(learning_rate=.015,max_leaves=3,min_samples_leaf=10,interactions=5)),
 ('EBM07',dict(learning_rate=.015,max_leaves=3,min_samples_leaf=4,interactions=0)),
 ('EBM08',dict(learning_rate=.015,max_leaves=3,min_samples_leaf=4,interactions=10)),
]
common=dict(random_state=42,n_jobs=-1,max_rounds=5000,early_stopping_rounds=50)
selection=[]; folds=[]; preds=[]; start=time.time()
for fold in sorted(dev.cv_fold.unique()):
    outer_train=dev[dev.cv_fold!=fold].copy(); outer_val=dev[dev.cv_fold==fold].copy()
    tr_idx,in_idx=train_test_split(np.arange(len(outer_train)),test_size=.15,random_state=42+int(fold),stratify=outer_train.Resigned.astype(int))
    inner_train=outer_train.iloc[tr_idx]; inner_val=outer_train.iloc[in_idx]
    best=None
    for cid,params in CANDIDATES:
        t=time.time(); m=ExplainableBoostingClassifier(feature_names=features,feature_types=feature_types,**common,**params)
        m.fit(inner_train[features],inner_train.Resigned.astype(int))
        prob=m.predict_proba(inner_val[features])[:,1]; ap=average_precision_score(inner_val.Resigned.astype(int),prob)
        row={'outer_fold':int(fold),'candidate_id':cid,'inner_average_precision':float(ap),'fit_seconds':time.time()-t,**params}; selection.append(row)
        if best is None or ap>best[0]: best=(ap,cid,params)
    ap,cid,params=best
    final=ExplainableBoostingClassifier(feature_names=features,feature_types=feature_types,**common,**params)
    t=time.time(); final.fit(outer_train[features],outer_train.Resigned.astype(int)); fit_s=time.time()-t
    y=outer_val.Resigned.astype(int).to_numpy(); p=final.predict_proba(outer_val[features])[:,1]
    folds.append({'cv_fold':int(fold),'selected_candidate':cid,'selected_inner_ap':float(ap),'average_precision':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),'brier_score':float(brier_score_loss(y,p)),'fit_seconds_refit':fit_s})
    preds.extend((int(e),int(fold),int(yy),float(pp),cid) for e,yy,pp in zip(outer_val.Employee_ID,y,p))
out=ROOT/'results/models/ebm/strict_nested'; out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(selection).to_csv(out/'candidate_selection.csv',index=False); pd.DataFrame(folds).to_csv(out/'fold_selection.csv',index=False)
op=pd.DataFrame(preds,columns=['Employee_ID','cv_fold','y_true','y_probability','selected_candidate']).sort_values('Employee_ID'); op['model_name']='EBM'; op['y_pred_0_5']=(op.y_probability>=.5).astype(int); op.to_csv(out/'oof_predictions.csv',index=False)
y=op.y_true.to_numpy(); p=op.y_probability.to_numpy(); fm=pd.DataFrame(folds)
summary={'model_name':'EBM','protocol':'strict_nested_candidate_selection','n_candidates':len(CANDIDATES),'n_outer_folds':5,'n_development_rows':len(op),'pooled_average_precision':float(average_precision_score(y,p)),'pooled_roc_auc':float(roc_auc_score(y,p)),'pooled_brier_score':float(brier_score_loss(y,p)),'cv_average_precision_mean':float(fm.average_precision.mean()),'cv_average_precision_sd':float(fm.average_precision.std(ddof=1)),'total_runtime_seconds':time.time()-start}
(out/'summary_metrics.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
