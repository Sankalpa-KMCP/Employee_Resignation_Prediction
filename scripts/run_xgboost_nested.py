from pathlib import Path
import json,time
import numpy as np,pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from xgboost import XGBClassifier

ROOT=Path('.')
data=pd.read_csv(ROOT/'data/raw/Extended_Employee_Performance_and_Productivity_Data.csv')
reg=pd.read_csv(ROOT/'data/processed/split_registry.csv')
dev_ids=set(reg.loc[reg.partition=='development','Employee_ID'])
dev=data[data.Employee_ID.isin(dev_ids)].copy()
reg_dev=reg[reg.partition=='development'][['Employee_ID','cv_fold']]
dev=dev.merge(reg_dev,on='Employee_ID',how='left',validate='one_to_one')
cat=['Department','Gender','Job_Title','Education_Level']
features=[c for c in data.columns if c not in ['Employee_ID','Hire_Date','Resigned']]
num=[c for c in features if c not in cat]
candidates=[
 ('XGB01',dict(n_estimators=400,learning_rate=.05,max_depth=5,subsample=.8,colsample_bytree=.8,reg_alpha=.1,reg_lambda=1.0)),
 ('XGB02',dict(n_estimators=500,learning_rate=.05,max_depth=3,subsample=.8,colsample_bytree=.8,reg_alpha=.1,reg_lambda=1.0)),
 ('XGB03',dict(n_estimators=400,learning_rate=.05,max_depth=7,subsample=.8,colsample_bytree=.8,reg_alpha=.1,reg_lambda=1.0)),
 ('XGB04',dict(n_estimators=700,learning_rate=.02,max_depth=5,subsample=.8,colsample_bytree=.8,reg_alpha=.1,reg_lambda=1.0)),
 ('XGB05',dict(n_estimators=300,learning_rate=.10,max_depth=5,subsample=.8,colsample_bytree=.8,reg_alpha=.1,reg_lambda=1.0)),
 ('XGB06',dict(n_estimators=500,learning_rate=.05,max_depth=5,subsample=.8,colsample_bytree=.8,reg_alpha=1.0,reg_lambda=5.0)),
 ('XGB07',dict(n_estimators=500,learning_rate=.05,max_depth=5,subsample=.7,colsample_bytree=.7,reg_alpha=.1,reg_lambda=1.0)),
 ('XGB08',dict(n_estimators=600,learning_rate=.03,max_depth=4,subsample=.8,colsample_bytree=.8,reg_alpha=.5,reg_lambda=5.0)),
]
base=dict(objective='binary:logistic',eval_metric='aucpr',tree_method='hist',random_state=42,verbosity=0,n_jobs=-1,early_stopping_rounds=50)
sel_rows=[]; fold_rows=[]; preds=[]
start_all=time.time()
for fold in sorted(dev.cv_fold.unique()):
    tr=dev[dev.cv_fold!=fold].copy(); va=dev[dev.cv_fold==fold].copy()
    tr_i, in_i=train_test_split(np.arange(len(tr)),test_size=.15,random_state=42+int(fold),stratify=tr.Resigned.astype(int))
    tr_inner=tr.iloc[tr_i]; inner=tr.iloc[in_i]
    enc=OneHotEncoder(handle_unknown='ignore',sparse_output=True)
    Xtr_cat=enc.fit_transform(tr_inner[cat]); Xin_cat=enc.transform(inner[cat])
    from scipy.sparse import hstack, csr_matrix
    Xtr=hstack([csr_matrix(tr_inner[num].to_numpy(float)),Xtr_cat],format='csr')
    Xin=hstack([csr_matrix(inner[num].to_numpy(float)),Xin_cat],format='csr')
    ytr=tr_inner.Resigned.astype(int).to_numpy(); yin=inner.Resigned.astype(int).to_numpy()
    spw=float((ytr==0).sum()/(ytr==1).sum())
    best=None
    for cid,params in candidates:
        t=time.time(); model=XGBClassifier(**base,**params,scale_pos_weight=spw)
        model.fit(Xtr,ytr,eval_set=[(Xin,yin)],verbose=False)
        p=model.predict_proba(Xin)[:,1]
        ap=average_precision_score(yin,p)
        row={'outer_fold':int(fold),'candidate_id':cid,'inner_average_precision':float(ap),'best_iteration':int(model.best_iteration) if model.best_iteration is not None else params['n_estimators']-1,'fit_seconds':time.time()-t,**params}
        sel_rows.append(row)
        if best is None or ap>best[0]: best=(ap,cid,params,row['best_iteration'])
    ap,cid,params,best_iter=best
    # refit encoder on complete outer-train, then refit using fixed selected tree count; no outer-val early stopping
    enc2=OneHotEncoder(handle_unknown='ignore',sparse_output=True)
    Xtr_cat=enc2.fit_transform(tr[cat]); Xva_cat=enc2.transform(va[cat])
    Xtr2=hstack([csr_matrix(tr[num].to_numpy(float)),Xtr_cat],format='csr')
    Xva=hstack([csr_matrix(va[num].to_numpy(float)),Xva_cat],format='csr')
    ytr2=tr.Resigned.astype(int).to_numpy(); yva=va.Resigned.astype(int).to_numpy()
    spw2=float((ytr2==0).sum()/(ytr2==1).sum())
    final_params=dict(params); final_params['n_estimators']=max(1,best_iter+1)
    finalbase={k:v for k,v in base.items() if k!='early_stopping_rounds'}
    final=XGBClassifier(**finalbase,**final_params,scale_pos_weight=spw2)
    t=time.time(); final.fit(Xtr2,ytr2,verbose=False); fit_s=time.time()-t
    pv=final.predict_proba(Xva)[:,1]
    fold_rows.append({'cv_fold':int(fold),'selected_candidate':cid,'selected_inner_ap':float(ap),'selected_n_estimators':int(final_params['n_estimators']),'average_precision':float(average_precision_score(yva,pv)),'roc_auc':float(roc_auc_score(yva,pv)),'brier_score':float(brier_score_loss(yva,pv)),'fit_seconds_refit':fit_s})
    for emp,y,p in zip(va.Employee_ID, yva,pv): preds.append((int(emp),int(fold),int(y),float(p),cid))
    print('fold',fold,'selected',cid,'inner AP',ap,'outer AP',fold_rows[-1]['average_precision'],flush=True)

out=ROOT/'results/models/xgboost/strict_nested'; out.mkdir(parents=True,exist_ok=True)
pd.DataFrame(sel_rows).to_csv(out/'candidate_selection.csv',index=False)
pd.DataFrame(fold_rows).to_csv(out/'fold_selection.csv',index=False)
op=pd.DataFrame(preds,columns=['Employee_ID','cv_fold','y_true','y_probability','selected_candidate']).sort_values('Employee_ID')
op['model_name']='XGBoost'; op['y_pred_0_5']=(op.y_probability>=.5).astype(int)
op[['Employee_ID','cv_fold','y_true','model_name','y_probability','y_pred_0_5','selected_candidate']].to_csv(out/'oof_predictions.csv',index=False)
y=op.y_true.to_numpy(); p=op.y_probability.to_numpy(); folds=pd.DataFrame(fold_rows)
summary={'model_name':'XGBoost','protocol':'strict_nested_candidate_selection','n_candidates':len(candidates),'n_outer_folds':5,'n_development_rows':len(op),'pooled_average_precision':float(average_precision_score(y,p)),'pooled_roc_auc':float(roc_auc_score(y,p)),'pooled_brier_score':float(brier_score_loss(y,p)),'cv_average_precision_mean':float(folds.average_precision.mean()),'cv_average_precision_sd':float(folds.average_precision.std(ddof=1)),'total_runtime_seconds':time.time()-start_all}
json.dump(summary,open(out/'summary_metrics.json','w'),indent=2)
json.dump({'candidate_ids':[c[0] for c in candidates],'selection_metric':'inner_average_precision','inner_validation_fraction':.15,'class_weighting':'fold-local scale_pos_weight','outer_validation_used_for_selection':False},open(out/'run_metadata.json','w'),indent=2)
print(json.dumps(summary,indent=2))
