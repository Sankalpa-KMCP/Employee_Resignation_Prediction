import numpy as np, pandas as pd
from src.evaluation.calibration import reliability_table, expected_calibration_error
from src.evaluation.fairness import subgroup_classification_metrics, top_k_representation
from src.prioritization.value_proxy import build_value_proxy
from src.prioritization.ranking import evaluate_ranking

def test_calibration_and_fairness_helpers_smoke():
    df=pd.DataFrame({'y_true':[0,1,0,1,0,1],'y_probability':[.1,.8,.2,.7,.3,.9],'Gender':['A','A','B','B','A','B']})
    assert reliability_table(df.y_true,df.y_probability,5)['count'].sum()==6
    assert 0<=expected_calibration_error(df.y_true,df.y_probability,5)<=1
    assert len(subgroup_classification_metrics(df,'Gender',min_count=1))==2
    assert top_k_representation(df,'Gender',.5)['selected_count'].sum()==3

def test_prioritization_helpers_smoke():
    df=pd.DataFrame({'Performance_Score':[1,2,3,4],'Monthly_Salary':[10,20,30,40],'Years_At_Company':[0,1,2,3],'y_true':[0,1,0,1],'risk':[.1,.9,.2,.8]})
    df['value']=build_value_proxy(df)
    result=evaluate_ranking(df,'risk',.5)
    assert result['selected_count']==2 and result['captured_positive_count']==2
