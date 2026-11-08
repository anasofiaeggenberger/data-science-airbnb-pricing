import pandas as pd
import pytest
from src.modeling_data import load_development, make_folds


def test_loader_preserves_ids_missing_values_and_host_groups(tmp_path):
    path = tmp_path / 'june.csv'
    pd.DataFrame({'id': ['9007199254740993','2','3','4','5','6'],
                  'host_id': ['a','a','b','c','d','e'],
                  'price': ['$100','$200','$300','$400','$500',None],
                  'beds': [1,None,2,3,4,5]}).to_csv(path,index=False)
    X, y, meta = load_development(path)
    assert len(X) == 5 and X.beds.isna().sum() == 1
    assert '9007199254740993' in meta.id.tolist()
    assert not {'price','price_mxn','id','host_id'} & set(X)
    folds = make_folds(meta, 3)
    seen = []
    for train, val in folds:
        assert not set(meta.host_id.iloc[train]) & set(meta.host_id.iloc[val])
        seen.extend(val)
    assert sorted(seen) == list(range(len(X)))
    assert y.tolist() == [200,300,400,500,100]


def test_duplicate_ids_rejected(tmp_path):
    path=tmp_path/'bad.csv'
    pd.DataFrame({'id':['1','1'],'host_id':['a','b'],'price':[1,2]}).to_csv(path,index=False)
    with pytest.raises(ValueError, match='duplicados'):
        load_development(path)
