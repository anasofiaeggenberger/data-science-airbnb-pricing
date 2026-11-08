import pandas as pd
from tests.test_ridge_model import sample
from src.train_ridge import search_ridge


def test_search_covers_all_rows_and_same_folds():
    X,y,groups=sample()
    metadata=pd.DataFrame({'id':range(len(X)), 'host_id':groups})
    best,results,oof=search_ridge(X,y,metadata,alphas=(1,10))
    assert best in (1,10) and len(results)==10
    assert not oof.isna().any().any()
    assert results.groupby('alpha').n_validation.sum().eq(len(X)).all()
