import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from src.ridge_model import RidgeQuantileModel


def sample():
    rng=np.random.default_rng(7)
    X=pd.DataFrame({'accommodates':np.tile([1,2,3,4],12), 'room_type':np.tile(['A','B'],24)})
    y=np.exp(6 + .2*X.accommodates + rng.normal(0,.15,len(X)))
    return X,y,np.repeat(np.arange(12),4)


def test_quantiles_are_positive_ordered_and_survive_serialization(tmp_path):
    X,y,groups=sample()
    model=RidgeQuantileModel().fit(X,y,groups)
    probe=pd.DataFrame({'accommodates':[2,None], 'room_type':['unseen',None]},index=[90,91])
    predictions=model.predict(probe)
    assert predictions.index.tolist()==[90,91]
    assert list(predictions)==['p10','p35','p50','p90']
    assert (predictions.to_numpy()>0).all()
    assert (np.diff(predictions.to_numpy(),axis=1)>=0).all()
    path=tmp_path/'model.joblib'; joblib.dump(model,path)
    pd.testing.assert_frame_equal(predictions,joblib.load(path).predict(probe))
    assert model.calibration_method_=='grouped_oof_log_residuals'


def test_unfitted_and_invalid_target_are_rejected():
    X,y,groups=sample()
    with pytest.raises(NotFittedError): RidgeQuantileModel().predict(X)
    y.iloc[0]=0
    with pytest.raises(ValueError): RidgeQuantileModel().fit(X,y,groups)


def test_calibration_does_not_use_outer_validation():
    X,y,groups=sample()
    model=RidgeQuantileModel().fit(X.iloc[:32],y.iloc[:32],groups[:32])
    original=model.residual_quantiles_.copy()
    model.predict(X.iloc[32:].assign(accommodates=100))
    np.testing.assert_array_equal(original,model.residual_quantiles_)
    assert model.n_training_rows_==32
