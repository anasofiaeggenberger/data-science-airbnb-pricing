import json
import numpy as np
import pandas as pd
import pytest
from src.recommend import recommend, validate_listing


def test_recommend_contract_and_bad_inputs():
    listing=json.loads(open('examples/listing.json').read())
    class Model:
        def predict(self,X): return pd.DataFrame([[100,200,300,400]],columns=['p10','p35','p50','p90'])
    assert recommend(Model(),listing)['recommended_price']==200
    for changes in [{'latitude':100},{'accommodates':0},{'beds':-1},{'longitude':np.inf},{'room_type':None}]:
        with pytest.raises(ValueError): validate_listing({**listing,**changes})
    validate_listing({**listing,'room_type':'unknown','beds':None})
    class Crossed:
        def predict(self,X): return pd.DataFrame([[400,200,300,100]],columns=['p10','p35','p50','p90'])
    with pytest.raises(ValueError,match='ordenados'): recommend(Crossed(),listing)
