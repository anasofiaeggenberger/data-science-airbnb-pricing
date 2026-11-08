import numpy as np
import pandas as pd
from src.modeling_data import prepare_features
from src.preprocessing import build_preprocessor


def test_imputation_is_learned_only_from_training():
    train = prepare_features(pd.DataFrame({'beds':[1,3,np.nan], 'room_type':['A','A','B']}))
    validation = prepare_features(pd.DataFrame({'beds':[1000,np.nan], 'room_type':['NEW',None]}))
    pre = build_preprocessor(min_category_count=2).fit(train)
    num = pre.named_transformers_['numeric']
    bed_position = pre.transformers_[0][2].index('beds')
    assert num.named_steps['impute'].statistics_[bed_position] == 2
    before = num.named_steps['scale'].mean_.copy()
    transformed = pre.transform(validation)
    assert transformed.shape[1] == pre.transform(train).shape[1]
    np.testing.assert_array_equal(before, num.named_steps['scale'].mean_)
    values = transformed.toarray() if hasattr(transformed,'toarray') else transformed
    assert np.isfinite(values).all()
    assert 'host_identity_verified' not in pre.feature_names_in_[pre._transformer_to_input_indices['categorical']]
