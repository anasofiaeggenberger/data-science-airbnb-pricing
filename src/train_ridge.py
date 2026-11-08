"""Búsqueda de Ridge en junio con métricas locales provisionales.

Pendiente de sustituir los cálculos locales por evaluate_quantile_predictions
cuando src.metrics se integre en esta rama y de regenerar la comparación común.
"""
import argparse
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error, mean_pinball_loss

from src.modeling_data import load_development, make_folds
from src.preparation import DEFAULT_JUNE_PATH, REPO_ROOT
from src.ridge_model import RidgeQuantileModel


METRICS_STATUS = 'provisional_pending_shared_metrics'
METRICS_IMPLEMENTATION = 'sklearn_metrics_and_local_interval_calculations'


def search_ridge(X, y, metadata, alphas=(0.1, 1.0, 10.0, 100.0), include_optional=False):
    """Selecciona alpha mediante métricas provisionales, no la evaluación común."""
    folds = make_folds(metadata)
    rows = []
    predictions = {}
    for alpha in alphas:
        oof = pd.DataFrame(index=X.index, columns=['p10','p35','p50','p90'], dtype=float)
        for fold, (train, valid) in enumerate(folds):
            model = RidgeQuantileModel(alpha=alpha, include_optional=include_optional)
            model.fit(X.iloc[train], y.iloc[train], groups=metadata.host_id.iloc[train])
            pred = model.predict(X.iloc[valid])
            oof.iloc[valid] = pred.to_numpy()
            truth = y.iloc[valid].to_numpy()
            rows.append({'alpha':alpha, 'fold':fold, 'n_validation':len(valid),
                         'pinball_recommended':float(mean_pinball_loss(truth,pred.p35,alpha=.35)),
                         'mae_recommended':float(mean_absolute_error(truth,pred.p35)),
                         'mae_median':float(mean_absolute_error(truth,pred.p50)),
                         'coverage':float(np.mean((truth>=pred.p10)&(truth<=pred.p90))),
                         'mean_interval_width':float((pred.p90-pred.p10).mean())})
        predictions[float(alpha)] = oof
    results = pd.DataFrame(rows)
    scores = results.groupby('alpha').pinball_recommended.mean().sort_values(kind='stable')
    best_alpha = float(scores.index[0])
    return best_alpha, results, predictions[best_alpha]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-optional',action='store_true')
    args=parser.parse_args()
    X,y,metadata=load_development()
    best,results,oof=search_ridge(X,y,metadata,include_optional=args.include_optional)
    model=RidgeQuantileModel(alpha=best,include_optional=args.include_optional)
    model.fit(X,y,groups=metadata.host_id)
    suffix='_optional' if args.include_optional else ''
    folds=make_folds(metadata)
    assignments=metadata.copy()
    for fold, (_, valid) in enumerate(folds): assignments.loc[valid,'fold']=fold
    assignments['fold']=assignments['fold'].astype(int)
    identity=hashlib.sha256(assignments.to_csv(index=False).encode()).hexdigest()
    summary={'snapshot':'2026-06-15', 'n_rows':len(X), 'n_hosts':int(metadata.host_id.nunique()),
             'test_evaluated':False, 'status':'ridge_candidate_not_global_winner',
             'metrics_status':METRICS_STATUS, 'metrics_implementation':METRICS_IMPLEMENTATION,
             'include_optional':args.include_optional, 'selected_alpha':best,
             'selection_metric':'mean_fold_pinball_p35', 'quantiles':[.1,.35,.5,.9],
             'outer_folds':5, 'inner_calibration_folds':3,
             'sklearn_version':sklearn.__version__,
             'june_sha256':hashlib.sha256(DEFAULT_JUNE_PATH.read_bytes()).hexdigest(),
             'fold_assignment_sha256':identity, 'fold_metrics':results.to_dict(orient='records'),
             'limitation':'Validación usada para selección; no es evaluación independiente. Cuantiles de residuos globales, cobertura no garantizada.'}
    (REPO_ROOT/'models').mkdir(exist_ok=True)
    joblib.dump({'model':model,'metadata':summary},REPO_ROOT/f'models/ridge{suffix}.joblib')
    (REPO_ROOT/f'docs/ridge_validation{suffix}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    out=REPO_ROOT/'data/processed'; out.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(out/'development_folds.csv',index=False)
    pd.concat([metadata,y.rename('price_mxn'),oof],axis=1).to_csv(out/f'ridge_oof{suffix}.csv',index=False)
    print(results.groupby('alpha').pinball_recommended.agg(['mean','std']).to_string())
    print('Métricas provisionales: pendiente integrar src.metrics y regenerar la comparación común.')
    print(f'Ridge candidato: alpha={best}; septiembre no evaluado.')


if __name__=='__main__': main()
