"""Inferencia sobre un artefacto ya entrenado; no abre ningún snapshot."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REQUIRED = ['latitude','longitude','accommodates','neighbourhood_cleansed','property_type','room_type']


def validate_listing(listing):
    if not isinstance(listing, dict):
        raise ValueError('La entrada debe ser un objeto JSON con características.')
    for key in REQUIRED:
        if key not in listing or listing[key] is None or str(listing[key]).strip()=='':
            raise ValueError(f'Falta el campo obligatorio: {key}')
    limits={'latitude':(-90,90),'longitude':(-180,180),'accommodates':(1,np.inf),
            'bathrooms':(0,np.inf),'bedrooms':(0,np.inf),'beds':(0,np.inf),
            'amenities_count':(0,np.inf),'minimum_nights':(1,np.inf),'maximum_nights':(1,np.inf)}
    for key,(low,high) in limits.items():
        if key not in listing or listing[key] is None:
            continue
        try: value=float(listing[key])
        except (TypeError,ValueError): raise ValueError(f'{key} debe ser numérico.') from None
        if not np.isfinite(value) or not low<=value<=high:
            raise ValueError(f'{key} fuera del rango permitido.')
    for key in ['neighbourhood_cleansed','property_type','room_type']:
        if not isinstance(listing[key],str): raise ValueError(f'{key} debe ser texto.')


def recommend(model, listing):
    validate_listing(listing)
    predictions=model.predict(pd.DataFrame([listing]))
    values=predictions.loc[:,['p10','p35','p50','p90']].iloc[0].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values<=0).any() or (np.diff(values)<0).any():
        raise ValueError('El modelo no produjo cuantiles positivos y ordenados.')
    return {'currency':'MXN','recommended_price':round(float(values[1]),2),
            'lower_price':round(float(values[0]),2),'upper_price':round(float(values[3]),2),
            'median_price':round(float(values[2]),2), 'nominal_coverage':.8,
            'note':'Referencia de precios publicados en CDMX; no garantiza reservas, ingresos ni cobertura individual.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True,help='Artefacto joblib propio y confiable.')
    parser.add_argument('--listing',type=Path,required=True)
    args=parser.parse_args()
    # joblib solo debe cargar archivos propios: la deserialización ejecuta código.
    artifact=joblib.load(args.model)
    model=artifact['model'] if isinstance(artifact,dict) else artifact
    result=recommend(model,json.loads(args.listing.read_text()))
    if isinstance(artifact,dict): result['model_status']=artifact.get('metadata',{}).get('status','unspecified')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
