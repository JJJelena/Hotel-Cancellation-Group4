"""Rebuild A-v4's unfitted preprocessing contract without loading a pickle.

The column order, imputation, scaling, one-hot settings and minimum category
frequency match the supplied preprocessor_template.joblib. Every model/fold
must fit a fresh clone on its own fitting rows.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    'lead_time', 'arrival_date_year', 'arrival_date_week_number',
    'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
    'adults', 'children', 'babies', 'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'total_guests', 'total_nights', 'family_booking',
]
CATEGORICAL_FEATURES = [
    'hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
    'distribution_channel', 'reserved_room_type', 'agent', 'lead_time_group',
]


def build_preprocessor_template():
    numeric = Pipeline([('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', min_frequency=50))])
    return ColumnTransformer([('num', numeric, list(NUMERIC_FEATURES)),
        ('cat', categorical, list(CATEGORICAL_FEATURES))], remainder='drop')


def load_shared_data(project_root):
    """Load the fixed A-v4 CSVs with a source-built, unfitted preprocessor."""
    root = Path(project_root)
    df = pd.read_csv(root / 'data/hotel_bookings_clean.csv')
    train_ids = pd.read_csv(root / 'data/splits/train_ids.csv').row_id
    test_ids = pd.read_csv(root / 'data/splits/test_ids.csv').row_id
    chrono = pd.read_csv(root / 'data/splits/chronological_split.csv')
    folds = pd.read_csv(root / 'data/splits/cv_folds.csv')
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert len(df) == 118564 and df.row_id.is_unique
    assert df.is_canceled.isin([0, 1]).all()
    assert not set(features) & {'row_id', 'is_canceled', 'required_car_parking_spaces'}
    def subset(ids):
        return df[df.row_id.isin(ids)].reset_index(drop=True)
    data = dict(project_root=root, df=df, features=features,
        preprocessor=build_preprocessor_template(),
        random_train=subset(train_ids), random_test=subset(test_ids),
        chrono_train=subset(chrono.loc[chrono.split.eq('train'), 'row_id']),
        chrono_test=subset(chrono.loc[chrono.split.eq('test'), 'row_id']),
        cv_folds=folds)
    for a, b in [('random_train', 'random_test'), ('chrono_train', 'chrono_test')]:
        first, second = set(data[a].row_id), set(data[b].row_id)
        assert first.isdisjoint(second) and first | second == set(df.row_id)
    assert (len(data['random_train']), len(data['random_test'])) == (94851, 23713)
    assert (len(data['chrono_train']), len(data['chrono_test'])) == (96428, 22136)
    train = data['random_train']
    calculated = np.zeros(len(train), dtype=int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for number, (_, validation) in enumerate(cv.split(train, train.is_canceled), 1):
        calculated[validation] = number
    assert folds.row_id.is_unique and set(folds.row_id) == set(train.row_id)
    supplied = folds.set_index('row_id').loc[train.row_id, 'cv_fold'].to_numpy()
    assert np.array_equal(calculated, supplied)
    return data
