from typing import Callable
import copy

import pandas as pd
import numpy as np
import sklearn

from random import randrange
from random import seed


class Framer:

    def __init__(self,
                 feature_name: list,
                 target: str,
                 index: list,
                 mappings: dict = None,
                 transform_in: Callable = None,
                 transform_out: Callable = None):
        self.feature_name = feature_name
        self.target = target
        self.mappings = mappings
        self.index = index
        self.categorical_feature = 'auto'
        if mappings:
            self.categorical_feature = list()
            for key in mappings.keys():
                assert key in self.feature_name, f'{key} is not in feature_names'
                self.categorical_feature.append(key)
        self.transform_in = transform_in
        self.transform_out = transform_out
        
    def set_index(self, new_index: list):
        self.index = new_index

    def get_features(self, input_df: pd.DataFrame):
        assert input_df.set_index(self.index).index.is_unique, f'input_df is not uniquely indexed by {self.index}'
        df = input_df[self.feature_name].copy()
        if self.mappings:
            for key in self.mappings.keys():
                value_list = self.mappings[key]
                mapping = {value: value_list.index(value) for value in value_list}
                if key in self.feature_name:
                    df[key] = df[key].map(mapping)
                assert not df[key].isna().any(), f'{key} encoding is not valid'
        return df.values

    def get_target(self, input_df: pd.DataFrame):
        if self.transform_in:
            target = self.transform_in(input_df[self.target].values)
        else:
            target = input_df[self.target].values
        return np.squeeze(target.reshape(-1, 1))

    def get_actual_v_pred(self, input_df: pd.DataFrame, pred_vector: np.ndarray):
        act_v_pred_df = input_df.set_index(self.index)[self.target].rename('actual').to_frame()
        act_v_pred_df['prediction'] = self.transform_out(pred_vector)
        act_v_pred_df['residual'] = act_v_pred_df['actual'] - act_v_pred_df['prediction']
        return act_v_pred_df

    def report_actual_v_pred(self, input_df: pd.DataFrame, pred_vector: np.ndarray, feature_weighting: str = None):
        act_v_pred_df = self.get_actual_v_pred(input_df=input_df, pred_vector=pred_vector)
        report_metrics = dict()
        report_metrics['r2 score'] = sklearn.metrics.r2_score(y_true=act_v_pred_df.actual,
                                                              y_pred=act_v_pred_df.prediction)
        report_metrics['mape'] = sklearn.metrics.mean_absolute_percentage_error(y_true=act_v_pred_df.actual,
                                                                                y_pred=act_v_pred_df.prediction)
        report_metrics['mape weighted by target'] = sklearn.metrics.mean_absolute_percentage_error(
            y_true=act_v_pred_df.actual, y_pred=act_v_pred_df.prediction, sample_weight=act_v_pred_df.actual)
        if feature_weighting:
            report_metrics[f'mape weighted by {feature_weighting}'] = sklearn.metrics.mean_absolute_percentage_error(
                y_true=act_v_pred_df.actual, y_pred=act_v_pred_df.prediction, sample_weight=input_df[feature_weighting])

        results = pd.DataFrame(report_metrics.values(), index=report_metrics.keys(), columns=['score'])

        return results


class CrossValidation:

    def __init__(self, model, framer: Framer, cv_splitting_feature: str, cv_scheme: dict):
        self.framer = framer
        self.model = model
        self.cv_scheme = cv_scheme
        self.cv_splitting_feature = cv_splitting_feature
        self.cv_models = None

    def cross_validation_split(self, dataset, folds=5):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / folds)
        for i in range(folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        flat_dataset_split = [item for sublist in dataset_split for item in sublist]
        if (flat_dataset_split in dataset_copy) is False:
            left_data = list(set(dataset_copy) - set(flat_dataset_split))
            for n in range(0, len(left_data)):
                dataset_split[n].append(left_data[n])
        return dataset_split

    def _data_prep(self, input_df: pd.DataFrame, cv_method='time series split', folds=5):
        stack_input_df = pd.DataFrame()
        if cv_method == 'time series split':
            for cv_round in self.cv_scheme:
                sub_df = input_df.copy()
                sub_df['cv_round'] = cv_round
                sub_df['cv_data'] = np.where(sub_df[self.cv_splitting_feature].dt.year.isin(self.cv_scheme[cv_round]),
                                             'training', 'holdout')
                stack_input_df = stack_input_df.append(sub_df.set_index(['cv_round', 'cv_data'], append=True))

        elif cv_method == 'kfold split':
            seed(1)
            folds_group = self.cross_validation_split(input_df['data_index'], folds)
            for i in range(0, len(folds_group)):
                sub_df = input_df.copy()
                sub_df['cv_round'] = i
                sub_df['cv_data'] = np.where(sub_df['data_index'].isin(folds_group[i]), 'holdout', 'training')
                stack_input_df = stack_input_df.append(sub_df.set_index(['cv_round', 'cv_data'], append=True))

        return stack_input_df

    def fit(self, input_df: pd.DataFrame, cv_method='time series split', folds=5, model_type ='lightgbm'):
        self.cv_models = dict()
        training_df = self._data_prep(input_df=input_df, cv_method=cv_method, folds=folds)
        if cv_method == 'time series split':
            for cv_round in self.cv_scheme:
                print(f'running {cv_round}...')
                sub_input_df = training_df.xs(cv_round, level='cv_round').xs('training', level='cv_data').reset_index(
                    drop=True)
                print(sub_input_df.shape[0])
                X = self.framer.get_features(sub_input_df)
                y = self.framer.get_target(sub_input_df)
                cv_model = copy.deepcopy(self.model)
                if model_type == 'lightgbm':
                    cv_model.fit(X=X, y=y, feature_name=self.framer.feature_name,
                             categorical_feature=self.framer.categorical_feature)
                elif model_type == 'rfr':
                    cv_model.fit(X=X, y=y)
                self.cv_models[cv_round] = cv_model

        elif cv_method == 'kfold split':
            for i in range(0, folds):
                print(f'running {i}...')
                sub_input_df = training_df.xs(i, level='cv_round').xs('training', level='cv_data').reset_index(
                    drop=True)
                print(sub_input_df.shape[0])
                X = self.framer.get_features(sub_input_df)
                y = self.framer.get_target(sub_input_df)
                cv_model = copy.deepcopy(self.model)
                if model_type == 'lightgbm':
                    cv_model.fit(X=X, y=y, feature_name=self.framer.feature_name,
                                 categorical_feature=self.framer.categorical_feature)
                elif model_type == 'rfr':
                    cv_model.fit(X=X, y=y)
                self.cv_models[i] = cv_model

    def get_actual_v_pred(self, input_df: pd.DataFrame, cv_method='time series split', folds=5, model_type = 'lightgbm'):
        stack_input_df = self._data_prep(input_df=input_df, folds=folds, cv_method=cv_method)
        stack_results = pd.DataFrame()
        stack_contri = pd.DataFrame()
        if cv_method == 'time series split':
            for cv_round in self.cv_models:
                model = self.cv_models[cv_round]
                sub_input_df = stack_input_df.xs(cv_round, level='cv_round').reset_index(drop=True)
                X = self.framer.get_features(input_df=sub_input_df)
                pred_vector = model.predict(X=X)
                act_v_pred_df = self.framer.get_actual_v_pred(input_df=sub_input_df, pred_vector=pred_vector)
                act_v_pred_df['cv_round'] = cv_round
                act_v_pred_df['cv_data'] = np.where(
                    input_df[self.cv_splitting_feature].dt.year.isin(self.cv_scheme[cv_round]), 'training', 'holdout')
                stack_results = stack_results.append(act_v_pred_df.set_index(['cv_round', 'cv_data'], append=True))

                if model_type == 'lightgbm':
                    contrib_values = model.predict(X=X, pred_contrib=True)
                    contrib_col_names = [feature + '_contrib' for feature in self.framer.feature_name + ['init_value']]
                    contrib_df = pd.DataFrame(contrib_values, columns=contrib_col_names)
                    full_result = sub_input_df.copy()
                    full_result[contrib_df.columns] = contrib_df
                    full_result[act_v_pred_df.columns] = act_v_pred_df.values
                    full_result['cv_round'] = cv_round
                    full_result['cv_data'] = np.where(
                        full_result[self.cv_splitting_feature].dt.year.isin(self.cv_scheme[cv_round]), 'training',
                        'holdout')
                    stack_contri = stack_contri.append(full_result)
                else:
                    pass

        elif cv_method == 'kfold split':
            seed(1)
            folds_group = self.cross_validation_split(input_df['data_index'], folds)
            for cv_round in self.cv_models:
                model = self.cv_models[cv_round]
                sub_input_df = stack_input_df.xs(cv_round, level='cv_round').reset_index(drop=True)
                X = self.framer.get_features(input_df=sub_input_df)
                pred_vector = model.predict(X=X)
                act_v_pred_df = self.framer.get_actual_v_pred(input_df=sub_input_df, pred_vector=pred_vector)
                act_v_pred_df['cv_round'] = cv_round
                act_v_pred_df['cv_data'] = np.where(input_df['data_index'].isin(folds_group[cv_round]), 'holdout',
                                                    'training')
                stack_results = stack_results.append(act_v_pred_df.set_index(['cv_round', 'cv_data'], append=True))

                if model_type == 'lightgbm':
                    contrib_values = model.predict(X=X, pred_contrib=True)
                    contrib_col_names = [feature + '_contrib' for feature in self.framer.feature_name + ['init_value']]
                    contrib_df = pd.DataFrame(contrib_values, columns=contrib_col_names)
                    full_result = sub_input_df.copy()
                    full_result[contrib_df.columns] = contrib_df
                    full_result[act_v_pred_df.columns] = act_v_pred_df.values
                    full_result['cv_round'] = cv_round
                    full_result['cv_data'] = np.where(full_result['data_index'].isin(folds_group[cv_round]), 'holdout',
                        'training')
                    stack_contri = stack_contri.append(full_result)
                else:
                    pass

        return stack_results, stack_contri

    def report_actual_v_pred(self, input_df: pd.DataFrame, feature_weighting: str = None, cv_method='time series split',
                             folds=5):
        stack_input_df = self._data_prep(input_df=input_df, cv_method=cv_method, folds=folds)
        stack_results = pd.DataFrame()

        for sub_group, sub_df in stack_input_df.groupby(['cv_round', 'cv_data']):
            cv_round = sub_group[0]
            cv_data = sub_group[1]
            model = self.cv_models[cv_round]

            print(f'cv_round: {cv_round} and cv_data: {cv_data}')
            print(sub_df.shape[0])
            X = self.framer.get_features(input_df=sub_df)
            pred_vector = model.predict(X=X)
            act_v_pred_df = self.framer.get_actual_v_pred(input_df=sub_df, pred_vector=pred_vector)

            report_metrics = dict()

            report_metrics['r2 score'] = sklearn.metrics.r2_score(y_true=act_v_pred_df.actual,
                                                                  y_pred=act_v_pred_df.prediction)
            report_metrics['mape'] = sklearn.metrics.mean_absolute_percentage_error(y_true=act_v_pred_df.actual,
                                                                                    y_pred=act_v_pred_df.prediction)
            report_metrics['mape weighted by target'] = sklearn.metrics.mean_absolute_percentage_error(
                y_true=act_v_pred_df.actual, y_pred=act_v_pred_df.prediction, sample_weight=act_v_pred_df.actual)
            if feature_weighting:
                report_metrics[
                    f'mape weighted by {feature_weighting}'] = sklearn.metrics.mean_absolute_percentage_error(
                    y_true=act_v_pred_df.actual, y_pred=act_v_pred_df.prediction,
                    sample_weight=sub_df[feature_weighting])

            results = pd.DataFrame(report_metrics.values(), index=report_metrics.keys(), columns=['score'])
            results['cv_round'] = cv_round
            results['cv_data'] = cv_data
            stack_results = stack_results.append(results.set_index(['cv_round', 'cv_data'], append=True))
        return stack_results.unstack(level='cv_round')
