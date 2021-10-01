import pandas as pd
# # Model Development

import lightgbm
import sqlalchemy
import snowflake.sqlalchemy
import datetime
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import cross_val_score, GridSearchCV

from framer.framer import Framer, CrossValidation
from dotenv import load_dotenv

snow_engine = sqlalchemy.create_engine(snowflake.sqlalchemy.URL(
    account = 'warnerbros_prod',
    user = os.environ.get("snowflake_service_user"),
    password = os.environ.get("snowflake_service_password"),
    database = 'SPLC',
    schema = 'Sandbox',
    warehouse = 'WH_CUST_ANALYTICS',
))


def fetch_input_df_agg():
    query = '''
    select * 
    from sandbox.nr_model_data
    where catman_trans='Y'
    and lifecycle='Y'
    and imdb_title_cd not in ('tt12801356')
    '''


    with snow_engine.connect() as con:
        import_df = pd.read_sql(query, con=con, parse_dates=['est_st_date', 'title_launch_date'])

    import_df.columns = [col.lower() for col in import_df.columns]


    input_df = import_df.copy()

    input_df = process_input_df(input_df)
    
    with snow_engine.connect() as con:
        ua_df = pd.read_sql("select * from lct_data_ua", con=con)

    ua_df = ua_df.drop_duplicates(subset=["IMDB ID", "TH Window"])

    ua_df = ua_df.loc[ua_df["TH Window"].notnull()]

    ua_df = ua_df.pivot(index="IMDB ID", columns="TH Window", values="Total")

    # don't need all UA, just like first 8 weeks or so
    ua_df = ua_df.drop(columns = [col for col in ua_df.columns if (int(col) > 8 and int(col) >= 0)])

    ua_df.columns = ["unaided_wk"+str(int(weeknum)) for weeknum in ua_df.columns]

    input_df = input_df.merge(ua_df, right_on="IMDB ID",  left_on="imdb_title_cd", how="left")

    return input_df
            




def fetch_input_df():
    query = '''
    select * 
    from sandbox.synthesized_results_weekly 
    where catman_trans='Y'
    and lifecycle='Y'
    and imdb_title_cd not in ('tt12801356')
    and (syn_est_units > 10 or syn_vod_units > 10)
    '''


    with snow_engine.connect() as con:
        import_df = pd.read_sql(query, con=con, parse_dates=['est_st_date', 'title_launch_date'])

    import_df.columns = [col.lower() for col in import_df.columns]


    input_df = import_df.copy()

    input_df = process_input_df(input_df)
    
    with snow_engine.connect() as con:
        ua_df = pd.read_sql("select * from lct_data_ua", con=con)

    ua_df = ua_df.drop_duplicates(subset=["IMDB ID", "TH Window"])

    ua_df = ua_df.loc[ua_df["TH Window"].notnull()]

    ua_df = ua_df.pivot(index="IMDB ID", columns="TH Window", values="Total")

    # don't need all UA, just like first 8 weeks or so
    ua_df = ua_df.drop(columns = [col for col in ua_df.columns if (int(col) > 8 and int(col) >= 0)])

    ua_df.columns = ["unaided_wk"+str(int(weeknum)) for weeknum in ua_df.columns]

    input_df = input_df.merge(ua_df, right_on="IMDB ID",  left_on="imdb_title_cd", how="left")

    return input_df
            
def process_input_df(input_df_og):
    input_df = input_df_og.copy()

    input_df["syn_est_revenue"] = input_df.est_revenue + input_df.pest_revenue
    input_df["syn_vod_revenue"] = input_df.vod_revenue + input_df.pvod_revenue
    
    input_df['est_release_year'] = input_df['est_st_date'].dt.year
    input_df['est_release_month'] = input_df['est_st_date'].dt.month
    input_df['syn_est_arp'] = input_df['syn_est_revenue']/input_df['syn_est_units']
    input_df['syn_vod_arp'] = input_df['syn_vod_revenue']/input_df['syn_vod_units']

    input_df['syn_prem_arp'] = input_df[['syn_est_arp', 'syn_vod_arp']].max(axis=1)
    input_df['syn_prem_window'] = input_df[['window_min_pvod','window_min_est','window_min_vod','window_min_pst']].min(axis=1)
    input_df['syn_prem_window'] = np.where(input_df['imdb_title_cd'] == 'tt5034838', 6, input_df['syn_prem_window'])

    input_df['syn_prem_units'] = np.where(
        input_df.title_launch_date >= pd.Timestamp(2020,4,1),
        input_df[['syn_est_units', 'syn_vod_units']].sum(axis=1),
        input_df['syn_est_units'])

    input_df['syn_prem_revenue'] = np.where(
        input_df.title_launch_date >= pd.Timestamp(2020,4,1),
        input_df[['syn_est_revenue', 'syn_vod_revenue']].sum(axis=1),
        input_df['syn_est_revenue'])



    input_df['pvod_title'] = np.where(input_df['pvod_st_date'].isna(), 'Standard', np.where((input_df['pvod_st_date'].notna()) & (input_df['est_st_date'] == input_df['pvod_st_date']),'PVOD-PEST','PVOD-EST'))

    input_df['title_launch_year'] = input_df['title_launch_date'].dt.year
    input_df['rating'] = input_df['rating'].fillna('Not Rated')
    input_df['wk1_8_box_revenue'] = input_df['w1_box_revenue'].fillna(0) + input_df['w2_8_box_revenue'].fillna(0)
    input_df['theatrical'] = np.where(input_df.wk1_8_box_revenue <= 1000000, 'N', 'Y')

    data_refresh_date = pd.Timestamp(datetime.datetime.today())
    input_df['est_8wk_complete'] = np.where(input_df['est_st_date'] < data_refresh_date  - pd.to_timedelta(70, unit='d'), 'Y', 'N')

    input_df['box_w2_8_share'] = input_df['w2_8_box_units'] / (input_df['w1_box_units'] + input_df['w2_8_box_units'])

    input_df['syn_wor_vod'] = input_df[['wor_vod', 'wor_pvod']].max(axis=1)
    input_df['syn_wor_est'] = input_df[['wor_est', 'wor_pest']].max(axis=1)

    input_df['syn_wor_est'] = np.where(input_df.imdb_title_cd == 'tt5034838', input_df['wor_est']+4, input_df['syn_wor_est'])
    input_df['syn_wor_prem'] = input_df[['syn_wor_vod', 'syn_wor_est']].max(axis=1)

    input_df['streaming_ind'] = np.where(input_df.imdb_title_cd.isin(['tt10016180','tt1361336','tt0293429','tt7126948','tt9784798','tt5109280','tt5034838']),1,0)


    input_df = input_df.rename({'total': 'unaided'}, axis=1)

    # remove bad titles: tt0837563 (Pet Senetary)
    # remove recent title: tt3215824 (Those Who Wish Me Dead)
    input_df = input_df[~input_df.imdb_title_cd.isin(['tt0837563', 'tt3215824'])]

    return input_df




def build_framer(target, ua_week=0, format=None):

    if format == "premium":
        feature_name= [
            'genre', 
            'mpaarating', 
            #'est_release_year',

            #'syn_prem_window', 
            #'criticsummary_score_value',
            #'fansummary_score_value',
            #'syn_prem_arp',
            'country_cd',
            'new_case',
            # 'unaided',
            'unaided_wk' + str(ua_week),
            #'streaming_ind',
            'syn_wor_prem',
            

            #'wk1_8_box_revenue',
            #'est_w1_8_arp', 'vod_w1_8_arp',
            #'action & adventure', 'animation', 'comedy',  
            #'drama', 'horror', 'kids & family',
            #'romance', 'science fiction & fantasy',  
        ]
    else:
        feature_name= [
            'genre', 
            'mpaarating', 
            #'est_release_year',

            'window_min_est', 
            #'criticsummary_score_value',
            #'fansummary_score_value',
            'syn_wor_est',
            'syn_wor_vod',
            'syn_est_arp',
            'syn_vod_arp',
            'country_cd',
            'new_case',
            # 'unaided',
            'unaided_wk' + str(ua_week),
            'streaming_ind',
            
            

            #'wk1_8_box_revenue',
            #'est_w1_8_arp', 'vod_w1_8_arp',
        #     'action & adventure', 'animation', 'comedy',  
        #     'drama', 'horror', 'kids & family',
        #      'romance', 'science fiction & fantasy',  
            ]


    index = ['imdb_title_cd', 'country_cd', 'wor']

    mappings = {
        'mpaarating': ['G', 'Not Rated', 'PG', 'R', 'PG-13', 'NC-17'],
        #'current_major_studio': ['ALL OTHER', 'DISNEY', 'LIONSGATE', 'NBC UNIVERSAL', 'SONY', 'VIACOM CBS', 'WARNER'],
        #'pvod_title': ['PVOD-EST', 'PVOD-PEST','Standard'],
        #'genre_aggr':['Horror & Mystery Suspense', 'Action/Adventure', 'Drama', 'Family','Comedy', 'Science Fiction', 'Others']
        'genre':['Drama', 'Animation', 'Musical', 'Action/Adventure', 'Action', 'Science Fiction', 'Horror', 'Mystery Suspense', 'Comedy', 'Family', 'Adventure', 'Not Available', 'Documentary', 'Childrens Nt'],
        'country_cd': ['US', 'CA']
    }

    transform_in = np.log
    transform_out = np.exp

    framer = Framer(
        feature_name=feature_name,
        target=target,
        index=index,
        mappings=mappings,
        transform_in=transform_in,
        transform_out=transform_out)
    
    return framer


def build_model(model_df, framer, features, target, cv_splitting_feature):

    
    # ## Model
    # 
    # * Use linear regression as a baseline?
    # * Try gradient boosting and random forest? I recommend lightGBM
    # * Other approaches?

    # ### Gradient Boosting



    model = lightgbm.LGBMRegressor(min_child_samples = 5)
    model.fit(X=np.nan_to_num(features), y=target, feature_name=framer.feature_name, categorical_feature=framer.categorical_feature)
    model_pred = model.predict(features)
    pred_df = framer.transform_out(model_pred)

    act_v_pred_df = framer.get_actual_v_pred(input_df=model_df, pred_vector=model_pred)


    # ### Model Fit Statitics:
    # 
    # * R-2
    # * MAPE weighted by target value
    # * MAPE weighted by box


    # framer.report_actual_v_pred(input_df=model_df, pred_vector=model_pred, feature_weighting='wk1_8_box_revenue')



    # framer.report_actual_v_pred(input_df=model_df, pred_vector=model_pred, feature_weighting='syn_est_units')


    # ### Feature Importance

    # #### Gradient Boosting


    splits = model.booster_.feature_importance(importance_type='split')
    gain = model.booster_.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame(splits, index=framer.feature_name, columns=['split'])
    feature_importance_df['split_pct'] = feature_importance_df['split'] / feature_importance_df['split'].sum()
    feature_importance_df['gain'] = gain
    feature_importance_df['gain_pct'] = feature_importance_df['gain'] / feature_importance_df['gain'].sum()


    # ### Feature Contributions


    contrib_values = model.predict(features, pred_contrib=True)
    contrib_col_names = [feature + '_contrib' for feature in framer.feature_name + ['init_value']]
    contrib_df = pd.DataFrame(contrib_values, columns=contrib_col_names)

    full_results = model_df.copy()
    full_results[contrib_df.columns] = contrib_df
    full_results[act_v_pred_df.columns] = act_v_pred_df.values

    assert full_results.set_index(['imdb_title_cd', 'country_cd', 'wor']).index.is_unique, 'full results not uniquely indexed'


    # full_results.to_sql('model_full_results', con=snow_engine, index=False, chunksize=5000, if_exists = 'replace')


    # ## Cross Validation
    # 
    # * https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    # * https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit
    # * Or hard-code/manual approach:
    #     * Use 2016-2017 to predict 2018
    #     * 2016-2018  to predict 2019
    #     * etc

    # #### Gradient Boosting

    cv_scheme = {
        f'2016-{year}': list(range(2016, year+1)) for year in range(2017, 2021+1)
    }

    # cv_splitting_feature = 'est_st_date'

    cv = CrossValidation(
        framer=framer,
        model=model,
        cv_splitting_feature=cv_splitting_feature,
        cv_scheme=cv_scheme)


    cv.fit(input_df=model_df,cv_method = 'kfold split', folds = 5)
    kfold_pred_result, kfold_feature_contri = cv.get_actual_v_pred(input_df=model_df, cv_method = 'kfold split', folds = 5)
    kfold_metric_result = cv.report_actual_v_pred(input_df=model_df, feature_weighting='syn_est_units', cv_method = 'kfold split', folds = 5)

    # ## Tune Parameter

    # #### Gradient Boosting


    search_grid={'n_estimators':[i for i in range(500,3000,500)],'learning_rate':[.001,0.01,.1],'num_leaves' : [10,20,30,40]}
    search=GridSearchCV(estimator=model,param_grid=search_grid,scoring='neg_mean_absolute_percentage_error')



    X = framer.get_features(model_df)
    y = framer.get_target(model_df)
    grid_search_results = search.fit(X,y)


    best_para = grid_search_results.best_params_


    # ## Refit Model with Best Parameter

    # #### Gradient Boosting


    model = lightgbm.LGBMRegressor(min_child_samples=5, learning_rate = best_para['learning_rate'], n_estimators = best_para['n_estimators'], num_leaves = best_para['num_leaves'])
    model.fit(X=features, y=target, feature_name=framer.feature_name, categorical_feature=framer.categorical_feature)
    model_pred = model.predict(features)
    pred_df = framer.transform_out(model_pred)


    splits = model.booster_.feature_importance(importance_type='split')
    gain = model.booster_.feature_importance(importance_type='gain')
    feature_importance_df = pd.DataFrame(splits, index=framer.feature_name, columns=['split'])
    feature_importance_df['split_pct'] = feature_importance_df['split'] / feature_importance_df['split'].sum()
    feature_importance_df['gain'] = gain
    feature_importance_df['gain_pct'] = feature_importance_df['gain'] / feature_importance_df['gain'].sum()



    cv = CrossValidation(
        framer=framer,
        model=model,
        cv_splitting_feature=cv_splitting_feature,
        cv_scheme=cv_scheme)


    cv.fit(input_df=model_df,cv_method = 'kfold split', folds = 5)
    kfold_pred_result, kfold_feature_contri = cv.get_actual_v_pred(input_df=model_df, cv_method = 'kfold split', folds = 5)
    cv.report_actual_v_pred(input_df=model_df, feature_weighting='wk1_8_box_revenue', cv_method = 'kfold split', folds = 5)

    return model