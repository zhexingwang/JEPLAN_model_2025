import os
import numpy as np
import pandas as pd

def load_files():
    # # 熱媒温度TIRCA-BD-101.PVを取得
    # df = pd.read_csv(os.getcwd() + '/fitting_dataset/result_df_202312041427.csv')
    # columns = ['Timestamp', 'JEP.EU_PIRCA-BD-101B.PV', 'JEP.EU_TIRCA-BD-101.PV', 'JEP.EU_WZIRA-EE3-201.PV', 'yield']
    # 残渣温度TIRA-BD-301.PVを取得
    df = pd.read_csv(os.getcwd() + '/fitting_dataset/v3_result_df_202401121430.csv')
    columns = ['Timestamp', 'JEP.EU_PIRCA-BD-101B.PV', 'JEP.EU_TIRA-BD-301.PV', 'JEP.EU_WZIRA-EE3-201.PV', 'yield']

    feed_com = pd.read_csv(os.getcwd() + '/fitting_dataset/feed_com.csv')

    #データセットの用意
    #1~3バッチは使わない
    #4~9バッチをフィッティング用、10~11バッチを検証用
    # fitting_df = df[['Timestamp', 'yield', 'JEP.EU_PIRCA-BD-101B.PV', 'JEP.EU_TIRCA-BD-101.PV', 'JEP.EU_WZIRA-EE3-201.PV', 'Batch Number']]
    fitting_df = df[['Timestamp', 'yield', 'JEP.EU_PIRCA-BD-101B.PV', 'JEP.EU_TIRA-BD-301.PV', 'JEP.EU_WZIRA-EE3-201.PV', 'Batch Number']]
    # fitting_df['feed_com'] = 0
    fitting_df.loc[:, 'feed_com'] = 0
    # feed_com[feed_com['Batch Number']==4]
    for i in range(len(feed_com)):
        fitting_df.loc[fitting_df['Batch Number']==feed_com.loc[i, 'Batch Number'], 'feed_com'] = feed_com.loc[i, 'BHET']
    fitting_data = fitting_df[(4 <= fitting_df['Batch Number']) & (fitting_df['Batch Number'] <= 9)]
    # fitting_data = fitting_df[(4 <= fitting_df['Batch Number']) & (fitting_df['Batch Number'] <= 5)]
    # fitting_data = fitting_df[(fitting_df['Batch Number'] == 6)]
    return df, fitting_df, fitting_data