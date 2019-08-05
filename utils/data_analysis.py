#!/usr/bin/python
#coding=utf-8
#encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import pandas as pd

if __name__ == '__main__':
    user_name = '雾里看花113'
    df = pd.read_csv('data/all_order.csv',encoding='utf-8')
    nomissing_df = df.dropna().reset_index(drop=True)
    spec_df = nomissing_df[nomissing_df['member_nick_name']==user_name].copy()
    spec_df['order_create_time'] = pd.to_datetime(spec_df['order_create_time'])
    spec_df.sort_values('order_create_time',inplace=True)
    spec_df.to_csv('user_specific/'+user_name+'_order.csv',encoding='utf_8_sig',index=False)