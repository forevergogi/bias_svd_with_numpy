#!/usr/bin/python
#coding=utf-8
'''
    Module:Data Pre-processing
    Author: Timmy Qiao
    Date: Jun 6 2019
    The main procedures include:
    1.Encode the user nicknames, product names and the order status into integer numbers
    2.Drop out some lines and replace the user names, product names and the order status with new id numbers
'''

import pandas as pd
import csv

def read_file(file_path,use_header=True):
    '''
    Read the csv file
    :param file_path:
    :param use_header:
    :return: the DataFrame object
    '''
    if not use_header:
        df = pd.read_csv(file_path,header=None,sep=',')
    else:
        df = pd.read_csv(file_path, sep=',')
    nomissing_df = df.dropna().reset_index(drop=True)
    return nomissing_df

def data_encode(dataFrame,write_csv=False):
    '''
    Encode the user names, product names and the order status
    :param dataFrame:
    :return: the dictionary arrays
    '''
    members_dic = {}
    items_dic = {}
    status_dic = {}
    for col in dataFrame.columns:
        col_name = str(col)
        cvals_keys = dataFrame[str(col)].value_counts().keys()
        if col_name == 'order_create_time' or \
            col_name == 'title' or col_name == 'cluster':
            continue
        if col_name == 'member_nick_name':
            id = 0
            for key in cvals_keys:
                members_dic[key] = id
                id += 1
        elif col_name == 'clean_title_name':
            id = 0
            for key in cvals_keys:
                items_dic[key] = id
                id += 1
        elif col_name == 'order_status':
            id = 0
            for key in cvals_keys:
                status_dic[key] = id
                id += 1
    members_dic_sorted = sorted(members_dic.items(),key=lambda members_dic:members_dic[1]) # return a list of tuples
    items_dic_sorted = sorted(items_dic.items(), key=lambda items_dic: items_dic[1])
    status_dic_sorted = sorted(status_dic.items(), key=lambda status_dic: status_dic[1])
    if write_csv:
        with open('data/user_ids.csv', 'w') as user_csv:
            writer = csv.writer(user_csv)
            writer.writerow(['user_id','user_name'])
            for item in members_dic_sorted:
                writer.writerow([item[1],item[0]])
        with open('data/item_ids.csv', 'w') as item_csv:
            writer = csv.writer(item_csv)
            writer.writerow(['item_id','item_name'])
            for item in items_dic_sorted:
                writer.writerow([item[1],item[0]])
        with open('data/status_ids.csv', 'w') as status_csv:
            writer = csv.writer(status_csv)
            writer.writerow(['status_id','status_name'])
            for item in status_dic_sorted:
                writer.writerow([item[1],item[0]])
    return [members_dic,items_dic,status_dic]

def create_new_dataset(file_path,oldDF,dics):
    '''
    Create new csv file for usage of the next time, drop out some unused lines.
    :param file_path:
    :param oldDF:
    :param dics:
    :return: None
    '''
    members_dic = dics[0]
    items_dic = dics[1]
    status_dic = dics[2]
    with open(file_path,'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['user_id','item_id','status_id','ctime'])
        nick_name_arr = oldDF['member_nick_name']
        title_name_arr = oldDF['clean_title_name']
        order_status_arr = oldDF['order_status']
        ctime_arr = oldDF['order_create_time']
        for i in range(oldDF.shape[0]):
            try:
                user_id = members_dic[nick_name_arr[i]]
                item_id = items_dic[title_name_arr[i]]
                status_id = status_dic[order_status_arr[i]]
                ctime = ctime_arr[i]
                writer.writerow([user_id, item_id, status_id, ctime])
                if (i + 1) % 10000 == 0:
                    print('%d of the data have been processed!' % (i + 1))
            except(Exception):
                print(str(i) + nick_name_arr[i]+' '+title_name_arr[i])
                break


if __name__ == '__main__':
    df = read_file('data/all_order.csv')
    print('------ Original file is read! -------')
    dics = data_encode(df,True)
    print('------ Encoding is done! -------')
    create_new_dataset('data/new_all_order.csv',df,dics)
    print('------ New CSV File is created ! -------')



