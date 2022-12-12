import os
import json
import gzip
import urllib
from ast import literal_eval
import pandas as pd
from polara import get_movielens_data

DATA_DIR = 'data'

def check_dir():
    if not os.path.isdir(DATA_DIR):
        os.mkdir(DATA_DIR)    

def ml_downloader():
    data = get_movielens_data(include_time=True)
    dest = os.path.join(DATA_DIR, 'ml-1m.gz')
    check_dir()
    (
        data
        .loc[:, ['userid', 'movieid', 'timestamp']]
        .to_csv(dest, index=False)
    )
    print(f'ML-1M data saved to {dest}.')


def parse_lines_amz(path, fields):
    with gzip.open(path, 'rt') as gz:
        for line in gz:
            yield json.loads(line, object_hook=lambda dct: tuple(dct[key] for key in fields))

def amz_downloader(dataset):
    dsname = {
        'amz-b': 'Beauty',
        'amz-g': 'Toys_and_Games',
    }
    name = dsname[dataset]
    url = f'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{name}_5.json.gz'
    tmp_file, _ = urllib.request.urlretrieve(url) # this may take some time depending on your internet connection
    print(f'Saved temporary file to {tmp_file}. Processing...')
    fields = ['reviewerID', 'asin', 'unixReviewTime']
    data = pd.DataFrame.from_records(parse_lines_amz(tmp_file, fields), columns=fields)
    dest = os.path.join(DATA_DIR, f'{dataset}.gz')
    check_dir()
    (
        data
        .rename(columns={'reviewerID': 'userid', 'unixReviewTime': 'timestamp'})
        .to_csv(dest, index=False)
    )
    print(f'{dataset.upper()} data saved to {dest}.')


def parse_lines_steam(path, fields):
    with gzip.open(path, 'rt') as gz:
        for line in gz:
            dct = literal_eval(line.strip())
            yield {key: dct[key] for key in fields}

def pcore_filter(data, pcore, userid, itemid):
    while pcore: # do only if pcore is specified
        item_check = True
        valid_items = data[itemid].value_counts() >= pcore
        if not valid_items.all():
            data = data.query(
                f'{itemid} in @valid_items.index[@valid_items]'
            )
            item_check = False
            
        user_check = True
        valid_users = data[userid].value_counts() >= pcore
        if not valid_users.all():
            data = data.query(
                f'{userid} in @valid_users.index[@valid_users]'
            )
            user_check = False
        
        if user_check and item_check:
            break
    return data.copy()

def steam_downloader():
    url = f'http://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz'
    tmp_file, _ = urllib.request.urlretrieve(url) # this may take some time depending on your internet connection
    print(f'Saved temporary file to {tmp_file}. Processing...')
    fields = ['username', 'product_id', 'date']
    raw_data = pd.DataFrame.from_records(parse_lines_steam(tmp_file, fields), columns=fields)
    data_dedup = raw_data.drop_duplicates(subset=['username', 'product_id'], keep='last')
    data_clean = pcore_filter(data_dedup, 5, 'username', 'product_id')

    data_clean.loc[:, 'timestamp'] = (
        pd.to_datetime(data_clean['date']) - pd.Timestamp("1970-01-01")
    ) // pd.Timedelta('1s')
    dest = os.path.join(DATA_DIR, 'steam.gz')
    check_dir()
    (
        data_clean
        .loc[:, ['username', 'product_id', 'timestamp']]
        .rename(columns={'username': 'userid'})
        .to_csv(dest, index=False)
    )
    print(f'Steam data saved to {dest}.')


def download_all_data():
    ml_downloader()
    amz_downloader('amz-b')
    amz_downloader('amz-g')
    steam_downloader()


if __name__ == '__main__':
    download_all_data()