import re
import numpy as np
import pandas as pd

def is_sequential(left, right, userid, timeid):
    r_time = right.groupby(userid)[timeid].min()
    l_time = left.query(f'{userid} in @r_time.index').groupby(userid)[timeid].max()
    # assume r_time contains unseen users
    maybe_seq = l_time.combine(r_time, lambda x, y: x <= y)
    result = maybe_seq.all()
    if not result: # maybe it's due to unseen users
        maybe_unseen = maybe_seq[~maybe_seq].index # potensially unseen users idx
        if maybe_unseen.isin(l_time).any(): # they are not unseen => contradiction
            return False
        # check that unseen users all go later
        result = r_time.loc[maybe_unseen].min() >= l_time.max() 
    # handle remaining users not present in the next step
    rest = left.query(f'{userid} not in @r_time.index')
    if len(rest):
        result = result and (rest[timeid].max() <= r_time.min())
    return result

def reindex_data(data, columns, base=1):
    '''Reindex starting from `base`. SASRec relies on indexing that starts from 1.'''
    categories = {col: data[col].astype('category').cat for col in columns}
    new_index = {col: pd.Index(np.r_[range(-base, 0), cat.categories]) for col, cat in categories.items()}
    new_data = data.assign(**{col: cat.codes+base for col, cat in categories.items()})
    return new_data, new_index

def verify_reindex(data, userid, itemid, data_index):
    new_data = (
        data
        .assign(**{col: idx.get_indexer_for(data[col]) for col, idx in data_index.items()})
    )    
    warm_users = (new_data[userid] == -1) & (new_data[itemid] != -1)
    if warm_users.any(): # allow unseen users with known items, update corresponding user index
        # do not explicitly assume that users are unique in the dataset
        new_user_cat = data.loc[warm_users, userid].astype('category').cat
        # assign new user index
        base = len(data_index[userid])
        new_data.loc[warm_users, userid] = base + new_user_cat.codes
        # update user index data
        data_index[userid] = data_index[userid].append(pd.Index(new_user_cat.categories))
    return new_data.query(f'{itemid} != -1') # skip unseen items

def read_offsets(offset_str):
    offsets = re.findall(r'(\d+)(\w+)', offset_str)
    return [pd.DateOffset(**{interval: int(n_intervals)}) for n_intervals, interval in offsets]

def split_offsets(data, offset_str, timeid, time_unit='s'):
    '''Data reading and proper indexing. Adapted for evaluation of SASRec that relies on 1-based indexing'''
    time_offsets = read_offsets(offset_str)
    timestamps = pd.to_datetime(data[timeid], unit=time_unit)
    test_time_threshold = timestamps.max() - time_offsets.pop()

    valid_time_threshold = test_time_threshold - time_offsets.pop()
    train_split = timestamps <= valid_time_threshold
    train_valid = data.loc[train_split]
    valid = data.loc[(~train_split) & (timestamps <= test_time_threshold)]

    test_split = timestamps > test_time_threshold
    test = data.loc[test_split]
    return train_valid, valid, test

def generate_partition(observed, holdout, userid, itemid, timeid, stepwise_eval=False):
    if observed is None:
        return
    idx_start = 1 # to conform with SASRec indexing
    train, data_index = reindex_data(observed, [userid, itemid], base=idx_start)
    
    usernum = len(data_index[userid]) - idx_start
    itemnum = len(data_index[itemid]) - idx_start

    user_train = (
        train
        .sort_values(timeid)
        .groupby(userid, sort=False)[itemid]
        .apply(list)
        .to_dict()
    )

    holdout = verify_reindex(holdout, userid, itemid, data_index)
    user_test = generate_sequences(holdout, userid, itemid, timeid, stepwise=stepwise_eval)
    return user_train, user_test, usernum, itemnum

def generate_sequences(data, userid, itemid, timeid, stepwise, max_steps=10, min_step_users=100):
    '''Each step is a sequence of users with their next item.'''
    data_sorted = data.sort_values(timeid)
    if stepwise:
        return (
            data_sorted
            .assign(step = lambda df:
                df
                .groupby([userid], sort=False)[timeid]
                .transform('cumcount')
            )
            .groupby('step')[[userid, itemid]]
            .apply(lambda x: list(x.itertuples(index=False, name=None))) # list (user,item) pairs
            .loc[lambda x: x.apply(len) >= min_step_users]
            .iloc[:max_steps] # `None` will not filter anything
            .sort_index()
        )
    return list(data_sorted[[userid, itemid]].itertuples(index=False, name=None))
