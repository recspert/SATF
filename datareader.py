from utils import data_partition
import pandas as pd
from polara import RecommenderData
from preprocessing import split_offsets, generate_partition

def generate_sequential(train_data, userid, itemid, maxlen, seqid='position'):
    return (
        pd.concat(
            {
                user: pd.DataFrame(
                    data = {itemid: items},
                    index = range(maxlen-len(items), maxlen)
                )
                for user, items in train_data.items()
            },
            names=[userid, seqid]
        )
        .sort_index()
        .reset_index()
    )

def entity_names(dataset_name):
    timeid = 'timestamp'
    userid = 'userid'
    if dataset_name.lower().startswith('ml-'):
        itemid = 'movieid'
    elif dataset_name.lower().startswith('amz'):
        itemid = 'asin'
    elif dataset_name.lower().startswith('steam'):
        itemid = 'product_id'
    else:
        raise ValueError('Unrecognized dataset')
    return userid, itemid, timeid

def data_to_df(dataset, userid, itemid, timeid):
    *data, usernum, itemnum = dataset
    def convert_data(dat):
        return (
            pd.Series(dat, name=itemid)
            .rename_axis(userid)
            .explode() # ravel item lists if present
            .reset_index()
            .assign(**{f'{timeid}': lambda x: range(len(x))})
        )
    return [convert_data(d) for d in data] + [usernum, itemnum]

def read_dataset(dataset_name, offset_str, stepwise_eval=False, part='all'):
    is_validation = part.lower().startswith('valid')
    read_all = part.lower() == 'all'
    dataset_valid = dataset_test = None
    # read data
    if offset_str:
        data = pd.read_csv(f'./data/{dataset_name}.gz')
        userid, itemid, timeid = entities = entity_names(dataset_name)
        train, valid, test = split_offsets(data, offset_str, timeid)
    else:
        user_train, user_valid, user_test, *stats = data_partition(dataset_name)
    # prepare validation part
    if is_validation or read_all:
        if offset_str:
            dataset_valid = generate_partition(train, valid, *entities, stepwise_eval=stepwise_eval)
        else:
            dataset_valid = [user_train, user_valid] + stats
    # prepare test part
    if not is_validation:
        if offset_str:
            train_full = train.append(valid)
            dataset_test = generate_partition(train_full, test, *entities, stepwise_eval=stepwise_eval)
        else:
            user_train_full = {user: user_train.get(user, []) + [item] for user, item in user_valid.items()}
            dataset_test = [user_train_full, user_test] + stats
    return dataset_valid, dataset_test

def sequential_training_data(user_train, userid, itemid, maxlen, seqid='position'):
    train_data = generate_sequential(user_train, userid, itemid, maxlen, seqid=seqid)
    recdata = RecommenderData(train_data, userid, itemid, feedback=seqid)
    recdata.verbose = False
    recdata.prepare_training_only()
    return recdata


def display_stats(dataset):
    [observed, holdout, usernum, itemnum] = dataset
    if isinstance(holdout, dict):
        num_test_users = {0: len(holdout)}
        print(f'# of test users per step: {num_test_users}')
    elif isinstance(holdout, list):
        num_test_events = len(holdout)
        print(f'# of test events: {num_test_events}')
    else:
        num_test_users = holdout.apply(len).to_dict()
        print(f'# of test users per step:\n{num_test_users}')
