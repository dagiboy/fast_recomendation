import json
from tqdm.auto import tqdm
from collections import defaultdict
import pickle

K = 10

def arg_min(l):
    cur_min = l[0]
    arg = 0

    for i, elem in enumerate(l):
        if elem < cur_min:
            elem = cur_min
            arg = i

    return arg, cur_min

def get_sim(item1: int, item2: int, item_to_users: dict[int, set[int]]) -> float:
    intersection = item_to_users[item1].intersection(item_to_users[item2])
    if len(intersection) == 0:
        return 0
    
    union = item_to_users[item1].union(item_to_users[item2])
    return len(intersection) / len(union)

def read_item_user_info(file_name: str):
    item_to_users = defaultdict(list)
    user_to_items = defaultdict(list)


    with open(file_name) as f:
        for row in tqdm(f, total=1671803, desc='Read data'):
            j = json.loads(row)
            session = j['session']

            for event in j['events']:
                item = event['aid']
                item_to_users[item].append(session)
                user_to_items[session].append(item)

    return dict(item_to_users), dict(user_to_items)

def renumber_items_users(item_to_users_raw: dict[int, list[int]], 
                         user_to_items_raw: dict[int, list[int]]) \
                            -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    users = list(user_to_items_raw.keys())
    user_to_idx = {user: i for i, user in enumerate(users)}

    items = list(item_to_users_raw.keys())
    item_to_idx = {item: i for i, item in enumerate(items)}

    item_to_users = defaultdict(list)
    user_to_items = defaultdict(list)

    for item, users in tqdm(item_to_users_raw.items(), desc='Renumber'):
        item_idx = item_to_idx[item]
        for user in users:
            user_idx = user_to_idx[user]

            item_to_users[item_idx].append(user_idx)
            user_to_items[user_idx].append(item_idx)

    return dict(item_to_users), dict(user_to_items)

def to_set(src: dict[int, list[int]]) -> dict[int, set[int]]:
    for key, values in src.items():
        src[key] = set(values)

    return src

def get_populest(item_to_users: dict[int, set[int]], k=K) -> list[int]:
    populest = [0] * k
    popularities = [0] * k

    cur_min = 0
    cur_arg = 0

    for item, users in tqdm(item_to_users.items(), desc='Find populest'):
        cur_popularity = len(users)

        if cur_popularity < cur_min:
            continue

        populest[cur_arg] = item
        popularities[cur_arg] = cur_popularity
        cur_arg, cur_min = arg_min(popularities)

    return populest

def compute_top_k(item_to_users:dict[int, set[int]], user_to_items: dict[int, set[int]], 
                  populest: list[int], k=K) -> dict[int, set[int]]:
    top_k = dict()

    for item1, users in tqdm(item_to_users.items(), desc='Find top_k'):
        top_sim = [-1] * K
        top_items = list(populest)

        for i, item2 in enumerate(top_items):
            top_sim[i] = get_sim(item1, item2, item_to_users)
        
        cur_arg, cur_min = arg_min(top_sim)

        processed_items = set([item1])

        for user in users:
            for item2 in user_to_items[user]:
                if item2 in processed_items:
                    continue

                processed_items.add(item2)
                sim = get_sim(item1, item2, item_to_users)

                if sim <= cur_min:
                    continue

                top_sim[cur_arg] = sim
                top_items[cur_arg] = item2
                cur_arg, cur_min = arg_min(top_sim)

        top_k[item1] = set(top_items)

def to_pickle(obj, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)

def main():
    item_to_users_raw, user_to_items_raw = read_item_user_info('data/otto-recsys-test.jsonl')
    item_to_users, user_to_items = renumber_items_users(item_to_users_raw, user_to_items_raw)

    item_to_users = to_set(item_to_users)
    user_to_items = to_set(user_to_items)

    populest = get_populest(item_to_users, K)
    top_k = compute_top_k(item_to_users, user_to_items, populest, K)

    to_pickle(item_to_users, "item_to_users_inf.pkl")
    to_pickle(user_to_items, "user_to_items_inf.pkl")
    to_pickle(top_k, "top_k_info.pkl")

if __name__ == "__main__":
    main()   
