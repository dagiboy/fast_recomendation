import json

from flask import Flask, request, jsonify
import pickle
from collections import defaultdict

app = Flask(__name__, static_url_path="")

def load_from_pickle(file_name: str):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

item_to_users: dict[int, set[int]] = load_from_pickle("data/item_to_users_inf.pkl")
items = list(item_to_users.keys())
print("Max item:", max(items))

user_to_items: dict[int, set[int]] = load_from_pickle("data/user_to_items_inf.pkl")
user_to_items = defaultdict(set, user_to_items)
users = list(user_to_items.keys())
print("Max user:", max(users))

top_k = load_from_pickle("data/top_k_info.pkl")

def get_sim(item1: int, item2: int, item_to_users: dict[int, set[int]]) -> float:
    intersection = item_to_users[item1].intersection(item_to_users[item2])
    if len(intersection) == 0:
        return 0
    
    union = item_to_users[item1].union(item_to_users[item2])
    return len(intersection) / len(union)


@app.route("/recommend", methods=['GET'])
def predict():
    user = int(request.args.get('user'))
    item = int(request.args.get('item'))

    user_to_items[user].add(item)


    C = set()
    for hist_item in user_to_items[user]:
        if hist_item in top_k:
            C = C.union(top_k[hist_item])

    C = list(C)
    sims = [0] * len(C)
    for i, cand in enumerate(C):
        if cand in user_to_items[user]:
            continue

        for hist_item in user_to_items[user]:
            sims[i] += get_sim(hist_item, cand, item_to_users)

    item_weights = sorted(zip(C, sims), key=lambda x: x[1], reverse=True)
    item_weights = list(item_weights)[:10]

    recommends, weights = list(zip(*item_weights))

    return jsonify({
        "recomends": recommends,
        "weights": weights
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)