from flask import Flask, request, jsonify
import pickle
from collections import defaultdict

from process_data import (get_sim, TOP_K_FILE_NAME, 
                           ITEM_TO_USERS_FILE_NAME, 
                           USER_TO_ITEMS_FILE_NAME)

app = Flask(__name__, static_url_path="")

def load_from_pickle(file_name: str):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

item_to_users: dict[int, set[int]] = load_from_pickle(ITEM_TO_USERS_FILE_NAME)
items = list(item_to_users.keys())
print("Max item:", max(items))

user_to_items: dict[int, set[int]] = load_from_pickle(USER_TO_ITEMS_FILE_NAME)
user_to_items = defaultdict(set, user_to_items)
users = list(user_to_items.keys())
print("Max user:", max(users))

top_k = load_from_pickle(TOP_K_FILE_NAME)


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