"""
SushiGo parity check — Python side.

Paste the JSON that SGParityCheck.java printed into JSON_FROM_JAVA below, and
paste the doubleVector array into VEC_FROM_JAVA. Run this script: it processes
the JSON through the author's exact wrapper logic and compares element by
element against the Java vector.

Run from the PyTAG repo root (so the import resolves):
    python check_parity.py
"""
import json
import numpy as np

# ---- paste from SGParityCheck.java output ----
JSON_FROM_JAVA = r'''{"PlayerID":1,"opp0playedCards":"Tempura,Dumpling,Pudding","opp0score":1.0,"cardsInHand":"Tempura,SalmonNigiri,Maki-3,Sashimi,Maki-2,Maki-2,Tempura","playerScore":3.0,"nPlayers":2,"rounds":0,"playedCards":"Dumpling,Maki,Dumpling"}'''
VEC_FROM_JAVA  = [0.0600, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0.0200]  # paste the [ ... ] array here (Python list literal)
# ----------------------------------------------

# The author's processing, copied verbatim from pytag/utils/wrappers.py
CARD_TYPES = ["Maki", "Maki-2", "Maki-3", "Chopsticks", "Tempura", "Sashimi",
              "Dumpling", "SquidNigiri", "SalmonNigiri", "EggNigiri", "Wasabi", "Pudding"]
MAX_CARDS_IN_HAND = 10

def get_card_id(card):
    emb = np.zeros(len(CARD_TYPES))
    if card != "EmptyDeck":
        if card in CARD_TYPES:
            emb[CARD_TYPES.index(card)] = 1
    return emb

def process_json_obs(json_obs):
    json_ = json.loads(str(json_obs))
    played_cards = json_["playedCards"].split(",")
    cards_in_hand = json_["cardsInHand"].split(",")
    score = json_["playerScore"] / 50
    rnd = json_["rounds"] / 3

    opp_scores = []
    opponent_played_cards_ = []
    for key in json_.keys():
        if "opp" in key and "playedCards" in key:
            opp_played_cards = json_[key].split(",")
            opponent_played_cards_.append([get_card_id(c) for c in opp_played_cards])
        if "opp" in key and "score" in key:
            opp_scores.append(json_[key] / 50)

    played_cards_ = [get_card_id(c) for c in played_cards]
    cards_in_hand_ = [get_card_id(c) for c in cards_in_hand]
    while len(cards_in_hand_) < MAX_CARDS_IN_HAND:
        cards_in_hand_.append(np.zeros(len(CARD_TYPES)))

    score = np.expand_dims(score, 0)
    rnd = np.expand_dims(rnd, 0)
    played_cards = np.sum(played_cards_, axis=0)
    cards_in_hand = np.stack(cards_in_hand_, 0).flatten()
    opp_played_cards = np.sum(opponent_played_cards_, axis=1).flatten()
    obs = np.concatenate([score, rnd, played_cards, cards_in_hand, opp_played_cards, opp_scores])
    return obs

py = process_json_obs(JSON_FROM_JAVA)
ja = np.array(VEC_FROM_JAVA, dtype=np.float32)

print(f"Python length: {len(py)}   Java length: {len(ja)}")
if len(py) != len(ja):
    print("LENGTH MISMATCH — layouts differ. Stop here and inspect.")
else:
    diff = np.abs(py - ja)
    print(f"Max abs diff: {diff.max():.6e}")
    if diff.max() < 1e-4:
        print("PARITY OK — vectors match.")
    else:
        bad = np.where(diff > 1e-4)[0]
        print(f"MISMATCH at {len(bad)} indices: {bad[:20].tolist()}{'...' if len(bad)>20 else ''}")
        for i in bad[:10]:
            print(f"  idx {i}: python={py[i]:.4f}  java={ja[i]:.4f}")
