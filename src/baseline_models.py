import random
from collections import Counter

import random
from collections import Counter

import random
from collections import Counter

# Baseline osztályok
class RandomBaseline:
    def __init__(self, seed=None):
        self.seed = seed

    def predict_rank(self, true_tail, candidate_list):
        if self.seed is not None:
            random.seed(self.seed)
        shuffled = candidate_list.copy()
        random.shuffle(shuffled)
        return shuffled.index(true_tail), shuffled

    def score_and_hits(self, true_tail, candidate_list, k_list):
        rank, shuffled = self.predict_rank(true_tail, candidate_list)

        # MRR FULL (normál MRR)
        mrr_full = 1.0 / (rank + 1)

        # MRR@10
        mrr_at_10 = 1.0 / (rank + 1) if rank < 10 else 0.0

        # hits@k
        hits = {k: int(rank < k) for k in k_list}

        return mrr_full, mrr_at_10, hits, shuffled

class MostPopularBaseline:
    def __init__(self):
        """
        top_popular_items: list of item_ids sorted by popularity (most popular first)
        Example: [75, 12, 44, ...] length ~100
        """
        self.top_popular_items = {'1658': 75, '5175': 549, '2980': 320, '94': 136, '3053': 892, '3565': 118, '334': 87, '4404': 276, '3584': 84,
         '4395': 411, '2740': 208, '3059': 148, '5326': 51, '7447': 288, '5435': 1062, '4393': 242, '476': 315,
         '4398': 178, '7380': 329, '3568': 392, '4458': 993, '5323': 480, '3575': 916, '2603': 389, '5177': 965,
         '1653': 817, '7016': 977, '5809': 398, '5404': 29, '3332': 1516, '6494': 865, '415': 324, '4935': 67,
         '5096': 78, '82': 40, '2922': 605, '2482': 404, '231': 146, '2450': 553, '2989': 821, '7928': 354, '433': 190,
         '5403': 548, '3995': 1568, '628': 306, '6385': 589, '4630': 439, '6965': 317, '3756': 867, '5118': 938,
         '4936': 448, '3439': 328, '5233': 336, '5405': 806, '6878': 1375, '2839': 341, '5393': 1586, '3057': 43,
         '119': 482, '3812': 312, '5564': 410, '4627': 339, '5799': 374, '2904': 2239, '2316': 431, '175': 790,
         '5103': 1419, '4038': 440, '2287': 111, '3889': 153, '3128': 594, '5355': 839, '7426': 95, '4310': 974,
         '4405': 119, '3447': 332, '2467': 1314, '5511': 1322, '249': 216, '2522': 472, '4926': 4, '5538': 333,
         '159': 361, '7437': 1096, '2367': 335, '7280': 378, '6820': 257, '5534': 836, '4360': 1856, '4970': 830,
         '6800': 1981, '1610': 495, '661': 815, '5574': 3739, '4504': 245, '4302': 61, '4096': 144, '3989': 300,
         '3088': 844, '4636': 957}

        self.top_popular_items = list(self.top_popular_items.values())
        self.rank_map = {item_id: rank for rank, item_id in enumerate(self.top_popular_items)}

    def predict_rank(self, true_tail, candidate_list):
        """
        Return a single popularity-based rank for the item.
        (Hits for multiple K values are handled outside.)
        """
        if true_tail in self.rank_map:
            return self.rank_map[true_tail]
        else:
            return len(candidate_list)


    def score_and_hits(self, true_tail, candidate_list, k_list=(1,3,5,10)):
        k_list = list(k_list)

        rank = self.predict_rank(true_tail, candidate_list)

        # MRR FULL
        mrr_full = 1.0 / (rank + 1) if rank < len(candidate_list) else 0.0

        # MRR@10
        mrr_at_10 = 1.0 / (rank + 1) if rank < 10 else 0.0

        hits = {k: int(rank < k) for k in k_list}

        return mrr_full, mrr_at_10, hits