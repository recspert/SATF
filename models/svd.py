import numpy as np
from polara import SVDModel
from polara.preprocessing.matrices import rescale_matrix
from polara.tools.timing import track_time
try:
    from sklearn.utils.extmath import randomized_svd
except ImportError:
    randomized_svd = None

class SVDError(Exception): pass


class ItemProjectorMixin:
    def store_item_projector(self, item_factors):
        if self.rescaled:
            self.factors['item_projector_left'] = item_factors / self.scaling_weights[:, np.newaxis]
            self.factors['item_projector_right'] = item_factors * self.scaling_weights[:, np.newaxis]
        else:
            self.factors['item_projector_left'] = self.factors['item_projector_right'] = item_factors

    def get_item_projector(self):
        if self.rescaled:
            vl = self.factors['item_projector_left']
            vr = self.factors['item_projector_right']
        else:
            vl = vr = self.factors[self.data.fields.itemid]
        return vl, vr

class HitPredictionMixin:
    def check_hit(self, target_item, seq, topn):
        predictions = self.score(seq)
        seen_items = self.item_index.loc[seq].values
        np.put(predictions, seen_items, predictions.min()-1)
        predicted_items = self.topsort(predictions, topn)

        true_item = self.item_index.loc[target_item] # extract item from (item, time) tuple
        (hit_index,) = np.where(predicted_items == true_item)
        return hit_index, predicted_items


class SVD(HitPredictionMixin, ItemProjectorMixin, SVDModel):
    def __init__(self, *args, randomized=True, **kwargs):
        if randomized and randomized_svd is None:
            raise SVDError('Randomized SVD is unavailable')
        super().__init__(*args, **kwargs)
        self.rescaled = False
        self.scaling = None
        self.scaling_weights = None
        self.randomized = randomized
        self.method = 'PureSVD'
        if self.randomized:
            self.method += '-r'
        if self.rescaled:
            self.method += '-s'

    def get_training_matrix(self, *args, **kwargs):
        matrix = super().get_training_matrix(*args, **kwargs)
        scaled_matrix, self.scaling_weights = rescale_matrix(
            matrix, self.scaling, 0, binary=True, return_scaling_values=True
        )
        self.item_index = (
            self.data
            .get_entity_index(self.data.fields.itemid)
            .set_index('old').new
        )        
        return scaled_matrix

    def build(self):
        itemid = self.data.fields.itemid
        if self.randomized:
            svd_matrix = self.get_training_matrix(dtype=np.float64)
            with track_time(self.training_time, verbose=self.verbose, model=self.method):
                *_, item_factors_t = randomized_svd(svd_matrix, self.rank)
                self.factors[itemid] = item_factors_t.T
        else:
            super().build()
        self.store_item_projector(self.factors[itemid])
        
    def score(self, seq):
        vl, vr = self.get_item_projector()
        known_items = self.item_index.loc[seq].values
        user_profile = vr[known_items, :].sum(axis=0)
        scores = vl @ user_profile
        return scores
