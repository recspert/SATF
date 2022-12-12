import numpy as np
from polara import RandomModel
from .svd import HitPredictionMixin

class RandomRec(HitPredictionMixin, RandomModel):
    def build(self):
        super().build()
        self.item_index = (
            self.data
            .get_entity_index(self.data.fields.itemid)
            .set_index('old').new
        )
        self.random_state = np.random.RandomState(self.seed)
        self.num_items = len(self.item_index)
    
    def score(self, seq):
        return self.random_state.random(self.num_items)