from polara import PopularityModel
from .svd import HitPredictionMixin

class MostPopular(HitPredictionMixin, PopularityModel):
    def build(self):
        super().build() # populates self.item_scores with popularity weights
        self.item_index = (
            self.data
            .get_entity_index(self.data.fields.itemid)
            .set_index('old').new
        )
    
    def score(self, seq):
        return self.item_scores