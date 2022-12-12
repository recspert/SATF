import numpy as np
from scipy.signal import fftconvolve
from scipy.linalg import solve_banded
from scipy.sparse.linalg import LinearOperator, svds
from numba import njit, prange
from polara.recommender.models import CoffeeModel
from polara.lib.sparse import arrange_indices
from polara.tools.timing import track_time

from .svd import HitPredictionMixin, ItemProjectorMixin
from .gsatf import form_attention_matrix, generate_banded_form, core_growth_callback, get_scaling_weights, initialize_columnwise_orthonormal

PARALLEL_MATVECS = True
FASTMATH_MATVECS = True
DTYPE = np.float64

class HOSSAError(Exception):
    pass


def hankel_series_compress_2d(left_factors, right_factors):
    hankel_compressed = fftconvolve(left_factors[:, np.newaxis, :], right_factors[:, :, np.newaxis])
    return hankel_compressed.reshape(hankel_compressed.shape[0], -1)


@njit(parallel=False)
def hankel_series_matvec1(pos, vec, dim1, dim2):
    res = np.empty(dim1, dtype='f8')
    for i in range(dim1):
        j = pos - i
        val = 0.
        if j >= 0:
            if j < dim2:
                val = vec[j]
        res[i] = val
    return res


def accum_kron_weights(arranged_position_index, idx, entity1_factors, entity2_factors):
    e1_rank = entity1_factors.shape[1]
    e2_rank = entity2_factors.shape[1]
    n_points = len(arranged_position_index)
    res = np.empty((n_points, e2_rank, e1_rank), dtype=np.float64)
    for n, idx_rows in enumerate(arranged_position_index):
        e1_idx = idx[idx_rows, 0]
        e2_idx = idx[idx_rows, 1]
        w1 = entity1_factors[e1_idx, :]
        w2 = entity2_factors[e2_idx, :]
        accum_w_i = w2.T @ w1
        res[n, ...] = accum_w_i
    return res.reshape(n_points, e2_rank*e1_rank)


def series_matvec(factors, series_key, attention=None):
    series = ['attention', 'sequences']
    other_series = series[1 - series.index(series_key)]
    sf_dim, sf_rank = factors[other_series].shape
    n_pos, ee_rank = factors['accum_w_i'].shape
    dim_2 = sf_dim
    dim_1 = n_pos - dim_2 + 1
    
    @njit(parallel=PARALLEL_MATVECS, fastmath=FASTMATH_MATVECS)
    def hankel_series_mul(W, V, accum_w_i, n_pos, dim_1, dim_2):
        res = np.empty((n_pos, dim_1), dtype=np.float64)
        for i in prange(n_pos):
            tmp = np.dot(V, accum_w_i[i])
            vec = np.dot(W, tmp)
            res[i, :] = hankel_series_matvec1(i, vec, dim_1, dim_2)
        return res.sum(axis=0)
    
    def matvec(vec):
        series_factors = factors[other_series]
        accum_w_i = factors['accum_w_i']
        V = vec.reshape(ee_rank, sf_rank).T
        res = hankel_series_mul(series_factors, V, accum_w_i, n_pos, dim_1, dim_2)
        if attention is not None:
            res = attention.T.dot(res)
        return res
    return matvec


def series_rmatvec(factors, series_key, attention=None):
    series = ['attention', 'sequences']
    other_series = series[1 - series.index(series_key)]
    def rmatvec(vec):
        series_factors = factors[other_series]
        accum_w_i = factors['accum_w_i']
        if attention is not None:
            vec = attention.dot(vec)
        hankel_weights = fftconvolve(series_factors, vec[:, np.newaxis])
        res = accum_w_i.T @ hankel_weights
        return res.ravel()
    return rmatvec


@njit(parallel=PARALLEL_MATVECS, fastmath=FASTMATH_MATVECS)
def entity_fiber_matvec(arranged_entity_index, idx, entity_mode, other_factors, hankel_weights, vec):
    _, o_rank = other_factors.shape
    _, s_rank = hankel_weights.shape
    V = vec.reshape(o_rank, s_rank) # kron(b, a) x = a^T X b
    VX_cache = np.dot(hankel_weights, V.T)

    other_entity_mode = 1 - entity_mode
    n_entities = len(arranged_entity_index)
    res = np.empty(n_entities, dtype=np.float64)
    for main_entity_id in prange(n_entities):
        idx_rows = arranged_entity_index[main_entity_id]
        tmp = 0.
        for row_id in idx_rows:
            pos = idx[row_id, 2]
            other_entity_id = idx[row_id, other_entity_mode]
            w_i = other_factors[other_entity_id, :]
            tmp += np.dot(VX_cache[pos, :], w_i) # kron(b, a) x = a^T X b
        res[main_entity_id] = tmp
    return res

def entity_matvec(arranged_entity_index, idx, factors, entity_key, scaling=None):
    entities = ['users', 'items']
    entity_mode = entities.index(entity_key)
    other_entity = entities[1-entity_mode]
    def matvec(vec):
        other_factors = factors[other_entity]
        hankel_weights = factors['hankel_weights']
        result = entity_fiber_matvec(arranged_entity_index, idx, entity_mode, other_factors, hankel_weights, vec)
        if scaling is not None:
            result = scaling * result
        return result
    return matvec


@njit(parallel=PARALLEL_MATVECS, fastmath=FASTMATH_MATVECS)
def entity_fiber_rmatvec(arranged_position_index, idx, entity_mode, other_factors, hankel_weights, vec):
    _, o_rank = other_factors.shape
    n_pos = len(arranged_position_index)
    other_entity_mode = 1 - entity_mode
    accum_wev = np.empty((n_pos, o_rank), dtype=np.float64)
    for i in prange(n_pos):
        idx_rows = arranged_position_index[i]
        wev = np.zeros(o_rank, dtype=np.float64)
        for row_id in idx_rows:
            entities = idx[row_id, :2]
            main_entity = entities[entity_mode]
            other_entity = entities[other_entity_mode]
            val = vec[main_entity]
            w_i = other_factors[other_entity, :]
            wev += w_i * val
        accum_wev[i, :] = wev
    res = np.dot(accum_wev.T, hankel_weights)
    return res.ravel()


def entity_rmatvec(arranged_position_index, idx, factors, entity_key, scaling=None):
    entities = ['users', 'items']
    entity_mode = entities.index(entity_key)
    other_entity = entities[1-entity_mode]
    def rmatvec(vec):
        other_factors = factors[other_entity]
        hankel_weights = factors['hankel_weights']
        if scaling is not None:
            vec = vec * scaling
        result = entity_fiber_rmatvec(arranged_position_index, idx, entity_mode, other_factors, hankel_weights, vec)
        return result
    return rmatvec


def hankel_hooi(
    idx, shape, mlrank, attention_matrix, scaling_weights,
    max_iters=20,
    growth_tol=0.001,
    seed=None,
    iter_callback=None,
    ):
    user_rank, item_rank, attn_rank, seqn_rank = mlrank
    n_users, n_items, n_positions = shape
    attention_span = attention_matrix.shape[0]
    sequences_size = n_positions - attention_span + 1

    arranged_indices = arrange_indices(idx, [True, True, True], shape=shape)
    arranged_user_index = arranged_indices[0][1]
    arranged_item_index = arranged_indices[1][1]
    arranged_position_index = arranged_indices[2][1]
    
    factors_store = {}
    rnd = np.random.RandomState(seed)
    factors_store['users'] = np.empty((n_users, user_rank), dtype=np.float64) # only to initialize linear operators
    factors_store['items'] = scaling_weights[:, np.newaxis] * initialize_columnwise_orthonormal((n_items, item_rank), rnd)
    attn_factors = factors_store['attention'] = attention_matrix.dot(initialize_columnwise_orthonormal((attention_span, attn_rank), rnd))
    seqn_factors = factors_store['sequences'] = initialize_columnwise_orthonormal((sequences_size, seqn_rank), rnd)

    factors_store['accum_w_i'] = np.empty((n_positions, user_rank*item_rank), dtype=np.float64) # only to initialize linear operators
    factors_store['hankel_weights'] = np.empty((n_positions, attn_rank*seqn_rank), dtype=np.float64) # only to initialize linear operators

    attn_matvec = series_matvec(factors_store, 'attention', attention_matrix.tocsc())
    attn_rmatvec = series_rmatvec(factors_store, 'attention', attention_matrix.tocsr())
    attn_linop = LinearOperator((attention_span, seqn_rank*user_rank*item_rank), attn_matvec, attn_rmatvec)
    
    seqn_matvec = series_matvec(factors_store, 'sequences')
    seqn_rmatvec = series_rmatvec(factors_store, 'sequences')
    seqn_linop = LinearOperator((sequences_size, attn_rank*user_rank*item_rank), seqn_matvec, seqn_rmatvec)
    
    user_matvec = entity_matvec(arranged_user_index, idx, factors_store, 'users')
    user_rmatvec = entity_rmatvec(arranged_position_index, idx, factors_store, 'users')
    user_linop = LinearOperator((n_users, attn_rank*seqn_rank*item_rank), user_matvec, user_rmatvec)
    
    item_matvec = entity_matvec(arranged_item_index, idx, factors_store, 'items', scaling_weights)
    item_rmatvec = entity_rmatvec(arranged_position_index, idx, factors_store, 'items', scaling_weights)
    item_linop = LinearOperator((n_items, attn_rank*seqn_rank*user_rank), item_matvec, item_rmatvec)
    
    if iter_callback is None:
        iter_callback = core_growth_callback(growth_tol)

    for step in range(max_iters):
        factors_store['hankel_weights'] = hankel_series_compress_2d(attn_factors, seqn_factors)

        u_user, *_ = svds(user_linop, k=user_rank, return_singular_vectors='u')
        user_factors = factors_store['users'] = np.ascontiguousarray(u_user)
        
        u_item, *_ = svds(item_linop, k=item_rank, return_singular_vectors='u')
        item_factors = factors_store['items'] = np.ascontiguousarray(u_item * scaling_weights[:, np.newaxis])    

        factors_store['accum_w_i'] = accum_kron_weights(arranged_position_index, idx, user_factors, item_factors)

        u_attn, *_ = svds(attn_linop, k=attn_rank, return_singular_vectors='u')
        attn_factors = factors_store['attention'] = np.ascontiguousarray(attention_matrix.dot(u_attn))
        
        u_seqn, ss, _ = svds(seqn_linop, k=seqn_rank, return_singular_vectors='u')
        seqn_factors = factors_store['sequences'] = np.ascontiguousarray(u_seqn)
        
        raw_factors = (u_user, u_item, u_attn, u_seqn)
        try:
            core_norm = np.linalg.norm(ss)
            iter_callback(step, core_norm, raw_factors)
        except StopIteration:
            break
    return raw_factors


class SequentialAttentionTensor(HitPredictionMixin, ItemProjectorMixin, CoffeeModel):
    def __init__(
        self,
        *args,
        attention_window=5,
        attention_decay=1.0,
        attention_cutoff=1e-6,
        attention_span=0,
        exponential_decay=False,
        reversed_attention=False,
        stochastic_attention_axis=None,
        scaling=1.0,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.attention_window = attention_window
        self.attention_matrix = None
        self.attention_decay = attention_decay
        self.attention_cutoff = attention_cutoff or 0
        self.attention_span = attention_span or np.iinfo('i8').max
        self.reversed_attention = reversed_attention
        self.exponential_decay = exponential_decay
        self.stochastic_attention_axis = stochastic_attention_axis
        self.rescaled = False
        self.scaling = scaling
        self.scaling_weights = None
        self.method = 'LA-SATF'

    def _check_reduced_rank(self, mlrank):
        pass

    def validate_index(self, idx):
        position_index = self.data.index.feedback.query('old >= 0')
        is_valid_position = np.isin(idx[:, 2], position_index.new.values)
        
        idx = idx[is_valid_position, :]
        idx[:, 2] = position_index.set_index('new').loc[idx[:, 2], 'old'].values
        return idx, is_valid_position
    
    def get_training_data(self):
        idx, val, shp = self.data.to_coo(tensor_mode=True)
        idx, is_valid_position = self.validate_index(idx)
        shp = (shp[0], shp[1], self.data.index.feedback.old.max()+1)
        val = val[is_valid_position]
        return idx, val, shp 

    def build(self, callback=None):
        idx, val, shp = self.get_training_data()
        n_users, n_items, n_positions = shp
        u_rank, i_rank, a_rank, s_rank = self.mlrank
        attention_span = self.attention_window
        sequences_size = n_positions - attention_span + 1
        
        if attention_span < a_rank:
            raise HOSSAError('Attention mode rank cannot exceed the attention span.')
        if sequences_size < s_rank:
            raise HOSSAError('Sequences mode rank cannot exceed the sequences size.')
        if n_users < u_rank:
            raise HOSSAError('Users mode rank cannot exceed the number of users.')
        if n_items < i_rank:
            raise HOSSAError('Items mode rank cannot exceed the number of items.')

        self.attention_matrix = form_attention_matrix(
            self.attention_window,
            self.attention_decay,
            cutoff = self.attention_cutoff,
            span = self.attention_span,
            exponential_decay = self.exponential_decay,
            reverse = self.reversed_attention,
            stochastic_axis = self.stochastic_attention_axis,
            format = 'csr'
        )
        itemid = self.data.fields.itemid
        self.item_index = (
            self.data
            .get_entity_index(itemid)
            .set_index('old').new
        )
        item_popularity = self.data.training[itemid].value_counts(sort=False).sort_index().values
        self.scaling_weights = get_scaling_weights(item_popularity, scaling=self.scaling)
        
        with track_time(self.training_time, verbose=self.verbose, model=self.method):
            user_factors, item_factors, attention_factors, sequences_factors = hankel_hooi(
                idx, shp, self.mlrank,
                attention_matrix = self.attention_matrix,
                scaling_weights = self.scaling_weights,
                max_iters = self.num_iters,
                growth_tol = self.growth_tol,
                seed = self.seed,
                iter_callback=callback,
            )
        self.store_factors(user_factors, item_factors, attention_factors, sequences_factors)

    def store_factors(self, user_factors, item_factors, attention_factors, sequences_factors):
        self.factors[self.data.fields.userid] = user_factors
        self.factors[self.data.fields.itemid] = item_factors
        self.factors['attention'] = attention_factors
        self.factors['sequences'] = sequences_factors
        self.store_item_projector(item_factors)
        self.store_position_projector(attention_factors, sequences_factors)
    
    def store_position_projector(self, attention_factors, sequences_factors):
        wa_left = solve_banded(*generate_banded_form(self.attention_matrix.T), attention_factors)
        wa_right = self.attention_matrix.dot(attention_factors)
        attention_projector_last = wa_right @ wa_left[-1, :]
        sequences_projector_last = sequences_factors @ sequences_factors[-1, :]
        self.factors['hankel_weights_last'] = fftconvolve(attention_projector_last, sequences_projector_last)
    
    def score(self, seq):
        item_proj_left, item_proj_right = self.get_item_projector()
        hankel_weights = self.factors['hankel_weights_last']
        known_items = self.item_index.loc[seq].values
        maxlen = len(hankel_weights)
        items = known_items[-(maxlen-1):] # prepare to shift left by 1 step
        start_pos = maxlen - len(items)
        weights = hankel_weights[start_pos-1:-1] # last item is the one we predict for
        scores = item_proj_left @ np.dot(item_proj_right[items, :].T, weights)
        return scores