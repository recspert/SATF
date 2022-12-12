grid_config = dict(
    batch_size = [64, 128, 256, 512],
    lr = [0.00001, 0.0001, 0.001],
    hidden_units = [64, 128, 256, 512, 768],
    num_blocks = [1, 2, 3],
    num_heads = [1],
    dropout_rate = [0.2, 0.4, 0.6],
    l2_emb = [0.0],
)