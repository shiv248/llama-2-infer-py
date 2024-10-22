import tensorflow as tf

class TransformerModel(tf.keras.Model):
    def __init__(self, config):
        super(TransformerModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.dim)
        self.position_embedding = self.add_weight("pos_embedding", shape=[config.seq_len, config.dim])
        self.layers = [TransformerLayer(config) for _ in range(config.n_layers)]
        self.ln_final = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.head = tf.keras.layers.Dense(config.vocab_size)

    def call(self, inputs, pos, training=False):
        # Token + positional embeddings
        x = self.embedding(inputs) + self.position_embedding[pos, :]

        # Forward pass through each transformer layer
        for layer in self.layers:
            x = layer(x, training=training)
        
        # Final layer normalization and output head
        x = self.ln_final(x)
        logits = self.head(x)
        return logits


class TransformerLayer(tf.keras.layers.Layer):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=config.n_heads, key_dim=config.dim)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(config.hidden_dim, activation='swish'),  # SiLU activation
            tf.keras.layers.Dense(config.dim)
        ])
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, training=False):
        # Self-attention with residual connection
        attn_output = self.attn(x, x)
        x = self.ln1(x + attn_output)
        
        # Feedforward network with residual connection
        ffn_output = self.ffn(x)
        x = self.ln2(x + ffn_output)
        
        return x


def bpe_encode(text, vocab, vocab_scores):
    tokens = []
    for char in text:
        if char in vocab:
            tokens.append(vocab.index(char))
        else:
            raise ValueError(f"Token {char} not in vocab!")
    return tokens


def run_model(args, config, vocab, vocab_scores):
    # Set up the model
    model = TransformerModel(config)
    
    # Load the checkpoint (assuming a checkpoint is saved with tf.keras.Model's save/restore functions)
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(args["checkpoint"]).assert_existing_objects_matched()

    # Process the prompt if provided
    if args["prompt"]:
        prompt_tokens = bpe_encode(args["prompt"], vocab, vocab_scores)
    else:
        prompt_tokens = [1]  # BOS token
    
    # Run through the model
    tokens = tf.convert_to_tensor([prompt_tokens], dtype=tf.int32)  # Shape (1, seq_len)
    pos = tf.range(len(prompt_tokens), dtype=tf.int32)[tf.newaxis, :]  # Shape (1, seq_len)

    for step in range(args['steps']):
        logits = model(tokens, pos, training=False)
        next_token = tf.argmax(logits[:, -1, :], axis=-1)
        tokens = tf.concat([tokens, tf.expand_dims(next_token, axis=1)], axis=1)
        pos = tf.concat([pos, [[step + 1]]], axis=1)

        # Output the generated token
        print(vocab[next_token.numpy()[0]], end='')
        if next_token.numpy()[0] == 1:  # EOS token
            break

if __name__ == "__main__":
    args = {
        "checkpoint": './out/stories15M.bin',  # Path to the saved checkpoint
        "temperature": 0.0,  # Not used in greedy decoding
        "steps": 256,  # Max number of steps to generate
        "prompt": "Once upon a time"  # Example prompt
    }

    config = Config(
        dim=512, hidden_dim=2048, n_layers=12, n_heads=8, n_kv_heads=8,
        vocab_size=30522, seq_len=512  # Replace these with your actual config
    )

    vocab = ["<pad>", "<s>", "</s>", "Once", "upon", "a", "time", ...]  # Example vocabulary
    vocab_scores = [0.0] * len(vocab)  # Example vocab scores
    
    run_model(args, config, vocab, vocab_scores)
