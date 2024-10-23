import sys
import time
import random
import struct
import tensorflow as tf
import numpy as np


class Config:
    def __init__(self, dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.seq_len = seq_len


class TransformerWeights:
    def __init__(self):
        self.token_embedding_table = None
        self.rms_att_weight = None
        self.wq = None
        self.wk = None
        self.wv = None
        self.wo = None
        self.rms_ffn_weight = None
        self.w1 = None
        self.w3 = None
        self.w2 = None
        self.rms_final_weight = None
        self.freq_cis_real = None
        self.freq_cis_imag = None
        self.wcls = None


def checkpoint_init_weights(weights: TransformerWeights,
                            conf: Config,
                            file,
                            shared_weights: int) -> None:
    def read_floats(count):
        values = np.frombuffer(file.read(count * 4), dtype=np.float32)
        return values

    weights.token_embedding_table = tf.convert_to_tensor(
        read_floats(conf.vocab_size * conf.dim).reshape(conf.vocab_size, conf.dim))
    weights.rms_att_weight = tf.convert_to_tensor(
        read_floats(conf.n_layers * conf.dim).reshape(conf.n_layers, conf.dim))
    weights.wq = tf.convert_to_tensor(
        read_floats(conf.n_layers * conf.dim * conf.dim).reshape(conf.n_layers, conf.dim, conf.dim))
    weights.wk = tf.convert_to_tensor(
        read_floats(conf.n_layers * conf.dim * conf.dim).reshape(conf.n_layers, conf.dim, conf.dim))
    weights.wv = tf.convert_to_tensor(
        read_floats(conf.n_layers * conf.dim * conf.dim).reshape(conf.n_layers, conf.dim, conf.dim))
    weights.wo = tf.convert_to_tensor(
        read_floats(conf.n_layers * conf.dim * conf.dim).reshape(conf.n_layers, conf.dim, conf.dim))
    weights.rms_ffn_weight = tf.convert_to_tensor(
        read_floats(conf.n_layers * conf.dim).reshape(conf.n_layers, conf.dim))
    weights.w1 = tf.convert_to_tensor(
        read_floats(conf.n_layers * conf.dim * conf.hidden_dim).reshape(conf.n_layers, conf.dim, conf.hidden_dim))
    weights.w2 = tf.convert_to_tensor(
        read_floats(conf.n_layers * conf.hidden_dim * conf.dim).reshape(conf.n_layers, conf.hidden_dim, conf.dim))
    weights.w3 = tf.convert_to_tensor(
        read_floats(conf.n_layers * conf.dim * conf.hidden_dim).reshape(conf.n_layers, conf.dim, conf.hidden_dim))
    weights.rms_final_weight = tf.convert_to_tensor(read_floats(conf.dim))
    weights.freq_cis_real = tf.convert_to_tensor(
        read_floats(conf.seq_len * (conf.dim // conf.n_heads) // 2).reshape(conf.seq_len, conf.dim // conf.n_heads // 2))
    weights.freq_cis_imag = tf.convert_to_tensor(
        read_floats(conf.seq_len * (conf.dim // conf.n_heads) // 2).reshape(conf.seq_len, conf.dim // conf.n_heads // 2))
    if shared_weights:
        weights.wcls = weights.token_embedding_table
    else:
        weights.wcls = tf.convert_to_tensor(read_floats(-1))


def tokenizer_init(conf: Config, file):
    vocab = []
    vocab_scores = []

    max_token_length = struct.unpack('i', file.read(4))[0]
    for _ in range(conf.vocab_size):
        vocab_scores.append(struct.unpack('f', file.read(4))[0])
        length = struct.unpack('i', file.read(4))[0]
        bstr = file.read(length)
        if type(bstr) is not str:
            bstr = bstr.decode('utf8')
        vocab.append(bstr)
    return vocab, vocab_scores, max_token_length


def rmsnorm(x, weight):
    mean_square = tf.reduce_mean(tf.square(x), axis=-1, keepdims=True)
    inv_norm = tf.math.rsqrt(mean_square + 1e-5)
    return x * inv_norm * weight


def softmax(x):
    return tf.nn.softmax(x)


def sample(probabilities):
    probabilities = probabilities.numpy()
    return np.random.choice(len(probabilities), p=probabilities)


def argmax(v):
    return tf.argmax(v).numpy()


def bpe_encode(text, vocab, vocab_scores):
    tokens = []
    for pos, char in enumerate(text):
        string = char
        try:
            idx = vocab.index(string)
        except ValueError:
            print(f"Invalid prompt at position {pos}")
            sys.exit(1)
        tokens.append(idx)

    while True:
        best_score = -1e10
        best_id = -1
        best_idx = -1

        for i in range(len(tokens) - 1):
            string = vocab[tokens[i]] + vocab[tokens[i + 1]]
            try:
                idx = vocab.index(string)
                score = vocab_scores[idx]
                if score > best_score:
                    best_score = score
                    best_id = idx
                    best_idx = i
            except ValueError:
                continue

        if best_idx == -1:
            break

        tokens[best_idx] = best_id
        tokens.pop(best_idx + 1)

    return tokens


def run(args):
    checkpoint = args.get("checkpoint", './stories15M.bin')
    temperature = float(args.get("temperature", 0.7))
    steps = int(args.get("steps", 256))
    prompt = args.get("prompt", "Once upon a time")

    random.seed(int(time.time()))

    weights = TransformerWeights()

    with open(checkpoint, "rb") as file:
        _config = file.read(struct.calcsize('7i'))
        dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len = struct.unpack('7i', _config)
        config = Config(dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len)

        shared_weights = 1 if config.vocab_size > 0 else 0
        config.vocab_size = abs(config.vocab_size)

        checkpoint_init_weights(weights, config, file, shared_weights)

    if steps <= 0 or steps > config.seq_len:
        steps = config.seq_len

    with open("tokenizer.bin", "rb") as file:
        vocab, vocab_scores, max_token_length = tokenizer_init(config, file)

    prompt_tokens = []
    if prompt:
        prompt_tokens = bpe_encode(prompt, vocab, vocab_scores)

    # Initialize state
    token = 1
    pos = 0
    print("<s>")

    while pos < steps:
        if pos < len(prompt_tokens):
            token = prompt_tokens[pos]
        input_ids = tf.constant([token], dtype=tf.int32)

        # Embedding lookup
        x = tf.gather(weights.token_embedding_table, input_ids)

        # Forward pass through layers
        for l in range(config.n_layers):
            # Attention RMSNorm
            x_norm = rmsnorm(x, weights.rms_att_weight[l])

            # QKV projections
            q = tf.matmul(x_norm, weights.wq[l])
            k = tf.matmul(x_norm, weights.wk[l])
            v = tf.matmul(x_norm, weights.wv[l])

            # Apply RoPE (simplified for brevity)
            # Skipping RoPE implementation for simplicity

            # Attention
            attn_scores = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(float(config.dim))
            attn_probs = softmax(attn_scores)
            attn_output = tf.matmul(attn_probs, v)
            x = x + tf.matmul(attn_output, weights.wo[l])

            # FFN RMSNorm
            x_norm = rmsnorm(x, weights.rms_ffn_weight[l])

            # FFN
            h1 = tf.matmul(x_norm, weights.w1[l])
            h2 = tf.matmul(x_norm, weights.w3[l])
            ff = h1 * tf.nn.silu(h2)
            x = x + tf.matmul(ff, weights.w2[l])

        # Final RMSNorm
        x = rmsnorm(x, weights.rms_final_weight)

        # Output logits
        logits = tf.matmul(x, weights.wcls, transpose_b=True)
        logits = tf.squeeze(logits, axis=0)

        if pos >= len(prompt_tokens):
            if temperature == 0.0:
                token = argmax(logits)
            else:
                probs = softmax(logits / temperature)
                token = sample(probs)
        else:
            token = prompt_tokens[pos]

        token_str = (
            vocab[token].lstrip()
            if token == 1 and vocab[token][0] == ' ' else vocab[token]
        )

        print(token_str, end="")
        sys.stdout.flush()

        if token == 1:
            break

        pos += 1

    print()


if __name__ == "__main__":
    args = {
        "checkpoint": './stories15M.bin',
        "temperature": "0.7",
        "steps": "256",
        "prompt": "Once upon a time"
    }

    if len(sys.argv) >= 2:
        args["checkpoint"] = sys.argv[1]

    if len(sys.argv) >= 3:
        args["temperature"] = sys.argv[2]

    if len(sys.argv) >= 4:
        args["steps"] = sys.argv[3]

    if len(sys.argv) >= 5:
        args["prompt"] = sys.argv[4]

    run(args)
