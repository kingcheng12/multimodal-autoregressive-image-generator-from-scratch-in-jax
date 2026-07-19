"""
Multimodal Autoregressive Image Generator from Scratch in JAX

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - generate_toy_images
def generate_toy_images(key, num_images, image_size):
    # TODO: return num_images grayscale images, each a bright square on a zero background
    square_size = image_size // 2

    key_row, key_col = jax.random.split(key)

    rows = jax.random.randint(
        key_row,
        shape=(num_images,),
        minval=0,
        maxval=image_size - square_size + 1,
    )
    cols = jax.random.randint(
        key_col,
        shape=(num_images,),
        minval=0,
        maxval=image_size - square_size + 1,
    )

    y = jnp.arange(image_size)[None, :, None]
    x = jnp.arange(image_size)[None, None, :]

    row_start = rows[:, None, None]
    col_start = cols[:, None, None]

    images = (
        (y >= row_start)
        & (y < row_start + square_size)
        & (x >= col_start)
        & (x < col_start + square_size)
    )

    return images.astype(jnp.float32)

# Step 2 - assign_image_labels
def assign_image_labels(images):
    # TODO: label each image 'left' or 'right' by comparing left vs right pixel mass

    images = jnp.asarray(images)

    width = images.shape[-1]
    midpoint = width // 2

    left_mass = images[:, :, :midpoint].sum(axis=(1, 2))
    right_mass = images[:, :, -midpoint:].sum(axis=(1, 2))

    # JAX arrays do not support string elements, so return a Python list.
    return np.where(
        np.asarray(left_mass >= right_mass),
        "left",
        "right",
    ).tolist()

# Step 3 - normalize_image_batch
def normalize_image_batch(images):
    # TODO: rescale images from [0, 1] into the symmetric [-1, 1] range
    images = jnp.asarray(images)

    return 2.0 * images - 1.0

# Step 4 - split_image_into_patches
def split_image_into_patches(image, patch_size):
    # TODO: Split a single (H, W) image into a grid of non-overlapping square patches.
    image = jnp.asarray(image)

    height, width = image.shape
    num_patch_rows = height // patch_size
    num_patch_cols = width // patch_size

    patches = image.reshape(
        num_patch_rows,
        patch_size,
        num_patch_cols,
        patch_size,
    )
    
    return patches.transpose(0, 2, 1, 3)

# Step 5 - flatten_patches
def flatten_patches(patches):
    # TODO: flatten each (p, p) patch in a (gh, gw, p, p) grid into a 1D pixel vector
    patches = jnp.asarray(patches)
    gh, gw, ph, pw = patches.shape

    return patches.reshape((gh*gw, ph*pw))

# Step 6 - init_patch_encoder
def init_patch_encoder(key, patch_dim, latent_dim):
    # TODO: return a (patch_dim, latent_dim) scaled Gaussian weight matrix from key
    scale = 1.0 / jnp.sqrt(patch_dim)

    return scale * jax.random.normal(key, shape=(patch_dim, latent_dim))

# Step 7 - encode_patches
def encode_patches(flat_patches, encoder_weight):
    # TODO: Project each flattened patch to a latent vector with the linear encoder.
    flat_patches = jnp.asarray(flat_patches)
    encoder_weight = jnp.asarray(encoder_weight)

    return flat_patches @ encoder_weight

# Step 8 - init_patch_decoder
def init_patch_decoder(key, latent_dim, patch_dim):
    # TODO: sample a (latent_dim, patch_dim) weight from key, scaled by 1/sqrt(latent_dim)

    weight = jax.random.normal(key, shape=(latent_dim, patch_dim))

    return weight / jnp.sqrt(latent_dim)

# Step 9 - decode_latents
def decode_latents(latents, decoder_weight):
    # TODO: project each latent vector back to flat patch pixels with the linear decoder
    return latents @ decoder_weight

# Step 10 - reassemble_patches_into_image
def reassemble_patches_into_image(flat_patches, grid_h, grid_w, patch_size):
    # TODO: reshape each flat patch to a square and tile them into the full image
    
    flat_patches = jnp.asarray(flat_patches)
    expected_num_patches = grid_h * grid_w
    expected_patch_dim = patch_size * patch_size

    # recover (grid_h, grid_w, patch_h, patch_w)
    patch_grid = flat_patches.reshape(
        grid_h,
        grid_w,
        patch_size,
        patch_size,
    )

    # recover image
    image = patch_grid.transpose(0, 2, 1, 3).reshape(
        grid_h * patch_size,
        grid_w * patch_size,
    )

    return image

# Step 11 - init_codebook
def init_codebook(key, num_codes, latent_dim):
    # TODO: draw a (num_codes, latent_dim) table of small random values from key
    
    tbl = jax.random.normal(key, shape = (num_codes, latent_dim))
    std = 0.1

    return tbl * std

# Step 12 - squared_distance_to_codebook
def squared_distance_to_codebook(latent, codebook):
    # TODO: squared Euclidean distance from one latent vector to every codebook vector
    diff = codebook - latent

    sqrt_dist = jnp.sum(diff**2, axis=-1)

    return sqrt_dist

# Step 13 - grid_distances_to_codebook
def grid_distances_to_codebook(latents, codebook):
    # TODO: squared distance from each latent (P, D) to each code (K, D) -> (P, K)
    
    latents = jnp.asarray(latents)
    codebook = jnp.asarray(codebook)

    differences = latents[:, None, :] - codebook[None, :, :]
    return jnp.sum(differences ** 2, axis=-1)

# Step 14 - assign_nearest_codes
import jax
import jax.numpy as jnp

def assign_nearest_codes(distances):
    # TODO: return the nearest codebook index for each latent via argmin over codes
    
    # distance: num_patches, num_codes

    return jnp.argmin(distances, axis = -1)

# Step 15 - lookup_codebook_vectors
def lookup_codebook_vectors(indices, codebook):
    # TODO: return the codebook row for each token index, shape (num_patches, latent_dim)
    return codebook[indices]

# Step 16 - straight_through_quantize
def straight_through_quantize(latents, quantized):
    # TODO: return quantized in value but with latents' gradient (straight-through)

    latents = jnp.asarray(latents)
    quantized = jnp.asarray(quantized)

    # stop_gradient(x) returns the same value as x during the forward pass but treat it as constant during auto-differentiation
    return latents + jax.lax.stop_gradient(quantized - latents)

# Step 17 - codebook_loss
def codebook_loss(latents, quantized):
    # TODO: mean squared error between stop_gradient(latents) and quantized
    target_latents = jax.lax.stop_gradient(latents)
    mse = jnp.mean((target_latents - quantized) ** 2)

    return mse

# Step 18 - commitment_loss
def commitment_loss(latents, quantized):
    # TODO: mean squared error between latents and stop_gradient(quantized)
    
    target_quantized = jax.lax.stop_gradient(quantized)
    mse = jnp.mean((latents - target_quantized) ** 2)

    return mse

# Step 19 - reconstruction_loss
def reconstruction_loss(image, reconstruction):
    # TODO: return the mean squared error between image and reconstruction
    
    return jnp.mean((image - reconstruction)**2)

# Step 20 - total_vqvae_loss
def total_vqvae_loss(recon_loss, cb_loss, commit_loss, commitment_weight):
    # TODO: return recon_loss + cb_loss + commitment_weight * commit_loss as a scalar
    
    return recon_loss + cb_loss + commitment_weight * commit_loss

# Step 21 - vqvae_loss_and_grads
def vqvae_loss_and_grads(params, image_batch, patch_size, commitment_weight):
    # TODO: Compute the VQ-VAE total loss and gradients wrt encoder/decoder/codebook over a batch of images.
    
    def loss_fn(current_params):
        encoder = current_params["encoder"]
        decoder = current_params["decoder"]
        codebook = current_params["codebook"]

        def process_image(image):
            # image: (H, W)
            patches = split_image_into_patches(image, patch_size)

            # patches: (grid_h, grid_w, patch_size, patch_size)
            grid_h, grid_w = patches.shape[:2]

            flat_patches = flatten_patches(patches)

            latents = encode_patches(flat_patches, encoder)

            distances = grid_distances_to_codebook(
                latents,
                codebook,
            )

            code_indices = assign_nearest_codes(distances)

            quantized = lookup_codebook_vectors(
                code_indices,
                codebook,
            )

            straight_through = straight_through_quantize(
                latents,
                quantized,
            )

            decoded_patches = decode_latents(
                straight_through,
                decoder,
            )

            reconstruction = reassemble_patches_into_image(
                decoded_patches,
                grid_h,
                grid_w,
                patch_size,
            )

            return reconstruction, latents, quantized

        reconstructions, latents, quantized = jax.vmap(
            process_image
        )(image_batch)

        recon_loss = reconstruction_loss(
            image_batch,
            reconstructions,
        )

        cb_loss = codebook_loss(
            latents,
            quantized,
        )

        commit_loss = commitment_loss(
            latents,
            quantized,
        )

        return total_vqvae_loss(
            recon_loss,
            cb_loss,
            commit_loss,
            commitment_weight,
        )

    return jax.value_and_grad(loss_fn)(params)

# Step 22 - apply_vqvae_update
def apply_vqvae_update(params, grads, opt_state, optimizer):
    # TODO: Apply one optax update to the VQ-VAE params and return new params + opt state.
    
    updates, new_opt_state = optimizer.update(
        grads,
        opt_state,
        params,
    )

    # Apply the updates to the parameter pytree.
    new_params = optax.apply_updates(params, updates)

    return new_params, new_opt_state

# Step 23 - encode_image_to_tokens
def encode_image_to_tokens(image, params, patch_size):
    # TODO: split, encode, quantize, and reshape patch codes into a token grid
    encoder = params["encoder"]
    codebook = params["codebook"]

    patches = split_image_into_patches(image, patch_size)
    grid_h, grid_w = patches.shape[:2]

    flat_patches = flatten_patches(patches)
    latents = encode_patches(flat_patches, encoder)
    distances = grid_distances_to_codebook(latents, codebook)
    token_indices = assign_nearest_codes(distances)
    token_grid = token_indices.reshape(grid_h, grid_w)

    return token_grid

# Step 24 - flatten_token_grid
def flatten_token_grid(token_grid):
    # TODO: Flatten a (grid_h, grid_w) token grid into a 1D sequence in row-major order.
    
    return token_grid.reshape(-1)

# Step 25 - reshape_tokens_to_grid
def reshape_tokens_to_grid(token_sequence, grid_h, grid_w):
    # TODO: reshape a 1D token sequence back into a 2D (grid_h, grid_w) grid
    return token_sequence.reshape((grid_h, grid_w))

# Step 26 - build_char_vocab
def build_char_vocab(labels):
    # TODO: map each unique character across all labels to a deterministic integer id
    chars = set()
    for label in labels:
        for char in label:
            chars.add(char)

    return {labels:idx for idx, labels in enumerate(sorted(chars))}

# Step 27 - encode_label_to_ids
def encode_label_to_ids(label, char_vocab):
    # TODO: map each character of label to its id and return a jnp int array
    chars = list(label)
    arr = jnp.asarray([char_vocab[char] for char in chars])

    return arr

# Step 28 - form_multimodal_sequence
def form_multimodal_sequence(text_ids, image_tokens, image_token_offset):
    # TODO: prepend text_ids before image_tokens shifted by image_token_offset
    
    return jnp.concatenate([text_ids, image_tokens+image_token_offset])

# Step 29 - init_token_embedding
def init_token_embedding(key, vocab_size, embed_dim):
    # TODO: build a small randomly initialized (vocab_size, embed_dim) embedding table
    scale = 0.02

    return scale * jax.random.normal(key, shape=(vocab_size, embed_dim))

# Step 30 - init_positional_embedding
def init_positional_embedding(key, max_seq_len, embed_dim):
    # TODO: sample a small-magnitude (max_seq_len, embed_dim) table from the PRNG key
    scale = 0.02

    return scale * jax.random.normal(key, shape=(max_seq_len, embed_dim))

# Step 31 - lookup_token_embeddings
def lookup_token_embeddings(token_embedding, token_ids):
    # TODO: Select the embedding row for each id to get (seq_len, d_model).
    
    return token_embedding[token_ids]

# Step 32 - add_positional_embeddings
def add_positional_embeddings(token_embeds, positional_embedding):
    # TODO: add the first seq_len positional rows to the token embeddings
    seq_len = len(token_embeds)
    return token_embeds + positional_embedding[:seq_len]

# Step 33 - build_causal_mask
def build_causal_mask(seq_len):
    # TODO: return (seq_len, seq_len) additive mask: 0.0 where j<=i else -1e9
    
    future_pos = jnp.ones((seq_len, seq_len))
    future_pos = jnp.triu(future_pos, k = 1)

    return jnp.where(future_pos, -1e9, 0)

# Step 34 - layer_norm
def layer_norm(x, scale, shift, eps=1e-5):
    # TODO: standardize x over its last axis, then apply learned scale and shift

    x = jnp.asarray(x)
    scale = jnp.asarray(scale)
    shift = jnp.asarray(shift)

    mean = jnp.mean(x, axis=-1, keepdims=True)
    variance = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    
    normalized = (x - mean) / jnp.sqrt(variance + eps)

    return normalized * scale + shift

# Step 35 - init_attention_params
def init_attention_params(key, d_model):
    # TODO: return dict with 'wq','wk','wv','wo', each (d_model, d_model) small random

    keys = jax.random.split(key, 4)
    scale = 0.02

    return {
        "wq": scale * jax.random.normal(
            keys[0], (d_model, d_model), dtype=jnp.float32
        ),
        "wk": scale * jax.random.normal(
            keys[1], (d_model, d_model), dtype=jnp.float32
        ),
        "wv": scale * jax.random.normal(
            keys[2], (d_model, d_model), dtype=jnp.float32
        ),
        "wo": scale * jax.random.normal(
            keys[3], (d_model, d_model), dtype=jnp.float32
        ),
    }

# Step 36 - project_qkv
def project_qkv(x, attn_params):
    # TODO: Project x into query, key, and value matrices with wq, wk, wv.
    q = x @ attn_params['wq']
    k = x @ attn_params['wk']
    v = x @ attn_params['wv']

    return q, k, v

# Step 37 - reshape_to_heads
def reshape_to_heads(matrix, num_heads):
    # TODO: split the (seq_len, d_model) projection into num_heads attention heads
    seq_len, d_model = matrix.shape

    d_head = d_model // num_heads

    return matrix.reshape(seq_len, num_heads, d_head).transpose(1, 0, 2)

# Step 38 - scaled_dot_product_scores
def scaled_dot_product_scores(q_heads, k_heads):
    # TODO: compute scaled dot-product attention scores between query and key heads
    d_head = q_heads.shape[-1]
    return q_heads @ jnp.swapaxes(k_heads, -1, -2) / jnp.sqrt(d_head)

# Step 39 - add_causal_mask_to_scores
def add_causal_mask_to_scores(scores, causal_mask):
    # TODO: broadcast-add the (seq_len, seq_len) mask onto (num_heads, seq_len, seq_len) scores
    
    scores = jnp.asarray(scores)
    causal_mask = jnp.asarray(causal_mask)

    return scores + causal_mask[None, :, :]

# Step 40 - attention_weights_softmax
def attention_weights_softmax(masked_scores):
    # TODO: numerically stable softmax over the last (key) axis of masked_scores
    masked_scores = masked_scores - jnp.max(masked_scores, axis=-1, keepdims=True)

    exp_scores = jnp.exp(masked_scores)

    return exp_scores / jnp.sum(exp_scores, axis=-1, keepdims=True)

# Step 41 - weighted_sum_of_values
def weighted_sum_of_values(attn_weights, v_heads):
    # TODO: per head, combine value vectors using the attention weights...

    return attn_weights @ v_heads

# Step 42 - merge_heads_and_project
def merge_heads_and_project(head_outputs, attn_params):
    # TODO: concatenate per-head outputs into d_model and apply the wo projection
    wo = attn_params["wo"]
    num_heads, seq_len, d_head = head_outputs.shape
    d_model = num_heads * d_head

    # head_outputs (num_heads, seq_len, d_head)
    merged = head_outputs.transpose(1, 0, 2).reshape(
        seq_len,
        d_model,
    )
    
    return merged @ wo

# Step 43 - init_feedforward_params
def init_feedforward_params(key, d_model, d_ff):
    # TODO: return dict with 'w1' (d_model, d_ff) and 'w2' (d_ff, d_model), small random
    
    w1_shape = (d_model, d_ff)
    w2_shape = (d_ff, d_model)
    scale = 0.02

    key1, key2 = jax.random.split(key)

    w1 = scale * jax.random.normal(
        key1,
        shape=w1_shape,
        dtype=jnp.float32,
    )

    w2 = scale * jax.random.normal(
        key2,
        shape=w2_shape,
        dtype=jnp.float32,
    )

    return {
        "w1": w1,
        "w2": w2,
    }

# Step 44 - feedforward_mlp
def feedforward_mlp(x, ff_params):
    # TODO: expand with w1, apply GELU, then project back with w2

    w1 = ff_params['w1']
    w2 = ff_params['w2']

    z1 = x @ w1

    a1 = jax.nn.gelu(z1)

    z2 = a1 @ w2

    return z2

# Step 45 - transformer_block
def transformer_block(x, block_params, causal_mask, num_heads):
    # TODO: pre-norm attention with a residual, then pre-norm MLP with a residual

    # layer norm
    x_norm = layer_norm(
        x,
        block_params["ln1_scale"],
        block_params["ln1_shift"],
    )

    # mh causal attn
    q, k, v = project_qkv(
        x_norm,
        block_params["attn"],
    )
    q_heads = reshape_to_heads(q, num_heads)
    k_heads = reshape_to_heads(k, num_heads)
    v_heads = reshape_to_heads(v, num_heads)

    scores = scaled_dot_product_scores(q_heads, k_heads)

    masked_scores = add_causal_mask_to_scores(
        scores,
        causal_mask,
    )

    attn_weights = attention_weights_softmax(masked_scores)
    head_outputs = weighted_sum_of_values(
        attn_weights,
        v_heads,
    )
    attn_output = merge_heads_and_project(
        head_outputs,
        block_params["attn"],
    )
    # skip connection
    x = x + attn_output

    # 2nd layer norm
    x_norm = layer_norm(
        x,
        block_params["ln2_scale"],
        block_params["ln2_shift"],
    )

    ff_output = feedforward_mlp(
        x_norm,
        block_params["ff"],
    )

    # skip connection
    x = x + ff_output

    return x

# Step 46 - transformer_backbone
def transformer_backbone(x, blocks_params, causal_mask, num_heads):
    # TODO: Apply each transformer block in sequence to the hidden states.
    
    hidden_states = x

    for block_params in blocks_params:
        hidden_states = transformer_block(
            hidden_states,
            block_params,
            causal_mask,
            num_heads,
        )

    return hidden_states

# Step 47 - init_output_projection
def init_output_projection(key, d_model, vocab_size):
    # TODO: build dict with 'w_out' (d_model, vocab_size) and 'b_out' (vocab_size,)
    scale = 0.02

    w_out = scale * jax.random.normal(
        key,
        shape=(d_model, vocab_size),
        dtype=jnp.float32,
    )

    b_out = jnp.zeros((vocab_size, ))

    return {
        'w_out': w_out,
        'b_out': b_out,
    }

# Step 48 - project_to_logits
def project_to_logits(hidden_states, output_params):
    # TODO: map each (d_model,) hidden vector to (vocab_size,) logits via a linear layer

    w_out = output_params['w_out']
    b_out = output_params['b_out']

    return hidden_states @ w_out + b_out

# Step 49 - image_token_cross_entropy
def image_token_cross_entropy(logits, target_ids, image_start_index):
    # TODO: mean next-token cross entropy over image-token positions only
    
    logits = jnp.asarray(logits)
    target_ids = jnp.asarray(target_ids)

    image_logits = logits[image_start_index - 1 : -1]
    image_targets = target_ids[image_start_index:]

    log_probs = jax.nn.log_softmax(image_logits, axis=-1)

    # jnp.take_along_axis(array, indices, axis)
    # in array, over axis, select indices
    target_log_probs = jnp.take_along_axis(
        log_probs,
        image_targets[:, None],
        axis=-1,
    ).squeeze(-1)

    return -jnp.mean(target_log_probs)

# Step 50 - transformer_loss_and_grads
from jax import config

config.update("jax_enable_x64", True)

def transformer_model_forward(
    params,
    token_ids,
    causal_mask,
    num_heads,
):
    """Convert token IDs into next-token logits."""
    # token_ids: (seq_len,)

    token_embeds = lookup_token_embeddings(
        params["token_embedding"],
        token_ids,
    )
    # (seq_len, d_model)

    hidden_states = add_positional_embeddings(
        token_embeds,
        params["positional_embedding"],
    )
    # (seq_len, d_model)

    hidden_states = transformer_backbone(
        hidden_states,
        params["blocks"],
        causal_mask,
        num_heads,
    )
    # (seq_len, d_model)

    logits = project_to_logits(hidden_states, params['output'])

    return logits

def transformer_loss_and_grads(params, batch_sequences, causal_mask, num_heads, image_start_index):
    # TODO: average per-sequence cross entropy then take value_and_grad over params
    
    def loss_fn(current_params):
        def sequence_loss(sequence):
            logits = transformer_model_forward(
                current_params,
                sequence,
                causal_mask,
                num_heads,
            )

            return image_token_cross_entropy(
                logits,
                sequence,
                image_start_index,
            )

        per_sequence_losses = jax.vmap(sequence_loss)(batch_sequences)
        return jnp.mean(per_sequence_losses)

    loss, grads = jax.value_and_grad(loss_fn)(params)
    return loss, grads

# Step 51 - apply_transformer_update
def apply_transformer_update(params, grads, opt_state, optimizer):
    # TODO: apply one optax update and return (new_params, new_opt_state)

    updates, opt_state = optimizer.update(grads, opt_state, params)

    new_params = optax.apply_updates(params, updates)

    return new_params, opt_state

# Step 52 - drop_text_prefix
def drop_text_prefix(sequence, key, image_start_index, drop_prob, null_token_id):
    # TODO: with prob drop_prob, replace text-prefix positions with null_token_id
    sequence = jnp.asarray(sequence)

    should_drop = jax.random.bernoulli(
        key,
        p=drop_prob,
    )

    positions = jnp.arange(sequence.shape[0])
    text_mask = positions < image_start_index

    mask = should_drop & text_mask

    out = jnp.where(mask, null_token_id, sequence)

    return jnp.asarray(out, dtype=jnp.int32)

# Step 53 - combine_guided_logits
def combine_guided_logits(cond_logits, uncond_logits, guidance_scale):
    # TODO: return uncond + guidance_scale * (cond - uncond) for classifier-free guidance

    return uncond_logits + guidance_scale * (cond_logits - uncond_logits)

# Step 54 - logits_to_probabilities
def logits_to_probabilities(logits, temperature):
    # TODO: scale logits by temperature, then apply a stable softmax
    
    # add temperature to all logits
    shift_logits = logits / temperature

    # stable softmax
    return attention_weights_softmax(shift_logits)

# Step 55 - top_k_filter_logits
def top_k_filter_logits(logits, k):
    # TODO: keep the k largest logits and set the rest to -1e9
    
    _, top_k_indices = jax.lax.top_k(logits, k)

    # (..., k, vocab_size)
    selected = jax.nn.one_hot(
        top_k_indices,
        num_classes=logits.shape[-1],
        dtype=jnp.bool_,
    )

    # Combine the k one-hot masks.
    mask = jnp.any(selected, axis=-2)

    return jnp.where(mask, logits, -1e9)

# Step 56 - sample_token_index
def sample_token_index(probabilities, key):
    # TODO: Sample one token id from a probability distribution using a PRNG key.

    token_ids = jnp.arange(probabilities.shape[0])

    return jax.random.choice(
        key,
        token_ids,
        shape=(),
        p=probabilities,
    )

# Step 57 - generate_image_tokens
def generate_image_tokens(params, text_prefix, key, num_image_tokens, num_heads, null_prefix, guidance_scale, temperature, top_k):
    # TODO: autoregressively sample image tokens with classifier-free guidance
    text_prefix = jnp.asarray(text_prefix, dtype=jnp.int32)
    null_prefix = jnp.asarray(null_prefix, dtype=jnp.int32)

    # Initialize two autoregressive sequences
    conditional_sequence = text_prefix
    unconditional_sequence = null_prefix

    generated_tokens = []

    def get_next_token_logits(sequence):
        # Build embeddings for the current text-and-image sequence.
        hidden_states = lookup_token_embeddings(
            params["token_embedding"],
            sequence,
        )

        hidden_states = add_positional_embeddings(
            hidden_states,
            params["positional_embedding"],
        )

        # Each iteration has a longer sequence, so construct a causal mask
        # matching the current sequence length.
        causal_mask = build_causal_mask(sequence.shape[0])

        hidden_states = transformer_backbone(
            hidden_states,
            params["blocks"],
            causal_mask,
            num_heads,
        )

        logits = project_to_logits(
            hidden_states,
            params["output"],
        )

        # The final position predicts the next token.
        return logits[-1]

    for _ in range(num_image_tokens):
        # Prediction conditioned on the real text prefix.
        conditional_logits = get_next_token_logits(
            conditional_sequence
        )

        # Prediction conditioned on the null prefix.
        unconditional_logits = get_next_token_logits(
            unconditional_sequence
        )

        guided_logits = (
            unconditional_logits
            + guidance_scale
            * (conditional_logits - unconditional_logits)
        )

        # Keep only the top-k candidate tokens.
        filtered_logits = top_k_filter_logits(
            guided_logits,
            top_k,
        )

        # Apply temperature and stable softmax.
        probabilities = logits_to_probabilities(
            filtered_logits,
            temperature,
        )

        # Split the key so every generation step uses a fresh subkey.
        key, sample_key = jax.random.split(key)

        next_token = sample_token_index(
            probabilities,
            sample_key,
        ).astype(jnp.int32)

        generated_tokens.append(next_token)

        # Both branches share the same generated image-token history.
        next_token_array = next_token[None]

        conditional_sequence = jnp.concatenate(
            [conditional_sequence, next_token_array]
        )

        unconditional_sequence = jnp.concatenate(
            [unconditional_sequence, next_token_array]
        )
        
    return jnp.stack(generated_tokens)

# Step 58 - decode_tokens_to_image
def decode_tokens_to_image(image_tokens, codebook, decoder_params, grid_size, patch_size):
    # TODO: grid the tokens, look up codebook latents, decode, and reassemble patches

    # fold seq to 2d
    token_grid = reshape_tokens_to_grid(image_tokens, grid_size, grid_size)

    # look up latent vectors
    latent_grid = lookup_codebook_vectors(token_grid, codebook)

    # linear decoder
    flat_patches = decode_latents(latent_grid, decoder_params['decoder'])

    # decoded patch to full image
    img = reassemble_patches_into_image(flat_patches, grid_size, grid_size, patch_size)

    return img

# Step 59 - next_token_accuracy
def next_token_accuracy(params, batch_sequences, causal_mask, num_heads, image_start_index):
    # TODO: run the transformer forward pass and score argmax predictions on image positions
    def sequence_accuracy(sequence):
        hidden_states = lookup_token_embeddings(
            params["token_embedding"],
            sequence,
        )
        hidden_states = add_positional_embeddings(
            hidden_states,
            params["positional_embedding"],
        )

        causal_mask = build_causal_mask(sequence.shape[0])

        hidden_states = transformer_backbone(
            hidden_states,
            params["blocks"],
            causal_mask,
            num_heads,
        )

        logits = project_to_logits(
            hidden_states,
            params["output"],
        )

        image_logits = logits[image_start_index - 1 : -1]
        image_targets = sequence[image_start_index:]

        predictions = jnp.argmax(image_logits, axis=-1)

        return jnp.mean(predictions == image_targets)

    per_sequence_accuracy = jax.vmap(sequence_accuracy)(
        batch_sequences
    )

    return jnp.mean(per_sequence_accuracy)

# Step 60 - average_reconstruction_error
def average_reconstruction_error(encoder_params, decoder_params, codebook, image_batch, patch_size):
    # TODO: encode, quantize, decode each image and average the squared reconstruction error

    norm_image = image = normalize_image_batch(image_batch)

    def image_reconstruction(image):

        patches = split_image_into_patches(image, patch_size)

        grid_h, grid_w = patches.shape[:2]

        flat_patches = flatten_patches(patches)

        latents = encode_patches(flat_patches, encoder_params['weight'])

        distances = grid_distances_to_codebook(
            latents,
            codebook,
        )

        code_indices = assign_nearest_codes(distances)

        quantized = lookup_codebook_vectors(
            code_indices,
            codebook,
        )

        straight_through = straight_through_quantize(
            latents,
            quantized,
        )

        decoded_patches = decode_latents(
            straight_through,
            decoder_params['weight'],
        )

        reconstruction = reassemble_patches_into_image(
            decoded_patches,
            grid_h,
            grid_w,
            patch_size,
        )

        return reconstruction

    per_seq_recon = jax.vmap(image_reconstruction)(norm_image)

    recon_loss = reconstruction_loss(
        norm_image,
        per_seq_recon,
    )

    return recon_loss

# Step 61 - nearest_neighbor_distance_to_dataset
def nearest_neighbor_distance_to_dataset(generated_image, dataset_images):
    # TODO: return the min squared Euclidean distance to any dataset image

    differences = dataset_images - generated_image

    # Sum across every image dimension, but not the dataset dimension.
    image_axes = tuple(range(1, differences.ndim))
    squared_distances = jnp.sum(
        differences ** 2,
        axis=image_axes,
    )

    return jnp.min(squared_distances)

# Step 62 - train_vqvae_on_toy_images (not yet solved)
# TODO: implement

# Step 63 - train_transformer_on_token_sequences (not yet solved)
# TODO: implement

# Step 64 - generate_image_from_label (not yet solved)
# TODO: implement

