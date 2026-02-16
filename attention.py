import numpy as np

def scaled_dot_product_attention(Q, K, V, return_attention=False):

   # Produto escalar
    scores = np.dot(Q, K.T)

    # Scaling
    dk = K.shape[1]
    scaled_scores = scores / np.sqrt(dk)

    # Softmax numericamente estável
    max_scores = np.max(scaled_scores, axis=1, keepdims=True)
    exp_scores = np.exp(scaled_scores - max_scores)
    attention_weights = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Combinação ponderada
    output = np.dot(attention_weights, V)

    if return_attention:
        return output, attention_weights

    return output
