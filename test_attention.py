import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from attention import scaled_dot_product_attention


tokens = ["Os", "pinguins", "não", "tem", "joelho"]


embeddings = np.array([
    [1, 0, 0, 0],   # Os
    [0, 1, 0, 0],   # pinguins
    [0, 0, 1, 0],   # não
    [0, 1, 1, 0],   # tem
    [0, 1, 0, 1],   # joelho
])


Q = embeddings
K = embeddings
V = embeddings

output, attention_weights = scaled_dot_product_attention(
    Q, K, V, return_attention=True
)

print("Embeddings iniciais:")
print(embeddings)

print("\nAttention Weights:")
print(attention_weights)

print("\nOutput final:")
print(output)

plt.figure(figsize=(6,5))
sns.heatmap(attention_weights,
            xticklabels=tokens,
            yticklabels=tokens,
            annot=True,
            cmap="viridis")
plt.title("Self-Attention Weights")
plt.xlabel("Keys")
plt.ylabel("Queries")
plt.tight_layout()
plt.show()
