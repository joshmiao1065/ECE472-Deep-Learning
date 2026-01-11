import jax
import jax.numpy as jnp
from flax import linen as nnx
import structlog

log = structlog.get_logger() 

class TextEmbeder(nnx.Module):
    vocab_size: int
    embed_dim: int

    @nnx.compact
    def __call__(self, input_ids):
        """
        train model embeddings intead of using pretrained embeddings so i have more control for ass. 7
        nnx.Embed was useful for creating a trainable embedding layer.
        """
        embed = nnx.Embed(num_embeddings=self.vocab_size, features=self.embed_dim, name="token_embed")
        x = embed(input_ids) # (batch_size, seq_length, embed_dim)
        return x
    
class MLPEncoder(nnx.Module):
    latent_dim: int
    hidden_dims: tuple = (512,)
    activation: callable = jax.nn.relu
    dropout_rate: float = 0.0

    @nnx.compact
    def __call__(self,x, deterministic: bool = True):
        """
        Encode pooled features to a latent vector.
        The paper in ass. 7 talks about latent factors so im going to compress the token embeddings
        into a lower dimensional space that i'll call z for now that a decoder can use later.
        I just applied a sequence of dense layers with ReLU activation and dropout the nprojected to
        the latent dimension

        """
        for h_dim in self.hidden_dims:
            x = nnx.Dense(features=h_dim)(x)
            x = self.activation(x)
            x = nnx.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)

        # Final layer to latent space which will be the target of sparsity for ass. 7
        z = nnx.Dense(features=self.latent_dim)(x)
        return z

class ClassifierHead(nnx.Module):
    num_classes: int
    dropout_rate: float = 0.0

    @nnx.compact
    def __call__(self, z, deterministic: bool = True):
        """
        Project latent z to logits
        """
        x = nnx.Dropout(rate=self.dropout_rate)(z, deterministic=deterministic)
        logits = nnx.Dense(features=self.num_classes)(x)
        return logits
    
class TextMLPModel(nnx.Module):
    vocab_size: int
    embed_dim: int
    latent_dim: int
    num_classes: int
    hidden_dims: tuple = (512,)
    dropout_rate: float = 0.0

    def setup(self):
        self.embedder = TextEmbeder(vocab_size=self.vocab_size, embed_dim=self.embed_dim)
        self.encoder = MLPEncoder(latent_dim=self.latent_dim, hidden_dims=self.hidden_dims, dropout_rate=self.dropout_rate)
        self.classifier = ClassifierHead(num_classes=self.num_classes, dropout_rate=self.dropout_rate)
    def __call__(self, input_ids, attention_mask=None, deterministic: bool = True):
        """
        Forward pass for the text MLP classifier. (batch, seq_len) input, logits is (batch, num_classes) and z is (batch, latent_dim)
        Embed tokens, mean pooling(i used max pooling for hw04 and that was ass(like pooling was the issue lol)), encode latent vector, create logit
        return a TWOple. get it?? because i have 2 items in my tuple?
        """
        x = self.embedder(input_ids)  # (batch, seq_len, embed_dim)

        # Pool token embeddings into a fixed-size vector per example.
        if attention_mask is None:
            # simple mean-pooling across sequence length
            pooled = jnp.mean(x, axis=1)
        else:
            # use attention mask for masked mean pooling
            mask = jnp.expand_dims(jnp.asarray(attention_mask, dtype=x.dtype), axis=-1)
            x_masked = x * mask
            summed = jnp.sum(x_masked, axis=1)
            denom = jnp.sum(mask, axis=1)
            pooled = summed / jnp.clip(denom, a_min=1e-9)

        # encoder to latent space
        z = self.encoder(pooled, deterministic=deterministic)
        # classifier to logits
        logits = self.classifier(z)
        return logits, z