import jax
import jax.numpy as jnp
import flax.linen as nn
import optax


class LinearEmbedding(nn.Module):
    vocab_size: int
    embedding_dim: int
    max_context_len: int

    @nn.compact
    def __call__(self, x: jax.Array):
        one_hot_tokens = jax.nn.one_hot(x, self.vocab_size)
        one_hot_positions = jax.nn.one_hot(jnp.arange(len(x))[::-1], self.max_context_len)
        token_embedding = nn.Dense(self.embedding_dim)(one_hot_tokens)
        position_embedding = nn.Dense(self.embedding_dim)(one_hot_positions)
        return token_embedding + position_embedding


class MLP(nn.Module):
    hidden_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x: jax.Array):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.output_dim)(x)
        return x


class Attention(nn.Module):
    head_size: int
    n_heads: int = 1

    @nn.compact
    def __call__(self, x):
        if self.n_heads == 1:
            keys = nn.Dense(self.head_size, use_bias=False)(x)
            values = nn.Dense(self.head_size, use_bias=False)(x)
            queries = nn.Dense(self.head_size, use_bias=False)(x)

            causal_mask = jnp.tril(jnp.ones((len(x), len(x))))
            weights: jax.Array = keys @ queries.T / jnp.sqrt(self.head_size)
            weights = jax.nn.softmax(jnp.where(causal_mask == 1, weights, -jnp.inf), axis=-1)
            return weights @ values
        else:
            return jnp.concatenate(
                [Attention(self.head_size)(x) for _ in range(self.n_heads)], axis=-1
            )


class TransformerBlock(nn.Module):
    head_size: int
    n_heads: int

    @nn.compact
    def __call__(self, x):
        embedding_dims = x.shape[-1]
        x = x + nn.Dense(embedding_dims)(Attention(self.head_size, self.n_heads)(x))
        x = x + MLP(4 * embedding_dims, embedding_dims)(x)
        return x


class LanguageModelMixin:
    vocab_size: int
    max_context_len: int

    def __call__(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError

    def logits(self, context: jax.Array) -> jax.Array:
        """Compute logits for next token.

        Args:
            context (context_len,): Sequence of token idxs.

        Returns:
            logits (context_len, vocab_size): Logits for next token.
        """
        return self(context)

    def generate_token(self, context: jax.Array, rng_key: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Generate next token.

        Args:
            context (context_len,): Sequence of token idxs.
            rng_key (PRNGKey): Random key.

        Returns:
            next_token (): Next token.
            updated_context (updated_context_len,): Updated context given the generated token.
        """
        logits = self.logits(context)[-1]
        next_token = jax.random.categorical(rng_key, logits, axis=-1)
        context = self.update_context(context, next_token)
        return next_token, context

    def update_context(self, context: jax.Array, next_token: jax.Array) -> jax.Array:
        """Update context given the generated token, truncated to max_context_len.

        Args:
            context (context_len,): Sequence of token idxs.
            next_token (): Next token.

        Returns:
            updated_context (updated_context_len,): Updated context given the generated token.
        """
        if len(context) < self.max_context_len:
            context = jnp.concatenate([context, next_token[None]])
        else:
            context = context.at[:-1].set(context[1:])
            context = context.at[-1].set(next_token)
        if len(context) > self.max_context_len:
            context = context[-self.max_context_len :]
        return context


class BigramLM(nn.Module, LanguageModelMixin):
    vocab_size: int
    max_context_len: int = 1

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        tokens_one_hot = jax.nn.one_hot(x, self.vocab_size)
        logits = nn.Dense(self.vocab_size)(tokens_one_hot)
        return logits


class TransormerLM(nn.Module, LanguageModelMixin):
    vocab_size: int
    max_context_len: int
    embedding_dim: int
    head_size: int
    n_heads: int = 1
    n_layers: int = 1

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = LinearEmbedding(self.vocab_size, self.embedding_dim, self.max_context_len)(x)
        for _ in range(self.n_layers):
            x = TransformerBlock(self.head_size, self.n_heads)(x)
        x = nn.Dense(self.vocab_size)(x)
        return x
