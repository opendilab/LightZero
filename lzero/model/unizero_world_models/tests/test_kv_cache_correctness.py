"""
Empirical correctness probes for the UniZero world-model KV cache.

These tests validate *behavior* with real tensors rather than code reading:

1. ``test_transformer_cache_concat_equivalence``: a split forward (prefix builds the cache, suffix
   attends through it) must match the single-shot full forward bit-for-bit-ish. This is the core
   "no misalignment" invariant of the cache concat + causal-mask mechanics.
2. ``test_collect_path_cache_key_ownership_and_isolation``: driving the actual collect-time
   ``forward_initial_inference`` path over a synthetic 2-env episode, the per-env init-cache keys
   must equal ``hash_state`` of that env's current latent (ownership), must not appear in other
   envs' dicts (isolation), and continuing steps with unchanged weights must hit (root_hit_cnt).
3. ``test_weight_change_self_invalidates_cache``: after any parameter update, re-encoded latents
   must produce keys disjoint from the stored ones, i.e. stale-weight K/V can never be hit.
4. ``test_trim_and_pad_zero_padding_bias``: quantify how much the attention output of a short-cache
   env shifts when its cache is zero-left-padded to a longer context (the trim_and_pad batching
   behavior), i.e. the magnitude of the inherited zero-logit padding bias.
"""
import numpy as np
import pytest
import torch
from easydict import EasyDict

from lzero.model.unizero_world_models.transformer import Transformer, TransformerConfig
from lzero.model.unizero_world_models.utils import hash_state
from .test_per_sample_is_weights import _build_world_model


def _build_tiny_transformer() -> Transformer:
    cfg = TransformerConfig(
        tokens_per_block=2, max_blocks=8, attention='causal',
        num_layers=2, num_heads=2, embed_dim=64,
        embed_pdrop=0., resid_pdrop=0., attn_pdrop=0.,
        task_embed_option='none', register_token_num=0, register_token_shared=False,
    )
    return Transformer(cfg)


@pytest.mark.unittest
class TestKVCacheCorrectness:

    def test_transformer_cache_concat_equivalence(self):
        torch.manual_seed(0)
        tr = _build_tiny_transformer().eval()
        B, T, C, split = 2, 7, 64, 4
        x = torch.randn(B, T, C)
        with torch.no_grad():
            y_full = tr(x)
            kv = tr.generate_empty_keys_values(n=B, max_tokens=16)
            tr(x[:, :split], past_keys_values=kv)
            y_suffix = tr(x[:, split:], past_keys_values=kv)
        delta = (y_full[:, split:] - y_suffix).abs().max().item()
        assert delta < 1e-5, f'cache split forward diverges from full forward: {delta}'

    def _drive_episode(self, wm, steps, actions, env_ids=(0, 1), perturb=False):
        """Drive `steps` continuing inferences for two envs with distinct synthetic observations."""
        B = len(env_ids)
        rng = np.random.RandomState(0)
        obs_prev = torch.from_numpy(rng.rand(B, 3, 64, 64).astype(np.float32))
        latents = []
        for t in range(steps):
            obs_cur = torch.from_numpy(rng.rand(B, 3, 64, 64).astype(np.float32))
            act = [-1] * B if t == 0 else list(actions)
            obs_act_dict = {
                'obs': obs_prev, 'action': act, 'current_obs': obs_cur,
                'ready_env_id': list(env_ids),
            }
            start_pos = np.zeros(B, dtype=np.int64)
            _, latent, _, _, _ = wm.forward_initial_inference(obs_act_dict, start_pos)
            latents.append(latent.detach().clone())
            obs_prev = obs_cur
        return latents

    def test_collect_path_cache_key_ownership_and_isolation(self):
        torch.manual_seed(0)
        wm = _build_world_model().eval()
        latents = self._drive_episode(wm, steps=3, actions=(2, 3))

        assert wm.root_total_query_cnt > 0, 'continuing steps should query the root cache'
        assert wm.root_hit_cnt > 0, 'continuing steps with unchanged weights must hit the cache'

        for t, latent in enumerate(latents):
            for e in range(2):
                key = hash_state(latent[e].view(-1).cpu().numpy())
                # ownership: the env's own dict contains the key of its latest latent
                assert key in wm.past_kv_cache_init_infer_envs[e], f'env{e} step{t}: key missing'
                # isolation: the key must not leak into another env's dict
                other = 1 - e
                assert key not in wm.past_kv_cache_init_infer_envs[other], f'env{e} step{t}: cross-env leak'

    def test_weight_change_self_invalidates_cache(self):
        torch.manual_seed(0)
        wm = _build_world_model().eval()
        fixed_observation = torch.from_numpy(
            np.random.RandomState(1).rand(2, 3, 64, 64).astype(np.float32)
        )
        with torch.no_grad():
            old_embeddings = wm.tokenizer.encode_to_obs_embeddings(fixed_observation)
        old_keys = {
            hash_state(old_embeddings[index].reshape(-1).cpu().numpy()) for index in range(2)
        }

        # simulate a learner update: perturb every parameter slightly
        with torch.no_grad():
            for p in wm.parameters():
                p.add_(1e-4 * torch.randn_like(p))

        with torch.no_grad():
            new_embeddings = wm.tokenizer.encode_to_obs_embeddings(fixed_observation)
        new_latent_keys = {
            hash_state(new_embeddings[index].reshape(-1).cpu().numpy()) for index in range(2)
        }

        overlap = old_keys & new_latent_keys
        assert not overlap, f'stale cache keys still match after weight update: {len(overlap)} hits'

    def test_trim_and_pad_zero_padding_bias(self):
        """A short cache zero-left-padded to a longer context changes the attention output; measure it.

        trim_and_pad_kv_cache writes *literal zeros* into the leading slots of a shorter env's cache
        to match the longest context in the batch, and the forward pass does not mask them out, so
        those positions enter softmax with logit q@0 = 0 (weight exp(0)=1). Emulate that exactly:
        take an exact 3-token cache, then build a variant whose first 3 slots are literal zeros with
        the real K/V shifted right, and compare the output of the same next-token forward.
        """
        torch.manual_seed(0)
        tr = _build_tiny_transformer().eval()
        B, C = 1, 64
        x_short = torch.randn(B, 3, C)
        x_new = torch.randn(B, 1, C)
        with torch.no_grad():
            kv_exact = tr.generate_empty_keys_values(n=B, max_tokens=16)
            tr(x_short, past_keys_values=kv_exact)  # cache size = 3
            y_exact = tr(x_new, past_keys_values=kv_exact)

            kv_pad = tr.generate_empty_keys_values(n=B, max_tokens=16)
            for layer in range(tr.config.num_layers):
                for kv in ('_k_cache', '_v_cache'):
                    src = getattr(kv_exact._keys_values[layer], kv)
                    dst = getattr(kv_pad._keys_values[layer], kv)
                    dst._cache.zero_()
                    dst._cache[:, :, 3:6, :] = src._cache[:, :, :3, :]  # real K/V shifted right by 3
                    dst._size = 6                                        # leading 3 slots are zeros
            y_pad = tr(x_new, past_keys_values=kv_pad)

        delta = (y_exact - y_pad).norm().item() / y_exact.norm().item()
        assert np.isfinite(delta)
        assert delta > 1e-4, 'unmasked zero-padding should measurably bias attention output'
