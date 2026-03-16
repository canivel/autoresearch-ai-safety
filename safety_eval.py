"""
Safety-oriented evaluation metrics for autoresearch-safety.

These metrics measure properties that are *relevant* to safety at the
representation and prediction level. They complement the primary val_bpb
metric from prepare.py.

Usage:
    from safety_eval import SafetyMetrics
    metrics = SafetyMetrics(model, tokenizer, device_batch_size)
    results = metrics.evaluate_all()
"""

import math
import torch
import torch.nn.functional as F
from prepare import MAX_SEQ_LEN, Tokenizer, make_dataloader


class SafetyMetrics:
    """Evaluate safety-relevant properties of a trained language model."""

    def __init__(self, model, tokenizer, batch_size, num_batches=10):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_batches = num_batches

    @torch.no_grad()
    def representation_diversity(self):
        """
        Measure how diverse the model's internal representations are.

        High diversity = representations use the full embedding space rather
        than collapsing into a low-dimensional subspace. This correlates with
        interpretability and steerability.

        Returns:
            dict with:
              - cos_sim_mean: mean pairwise cosine similarity (lower = more diverse)
              - effective_rank: approximate rank of the representation matrix
                (higher = more diverse, max = embedding_dim)
        """
        val_loader = make_dataloader(self.tokenizer, self.batch_size, MAX_SEQ_LEN, "val")
        all_cos_sims = []
        all_singular_values = []

        for _ in range(self.num_batches):
            x, _, _ = next(val_loader)

            # Get final-layer representations (before lm_head)
            # We need to run the model manually to extract intermediate states
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                B, T = x.size()
                cos_sin = self.model.cos[:, :T], self.model.sin[:, :T]

                h = self.model.transformer.wte(x)
                h = F.rms_norm(h, (h.size(-1),))
                h0 = h
                for i, block in enumerate(self.model.transformer.h):
                    h = self.model.resid_lambdas[i] * h + self.model.x0_lambdas[i] * h0
                    ve_key = str(i)
                    ve = self.model.value_embeds[ve_key](x) if ve_key in self.model.value_embeds else None
                    h = block(h, ve, cos_sin, self.model.window_sizes[i])
                h = F.rms_norm(h, (h.size(-1),))

            # Sample representations for efficiency
            h_flat = h.float().view(-1, h.size(-1))
            n_sample = min(512, h_flat.size(0))
            indices = torch.randperm(h_flat.size(0), device=h_flat.device)[:n_sample]
            h_sample = h_flat[indices]

            # Cosine similarity
            h_normed = F.normalize(h_sample, dim=-1)
            sim_matrix = h_normed @ h_normed.T
            mask = ~torch.eye(n_sample, device=sim_matrix.device, dtype=torch.bool)
            all_cos_sims.append(sim_matrix[mask].mean().item())

            # Singular values for effective rank
            _, s, _ = torch.svd(h_sample - h_sample.mean(dim=0, keepdim=True))
            all_singular_values.append(s.cpu())

        cos_sim_mean = sum(all_cos_sims) / len(all_cos_sims)

        # Effective rank: exp(entropy of normalized singular values)
        avg_sv = torch.stack(all_singular_values).mean(dim=0)
        p = avg_sv / avg_sv.sum()
        p = p[p > 1e-10]  # filter near-zero
        entropy = -(p * p.log()).sum().item()
        effective_rank = math.exp(entropy)

        return {
            "cos_sim_mean": cos_sim_mean,
            "effective_rank": effective_rank,
        }

    @torch.no_grad()
    def prediction_calibration(self):
        """
        Measure how well-calibrated the model's predictions are.

        A well-calibrated model assigns probability p to events that happen
        ~p fraction of the time. Overconfident models are harder to align
        because they don't distinguish knowledge from guessing.

        Returns:
            dict with:
              - mean_entropy: average prediction entropy (higher = less overconfident)
              - top1_confidence: average probability of top prediction
              - ece: expected calibration error (lower = better calibrated)
        """
        val_loader = make_dataloader(self.tokenizer, self.batch_size, MAX_SEQ_LEN, "val")
        all_entropies = []
        all_top1_conf = []

        # For ECE: bin predictions by confidence and track accuracy
        n_bins = 10
        bin_correct = torch.zeros(n_bins)
        bin_confidence = torch.zeros(n_bins)
        bin_count = torch.zeros(n_bins)

        for _ in range(self.num_batches):
            x, y, _ = next(val_loader)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(x)  # no targets = return logits

            logits = logits.float()
            probs = F.softmax(logits, dim=-1)

            # Entropy
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)
            all_entropies.append(entropy.mean().item())

            # Top-1 confidence
            top1_prob, top1_pred = probs.max(dim=-1)
            all_top1_conf.append(top1_prob.mean().item())

            # ECE binning
            confidences = top1_prob.view(-1).cpu()
            predictions = top1_pred.view(-1).cpu()
            targets = y.view(-1).cpu()

            correct = (predictions == targets).float()
            bin_indices = torch.clamp((confidences * n_bins).long(), 0, n_bins - 1)

            for b in range(n_bins):
                mask = bin_indices == b
                if mask.sum() > 0:
                    bin_correct[b] += correct[mask].sum()
                    bin_confidence[b] += confidences[mask].sum()
                    bin_count[b] += mask.sum()

        # Compute ECE
        ece = 0.0
        total = bin_count.sum()
        for b in range(n_bins):
            if bin_count[b] > 0:
                avg_conf = bin_confidence[b] / bin_count[b]
                avg_acc = bin_correct[b] / bin_count[b]
                ece += (bin_count[b] / total) * abs(avg_conf - avg_acc)

        return {
            "mean_entropy": sum(all_entropies) / len(all_entropies),
            "top1_confidence": sum(all_top1_conf) / len(all_top1_conf),
            "ece": ece.item(),
        }

    @torch.no_grad()
    def attention_interpretability(self):
        """
        Measure how interpretable the model's attention patterns are.

        Attention heads that attend either very broadly (uniform) or very
        sharply (focused) are more interpretable than those with intermediate,
        diffuse patterns.

        Returns:
            dict with:
              - mean_attn_entropy: average attention entropy across heads
              - entropy_variance: variance of attention entropy across heads
                (higher = more head specialization)
        """
        # This metric requires access to attention weights, which FA3 doesn't
        # expose by default. Return placeholder values and note this as a
        # future direction.
        return {
            "mean_attn_entropy": float("nan"),
            "entropy_variance": float("nan"),
            "note": "Requires attention weight extraction (not available with FA3)",
        }

    def evaluate_all(self):
        """Run all safety metrics and return combined results."""
        results = {}
        results["representation"] = self.representation_diversity()
        results["calibration"] = self.prediction_calibration()
        results["attention"] = self.attention_interpretability()
        return results

    def print_report(self):
        """Run all metrics and print a formatted report."""
        results = self.evaluate_all()
        print("\n=== Safety Metrics Report ===")
        print(f"\nRepresentation Diversity:")
        print(f"  Cosine similarity (lower=better): {results['representation']['cos_sim_mean']:.4f}")
        print(f"  Effective rank (higher=better):    {results['representation']['effective_rank']:.1f}")
        print(f"\nPrediction Calibration:")
        print(f"  Mean entropy:                      {results['calibration']['mean_entropy']:.4f}")
        print(f"  Top-1 confidence:                  {results['calibration']['top1_confidence']:.4f}")
        print(f"  ECE (lower=better):                {results['calibration']['ece']:.4f}")
        print(f"\nAttention Interpretability:")
        if math.isnan(results['attention']['mean_attn_entropy']):
            print(f"  (Not available with FA3 — requires attention weight extraction)")
        else:
            print(f"  Mean attention entropy:             {results['attention']['mean_attn_entropy']:.4f}")
            print(f"  Entropy variance:                   {results['attention']['entropy_variance']:.4f}")
        print("=" * 32)
        return results
