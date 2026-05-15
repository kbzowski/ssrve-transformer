"""Repair / output activation layer for SSRVE generator.

Decoder outputs:
- nuc_ratio    -> sigmoid * max_nucleation_ratio (bounded to [0, max_nuc]
                 both in training and inference). Ensures physically
                 meaningful values (no negative nucleation ratios) and
                 prevents the loss from wasting capacity on out-of-range
                 predictions.
- phase_logits to softmax over num_phases.
- active_logit to sigmoid (BCE supervised).

xy is NOT produced by the decoder - nucleation positions are sampled
uniformly at inference time, which corresponds to the physical randomness
of nucleation sites in a specific fragment of steel.
"""

import torch
import torch.nn as nn


class RepairLayer(nn.Module):
    def __init__(self, max_nucleation_ratio: float = 0.3, num_phases: int = 4):
        super().__init__()
        self.max_nucleation_ratio = max_nucleation_ratio
        self.num_phases = num_phases

    def forward(
        self,
        nuc_raw: torch.Tensor,
        phase_logits: torch.Tensor,
        active_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            nuc_raw:       (B, P, 1) - unbounded decoder output
            phase_logits:  (B, P, num_phases)
            active_logit:  (B, P, 1)

        Returns:
            nuc:          (B, P, 1) - in [0, max_nucleation_ratio]
            phase_probs:  (B, P, num_phases)
            active_prob:  (B, P, 1)
        """
        nuc = torch.sigmoid(nuc_raw) * self.max_nucleation_ratio
        phase_probs = torch.softmax(phase_logits, dim=-1)
        active_prob = torch.sigmoid(active_logit)
        return nuc, phase_probs, active_prob

    def get_phase_ids(self, phase_logits: torch.Tensor) -> torch.Tensor:
        """Return discrete phase ids in 1..num_phases."""
        return phase_logits.argmax(dim=-1) + 1

    def get_phase_ids_budgeted(
        self,
        phase_probs: torch.Tensor,
        active_mask: torch.Tensor,
        target_fractions: torch.Tensor,
    ) -> torch.Tensor:
        """Budget-constrained phase assignment for inference.

        Instead of per-slot argmax (which collapses to dominant phase),
        assigns phases so that the resulting counts match target_fractions.
        Minority phases are assigned first so they can pick their preferred
        slots; ferrite fills the rest.

        Args:
            phase_probs:       (B, P, num_phases) - softmax outputs
            active_mask:       (B, P) bool - True = active slot
            target_fractions:  (B, num_phases) - e.g. from fraction_head

        Returns:
            phase_id: (B, P) long, values in 1..num_phases
        """
        B, P, C = phase_probs.shape
        phase_id = torch.ones(B, P, dtype=torch.long, device=phase_probs.device)

        for b in range(B):
            active_idx = active_mask[b].nonzero(as_tuple=False).squeeze(-1)
            n_active = active_idx.numel()
            if n_active == 0:
                continue

            fracs = target_fractions[b]  # (C,)
            n_per_phase = (fracs * n_active).round().long()

            # Fix rounding so counts sum exactly to n_active
            diff = n_active - n_per_phase.sum().item()
            if diff != 0:
                n_per_phase[n_per_phase.argmax()] += diff

            probs_active = phase_probs[b, active_idx]  # (n_active, C)
            assigned = torch.zeros(n_active, dtype=torch.bool, device=phase_probs.device)
            assignment = torch.zeros(n_active, dtype=torch.long, device=phase_probs.device)

            # Process minority phases first so they get best-matching slots
            phase_order = n_per_phase.argsort()
            for k in phase_order.tolist():
                n_k = int(n_per_phase[k].item())
                if n_k <= 0:
                    continue
                scores = probs_active[:, k].clone()
                scores[assigned] = -1.0
                n_available = int((~assigned).sum().item())
                topk_idx = scores.topk(min(n_k, n_available)).indices
                assignment[topk_idx] = k
                assigned[topk_idx] = True

            # Any remaining unassigned slots go to ferrite (phase index 0)
            assignment[~assigned] = 0

            phase_id[b, active_idx] = assignment + 1  # 1-indexed

        return phase_id

    def clamp_nuc_inference(self, nuc: torch.Tensor) -> torch.Tensor:
        """Redundant after sigmoid activation but kept for safety/API compat."""
        return nuc.clamp(0.0, self.max_nucleation_ratio)
