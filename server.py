import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from mcp.server.fastmcp import FastMCP

# Initialize the FastMCP Server
mcp = FastMCP(
    name="Third-Factor-Meta-Evaluator",
    instructions="A metacognitive evaluation server implementing Dąbrowski's Third Factor algorithm for assessing branch stability and invariance."
)

@mcp.tool()
def evaluate_third_factor(
    ideal_vector: List[float],
    candidate_states: List[List[float]],
    f1_reflex_logits: List[List[float]],
    branch_ids: List[str],
    k_perturbations: int = 5,
    noise_std: float = 0.05,
    lambda_fragility: float = 0.4,
    lambda_suppression: float = 0.3,
    fragility_threshold: float = 0.15
) -> Dict[str, Any]:
    """
    Evaluates reasoning branches using Kazimierz Dąbrowski's Third Factor algorithm.

    Args:
        ideal_vector: The axiomatic reference embedding (Personality Ideal / Target).
        candidate_states: List of state vectors representing current candidate branches.
        f1_reflex_logits: List of reflex/prior logits from the primary generator.
        branch_ids: Unique identifiers for each branch.
        k_perturbations: Number of Gaussian noise attacks to execute per branch.
        noise_std: Standard deviation of seismographic noise injection.
        lambda_fragility: Penalty weight for structural fragility.
        lambda_suppression: Penalty weight for greedy reflex compliance.
        fragility_threshold: Maximum allowed drift before classifying a branch as SHATTERED.

    Returns:
        Structured evaluation metrics, classifications (CRYSTALLINE, DUCTILE, SHATTERED), and summary counts.
    """
    # Validation
    num_branches = len(branch_ids)
    if len(candidate_states) != num_branches or len(f1_reflex_logits) != num_branches:
        return {
            "error": f"Dimension mismatch: candidate_states ({len(candidate_states)}), f1_reflex_logits ({len(f1_reflex_logits)}), and branch_ids ({num_branches}) must have matching lengths."
        }

    with torch.no_grad():
        ideal_t = torch.tensor(ideal_vector, dtype=torch.float32)
        states_t = torch.tensor(candidate_states, dtype=torch.float32)
        f1_t = torch.tensor(f1_reflex_logits, dtype=torch.float32)

        # 1. Creative Tension (Ideal Gap)
        loss_ideal = F.mse_loss(states_t, ideal_t.expand_as(states_t), reduction='none').mean(dim=-1)

        # 2. Seismographic Fragility (Perturbation Analysis)
        fragility_scores = []
        for _ in range(k_perturbations):
            noise = torch.randn_like(states_t) * noise_std
            perturbed = states_t + noise
            drift = torch.norm(perturbed - states_t, p=2, dim=-1)
            fragility_scores.append(drift)
        loss_fragility = torch.stack(fragility_scores, dim=0).mean(dim=0)

        # 3. Reflex Drive Suppression (Distance from reflex pull)
        override_dir = ideal_t.expand_as(states_t) - states_t
        cos_sim = F.cosine_similarity(override_dir, f1_t, dim=-1)
        loss_suppression = torch.clamp(1.0 - cos_sim, min=0.0)

        # Total Loss Calculation
        total_loss = loss_ideal + (lambda_fragility * loss_fragility) + (lambda_suppression * loss_suppression)

        # Classification Logic
        median_loss = total_loss.median().item()
        is_shattered = loss_fragility > fragility_threshold
        is_crystalline = (loss_fragility <= fragility_threshold) & (total_loss <= median_loss)

        # Build Branch Results
        branches_out = []
        for i in range(num_branches):
            if is_shattered[i].item():
                status = "SHATTERED"
            elif is_crystalline[i].item():
                status = "CRYSTALLINE"
            else:
                status = "DUCTILE"

            branches_out.append({
                "branch_id": branch_ids[i],
                "total_loss": round(total_loss[i].item(), 6),
                "creative_tension": round(loss_ideal[i].item(), 6),
                "fragility": round(loss_fragility[i].item(), 6),
                "classification": status
            })

        crystalline_count = sum(1 for b in branches_out if b["classification"] == "CRYSTALLINE")
        shattered_count = sum(1 for b in branches_out if b["classification"] == "SHATTERED")
        ductile_count = num_branches - crystalline_count - shattered_count

        return {
            "summary": {
                "total_evaluated": num_branches,
                "crystalline_count": crystalline_count,
                "shattered_count": shattered_count,
                "ductile_count": ductile_count
            },
            "branches": branches_out
        }

if __name__ == "__main__":
    # Start the FastMCP server with SSE transport bound for public network accessibility
    print("Starting Third Factor MCP Server on http://0.0.0.0:8000/sse ...")
    mcp.run(transport="sse", host="0.0.0.0", port=8000)
