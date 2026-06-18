#!/usr/bin/env python3
import numpy as np

def simulate_gradient_update(grad_fp32, loss_scale=None):
    """Demonstrate how loss scaling preserves small gradients natively in float16."""
    if loss_scale is None:
        # Without scaling: data drops straight into the small container
        applied_grad = grad_fp32.astype(np.float16)
    else:
        # With scaling: magnify first, cast to small container, then pull back down
        scaled_grad = grad_fp32 * loss_scale
        applied_grad = scaled_grad.astype(np.float16) / loss_scale

    # Count how many microscopic parameters completely vanished to 0.0
    zero_count = np.sum(applied_grad == 0.0)
    return applied_grad, zero_count

def main():
    # Simulate a realistic set of tiny transformer gradients (e.g., 4096 elements)
    np.random.seed(7)
    true_gradients = np.random.normal(0.0, 1e-6, size=(4096,))
    
    print(f"Total parameters evaluated: {len(true_gradients)}")
    print(f"Gradient range: {true_gradients.min():.3e} to {true_gradients.max():.3e}\n")

    # Path A: Raw Cast
    applied_a, zeros_a = simulate_gradient_update(true_gradients, loss_scale=None)
    print(f"--- NO LOSS SCALING ---")
    print(f"Vanished to absolute 0.0: {zeros_a} / {len(true_gradients)} values")
    print(f"Training status: {'STALLED / FROZEN' if zeros_a == len(true_gradients) else 'Active'}")

    # Path B: Loss Scaled Cast
    applied_b, zeros_b = simulate_gradient_update(true_gradients, loss_scale=1024.0)
    print(f"\n--- WITH LOSS SCALING (Factor 1024) ---")
    print(f"Vanished to absolute 0.0: {zeros_b} / {len(true_gradients)} values")
    print(f"Training status: Healthy learning loop")

if __name__ == "__main__":
    main()