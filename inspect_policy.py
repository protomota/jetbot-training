#!/usr/bin/env python3
"""Inspect the policy to determine input/output dimensions."""

import torch

# Load the policy
policy = torch.jit.load("jetbot_policy.pt")
policy.eval()

print("=" * 50)
print("Policy Inspection")
print("=" * 50)

# Try to get model graph info
print("\nModel structure:")
print(policy.graph)

# Test with dummy input to find dimensions
print("\n" + "=" * 50)
print("Testing input/output dimensions:")
print("=" * 50)

# Try different input sizes to find what works
found = False
for size in range(2, 101):  # Try sizes from 2 to 100
    try:
        dummy_input = torch.randn(1, size)
        with torch.no_grad():
            output = policy(dummy_input)
        print(f"\n✓ Input size {size} works!")
        print(f"  Input shape:  {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

        # Try a few more times to verify it's stable
        print("\n  Testing with different random inputs:")
        for i in range(3):
            test_input = torch.randn(1, size)
            with torch.no_grad():
                test_output = policy(test_input)
            print(f"    Test {i+1}: Output = {test_output.numpy().flatten()}")

        found = True
        break
    except Exception as e:
        if size <= 20 or size % 10 == 0:  # Only print some failures to avoid spam
            print(f"✗ Input size {size} failed: {type(e).__name__}")

if not found:
    print("\n⚠ Could not find working input size in range 2-100")
    print("  Your policy may have unusual input requirements.")

print("\n" + "=" * 50)
print("Next steps:")
print("=" * 50)
print("1. Note the input size that worked above")
print("2. Update get_observation() in run_policy.py to return that many values")
print("3. Fill those values with real sensor data matching your training")
print()
