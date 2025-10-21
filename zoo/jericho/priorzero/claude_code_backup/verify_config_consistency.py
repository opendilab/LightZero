#!/usr/bin/env python3
"""
Verify PriorZero Configuration Consistency

This script checks that all sequence-length related parameters are consistent
to prevent the attention mask dimension mismatch error.
"""

from priorzero_config import get_priorzero_config_for_quick_test


def verify_config_consistency():
    """Verify that all sequence-length parameters are consistent."""
    print("="*80)
    print("Verifying PriorZero Quick Test Configuration Consistency")
    print("="*80)

    main_cfg, _ = get_priorzero_config_for_quick_test()

    # Extract relevant parameters
    policy_num_unroll = main_cfg.policy.num_unroll_steps
    wm_num_unroll = main_cfg.policy.model.world_model_cfg.num_unroll_steps
    wm_max_blocks = main_cfg.policy.model.world_model_cfg.max_blocks
    wm_max_tokens = main_cfg.policy.model.world_model_cfg.max_tokens
    wm_tokens_per_block = main_cfg.policy.model.world_model_cfg.tokens_per_block
    wm_infer_context = main_cfg.policy.model.world_model_cfg.infer_context_length
    wm_context_length = main_cfg.policy.model.world_model_cfg.context_length

    print("\n[Extracted Configuration Parameters]")
    print(f"  policy.num_unroll_steps:                        {policy_num_unroll}")
    print(f"  policy.model.world_model_cfg.num_unroll_steps:  {wm_num_unroll}")
    print(f"  policy.model.world_model_cfg.max_blocks:        {wm_max_blocks}")
    print(f"  policy.model.world_model_cfg.max_tokens:        {wm_max_tokens}")
    print(f"  policy.model.world_model_cfg.tokens_per_block:  {wm_tokens_per_block}")
    print(f"  policy.model.world_model_cfg.infer_context_length: {wm_infer_context}")
    print(f"  policy.model.world_model_cfg.context_length:    {wm_context_length}")

    # Verify consistency
    print("\n[Consistency Checks]")
    all_checks_passed = True

    # Check 1: policy.num_unroll_steps == world_model_cfg.num_unroll_steps
    check1 = policy_num_unroll == wm_num_unroll
    status1 = "✓ PASS" if check1 else "✗ FAIL"
    print(f"  {status1}: policy.num_unroll_steps == wm.num_unroll_steps")
    print(f"          ({policy_num_unroll} == {wm_num_unroll})")
    all_checks_passed &= check1

    # Check 2: max_blocks == num_unroll_steps
    check2 = wm_max_blocks == wm_num_unroll
    status2 = "✓ PASS" if check2 else "✗ FAIL"
    print(f"  {status2}: wm.max_blocks == wm.num_unroll_steps")
    print(f"          ({wm_max_blocks} == {wm_num_unroll})")
    all_checks_passed &= check2

    # Check 3: max_tokens == num_unroll_steps * tokens_per_block
    expected_max_tokens = wm_num_unroll * wm_tokens_per_block
    check3 = wm_max_tokens == expected_max_tokens
    status3 = "✓ PASS" if check3 else "✗ FAIL"
    print(f"  {status3}: wm.max_tokens == wm.num_unroll_steps * wm.tokens_per_block")
    print(f"          ({wm_max_tokens} == {wm_num_unroll} * {wm_tokens_per_block} = {expected_max_tokens})")
    all_checks_passed &= check3

    # Check 4: context_length == infer_context_length * tokens_per_block
    expected_context_length = wm_infer_context * wm_tokens_per_block
    check4 = wm_context_length == expected_context_length
    status4 = "✓ PASS" if check4 else "✗ FAIL"
    print(f"  {status4}: wm.context_length == wm.infer_context_length * wm.tokens_per_block")
    print(f"          ({wm_context_length} == {wm_infer_context} * {wm_tokens_per_block} = {expected_context_length})")
    all_checks_passed &= check4

    # Summary
    print("\n" + "="*80)
    if all_checks_passed:
        print("✓ ALL CONSISTENCY CHECKS PASSED!")
        print(f"\nExpected sequence lengths:")
        print(f"  - Training unroll:    {wm_num_unroll} timesteps = {wm_max_tokens} tokens")
        print(f"  - Inference context:  {wm_infer_context} timesteps = {wm_context_length} tokens")
        print(f"\nAttention mask will be created for {wm_max_tokens} tokens (shape: [{wm_max_tokens}, {wm_max_tokens}])")
        print("This should match the sequence length being processed.")
    else:
        print("✗ SOME CONSISTENCY CHECKS FAILED!")
        print("Please review the configuration to fix inconsistencies.")
    print("="*80)

    return all_checks_passed


if __name__ == "__main__":
    success = verify_config_consistency()
    exit(0 if success else 1)
