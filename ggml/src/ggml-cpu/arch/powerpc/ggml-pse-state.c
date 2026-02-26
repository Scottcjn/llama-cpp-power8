/*
 * PSE Global State - POWER8/9
 * Elyan Labs 2025
 */

#include "ggml-pse-integration.h"

/* PSE collapse parameters */
pse_collapse_state_t g_pse_state = {
    .topk = PSE_TOPK_DEFAULT,
    .amplify_factor = PSE_AMPLIFY_DEFAULT,
    .prune_threshold = PSE_PRUNE_THRESHOLD,
    .entropy_strength = PSE_ENTROPY_STRENGTH,
    .burst_interval = PSE_BURST_INTERVAL,
    .token_count = 0,
    .last_entropy = 0,
};

/* PSE global system state */
pse_global_state_t g_pse_global = {
    .initialized = false,
    .enabled = true,
    .numa_nodes = 0,
    .total_ram_gb = 0,
    .arch_name = "unknown",
};
