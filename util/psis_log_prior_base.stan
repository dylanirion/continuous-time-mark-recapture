real psis_log_prior(int n_observable_states, array[] int n_free_per_state,
                    vector log_exit_rate_raw, vector free_logits) {
  real lprior;
  {
    real state_probs_prior = 0.0;
    int start_idx = 1;
    for (r in 1 : n_observable_states) {
      int K = n_free_per_state[r];
      if (K > 1) {
        vector[K] state_probs = inv_ilr_simplex_constrain(
                                  segment(free_logits, start_idx, K - 1));
        state_probs_prior += dirichlet_lpdf(state_probs |
                                            rep_vector(1.0, K));
        start_idx += (K - 1);
      }
    }
    lprior = std_normal_lpdf(log_exit_rate_raw)
             + std_normal_lpdf(free_logits) + state_probs_prior;
  }
  return lprior;
}
