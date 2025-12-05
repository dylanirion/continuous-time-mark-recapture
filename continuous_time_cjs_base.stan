// base cormac-jolly-seber model
functions {
  #include util/inv_ilr_simplex_constrain.stan
  #include likelihood/cjs/multistate_observed_individual_base.stan
  
  #include util/partial_sum_base.stan
  
  #include util/psis_loo_log_lik_base.stan
  #include util/psis_log_prior_base.stan
}
data {
  // model dimensions
  int<lower=1> n_individuals;
  int<lower=1> n_states;
  array[n_individuals] int<lower=0> n_captures;
  
  // capture data
  array[n_individuals, max(n_captures)] real<lower=0> capture_times;
  array[n_individuals, max(n_captures)] int<lower=0, upper=n_states> capture_states;
  
  real<lower=0> T_end;
  
  // NB: diagonal elements are not included (they're determined by row constraint)
  // so for each row r: columns 1:(r-1), (r+1):n_states are other states, column n_states is detection (lambda)
  array[n_states, n_states] int<lower=0, upper=1> constrain_to_zero; //boolean, 1 for constrained, 0 for not
  
  // priors
  //TODO: is there a way to know/calculate n_observable here?
  vector[n_states] log_exit_rate_prior_mu;
  vector[n_states] log_exit_rate_prior_sigma;
  
  int<lower=1> grainsize;
}
transformed data {
  array[n_individuals] int individual_indices = linspaced_int_array(
                                                  n_individuals, 1,
                                                  n_individuals);
  array[n_states] int n_free_per_state;
  array[n_states] int is_absorbing;
  array[n_states] int is_observable;
  
  // identify which states are absorbing (dead) and which are observable
  for (r in 1 : n_states) {
    int has_transitions = 0;
    // last column indicates if state is observable (can be detected)
    is_observable[r] = (constrain_to_zero[r, n_states] == 0) ? 1 : 0;
    
    // check if state has any outgoing transitions (excluding detection)
    for (c in 1 : (n_states - 1)) {
      if (constrain_to_zero[r, c] == 0) {
        has_transitions = 1;
      }
    }
    is_absorbing[r] = (has_transitions == 0) ? 1 : 0;
  }
  
  for (r in 1 : n_states) {
    int free_count = 0;
    
    // skip absorbing states (for now!) - they have no parameters to estimate
    //TODO: check is_observable? could we have an absorbing but observable state?
    if (is_absorbing[r] == 1) {
      n_free_per_state[r] = 0;
      continue;
    }
    
    for (j in 1 : n_states) {
      if (constrain_to_zero[r, j] == 0) 
        free_count += 1;
    }
    n_free_per_state[r] = free_count;
    if (n_free_per_state[r] == 0) 
      reject("Non-absorbing state ", r, " has no free parameters.");
  }
  
  /*
  // this might not be true if there are constraints that limit transitions into specific absorbing states
  if(sum(is_absorbing) > 1)
    reject(">1 absorbing states will not be identifiable.");
  */
  
  int n_effective_states = n_states - sum(is_absorbing);
}
parameters {
  vector[n_effective_states] log_exit_rate_raw;
  vector[sum(n_free_per_state) - n_effective_states] free_logits;
}
transformed parameters {
  vector[n_effective_states] log_exit_rate = rep_vector(0.0, n_effective_states);
  vector<lower=0>[n_states] exit_rate = rep_vector(0.0, n_states);
  
  // fill in exit rates only for non-absorbing states
  {
    int non_abs_idx = 1;
    for (r in 1 : n_states) {
      if (is_absorbing[r] == 1) 
        continue;
      log_exit_rate[non_abs_idx] = log_exit_rate_prior_mu[r]
                         + log_exit_rate_prior_sigma[r]
                           * log_exit_rate_raw[non_abs_idx];
      exit_rate[r] = exp(log_exit_rate[non_abs_idx]);
      non_abs_idx += 1;
    }
  }
  
  // event_probs[] indices 1 to (n_states-1) for transitions, and index n_states for detection.
  array[n_effective_states] simplex[n_states] event_probs;
  {
    int start_idx = 1;
    int observable_idx = 1;
    for (r in 1 : n_states) {
      // absorbing state: no events (will have zero exit rate)
      //TODO: check is_observable? could we have an absorbing but observable state?
      if (is_absorbing[r] == 1) 
        continue;
      
      vector[n_states] full_probs = rep_vector(0.0, n_states);
      int K = n_free_per_state[r];
      
      if (K == 1) {
        // only one free parameter - deterministic transition
        full_probs = rep_vector(0.0, n_states);
      } else {
        // multiple free parameters - use ILR transform
        vector[K] state_probs = inv_ilr_simplex_constrain_lp(
                                  segment(free_logits, start_idx, K - 1));
        int prob_idx = 1;
        //NB: explicitly not using c here, because 'columns' of constrain_to_zero do not align directly with states
        for (j in 1 : n_states) {
          if (constrain_to_zero[r, j] == 0) {
            full_probs[j] = state_probs[prob_idx];
            prob_idx += 1;
          }
        }
        start_idx += (K - 1);
      }
      event_probs[observable_idx] = full_probs;
      observable_idx += 1;
    }
  }
  
  // detection rates for states
  vector<lower=0>[n_effective_states] lambda = rep_vector(0.0,
                                                          n_effective_states);
  //vector<lower=0>[n_observable_states] mu;
  //modified Q matrix (with detection leakage)
  matrix[n_states, n_states] Q_mod = rep_matrix(0.0, n_states, n_states);
  
  {
    int observable_idx = 1;
    for (r in 1 : n_states) {
      // absorbing state: no transitions out
      if (is_absorbing[r] == 1) 
        continue;
      
      // fill off-diagonal elements (transitions to other states)
      int event_idx = 1;
      for (c in 1 : n_states) {
        if (r == c) 
          continue;
        Q_mod[r, c] = exit_rate[r] * event_probs[r][event_idx];
        event_idx += 1;
      }
      
      if (is_observable[r] == 1) {
        lambda[observable_idx] = exit_rate[r] * event_probs[r][event_idx];
        // used to also generate mu here, but maybe we have a model with not absorbing states?
        observable_idx += 1;
      }
      
      // diagonal: negative sum of row (including detection)
      real row_sum = sum(Q_mod[r,  : ]);
      Q_mod[r, r] = -row_sum - lambda[r];
    }
  }
}
model {
  // priors
  log_exit_rate_raw ~ std_normal();
  free_logits ~ std_normal();
  
  target += reduce_sum(partial_sum, individual_indices, grainsize,
                       capture_times, capture_states, n_captures, T_end,
                       Q_mod, lambda);
}
generated quantities {
  // convert back to original parameterization for interpretation (non diagonal elements, excluding absorbing)
  vector[n_effective_states * n_effective_states - n_effective_states] q;
  {
    int k = 1;
    for (r in 1 : n_states) {
      if (is_absorbing[r] == 1) 
        continue;
      
      int event_idx = 1;
      for (c in 1 : n_states) {
        if (r != c) {
          if (is_absorbing[c] != 1) {
            q[k] = exit_rate[r] * event_probs[r][event_idx];
            k += 1;
          }
          event_idx += 1;
        }
      }
    }
  }
  
  // pointwise log-likelihood for psis-loo
  vector[n_individuals] log_lik = psis_loo_log_lik(n_individuals, n_captures,
                                                   capture_times,
                                                   capture_states, T_end,
                                                   Q_mod, lambda);
  // log prior for power-scaling sensitivity
  real lprior = psis_log_prior(n_states, n_free_per_state, log_exit_rate_raw,
                               free_logits);
}
