// log likelihood of an individual capture history
matrix piecewise_prob(matrix Q, vector lambda, real dt,
                      array[] int is_observable) {
  matrix[rows(Q), cols(Q)] Q_piecewise = Q;
    
  // scale detection (lambda)
  int observable_idx = 1;
  for (r in 1 : rows(Q)) {
    if (is_observable[r]) {
      // subtract the detection rate from the Q diagonal (leakage).
      Q_piecewise[r, r] -= lambda[r];
      observable_idx += 1;
    }
  }
    
  return matrix_exp(Q_piecewise * dt);
}
  
real observed_individual_lpdf(array[] real capture_times,
                              array[] int capture_states, real T_end,
                              matrix Q, vector lambda,
                              array[] int is_observable,
                              array[] real effort_times,
                              array[,] real effort_values) {
  int n_captures = size(capture_times);
  real log_prob = 0.0;
  int current_effort_idx = 1; // cache to speed up search
    
  // Subsequent captures
  for (i in 2 : n_captures) {
    real dt = capture_times[i] - capture_times[i - 1];
    int s_prev = capture_states[i - 1];
    int s_curr = capture_states[i];
    if (dt < 1e-9) {
      if (s_prev != s_curr) 
        return negative_infinity();
      continue;
    }
      
    matrix[rows(Q), cols(Q)] P_trans = diag_matrix(
                                                   rep_vector(1.0, rows(Q)));
      
    // piecewise integration
    real t_piecewise = capture_times[i - 1];
      
    while (t_piecewise < capture_times[i]) {
      current_effort_idx = find_interval_index(t_piecewise, effort_times,
                             current_effort_idx);
        
      real t_next_effort = (current_effort_idx < size(effort_times))
                           ? effort_times[current_effort_idx + 1] : T_end;
        
      real t_step_end = fmin(capture_times[i], t_next_effort);
      real dt_piecewise = t_step_end - t_piecewise;
      real effort = effort_values[current_effort_idx, s_curr];
      P_trans = P_trans
                * piecewise_prob(Q, lambda * effort, dt_piecewise,
                                 is_observable);
      t_piecewise = t_step_end;
    }
      
    if (P_trans[s_prev, s_curr] < 1e-30) 
      return negative_infinity();
    log_prob += log(P_trans[s_prev, s_curr])
                + log(
                      lambda[s_curr]
                      * effort_values[find_interval_index(capture_times[i],
                                        effort_times, current_effort_idx), s_curr]);
  }
    
  // No captures after last event
  real t_last = capture_times[n_captures];
  int s_last = capture_states[n_captures];
  if (T_end - t_last > 1e-9) {
    matrix[rows(Q), cols(Q)] P_final = diag_matrix(
                                                   rep_vector(1.0, rows(Q)));
      
    // piecewise integration
    real t_piecewise = t_last;
      
    while (t_piecewise < T_end) {
      current_effort_idx = find_interval_index(t_piecewise, effort_times,
                             current_effort_idx);
      real t_next_effort = (current_effort_idx < size(effort_times))
                           ? effort_times[current_effort_idx + 1] : T_end;
      real t_step_end = fmin(T_end, t_next_effort);
      real dt_piecewise = t_step_end - t_piecewise;
      real effort = effort_values[current_effort_idx, s_last];
      P_final = P_final
                * piecewise_prob(Q, lambda * effort, dt_piecewise,
                                 is_observable);
      t_piecewise = t_step_end;
    }
      
    real prob_no_detection = sum(P_final[capture_states[n_captures],  : ]);
    if (prob_no_detection < 1e-30) 
      return negative_infinity();
    log_prob += log(prob_no_detection);
  }
  return log_prob;
}
