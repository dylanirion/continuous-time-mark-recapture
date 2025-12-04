// log likelihood of an individual capture history
real observed_individual_lpdf(array[] real capture_times,
                              array[] int capture_states, real T_end,
                              matrix Q_mod, vector lambda) {
  int n_captures = size(capture_times);
  real log_prob = 0.0;
    
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
    matrix[rows(Q_mod), cols(Q_mod)] P_trans = matrix_exp(Q_mod * dt);
    if (P_trans[s_prev, s_curr] < 1e-30) 
      return negative_infinity();
    log_prob += log(P_trans[s_prev, s_curr]) + log(lambda[s_curr]);
  }
    
  // No captures after last event
  real t_last = capture_times[n_captures];
  if (T_end - t_last > 1e-9) {
    matrix[rows(Q_mod), cols(Q_mod)] P_final = matrix_exp(
                                                          Q_mod
                                                          * (T_end - t_last));
    real prob_no_detection = sum(P_final[capture_states[n_captures],  : ]);
    if (prob_no_detection < 1e-30) 
      return negative_infinity();
    log_prob += log(prob_no_detection);
  }
  return log_prob;
}
