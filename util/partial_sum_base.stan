real partial_sum(array[] int slice, int start, int end,
                 array[,] real capture_times, array[,] int capture_states,
                 array[] int n_captures, real T_end, matrix Q_mod,
                 vector lambda) {
  real total_log_prob = 0;
  for (i in 1 : size(slice)) {
    int idx = slice[i];
    
    array[n_captures[idx]] real times = capture_times[idx, 1 : n_captures[idx]];
    array[n_captures[idx]] int states = capture_states[idx, 1 : n_captures[idx]];
    total_log_prob += observed_individual_lpdf(times | states, T_end,
                        Q_mod, lambda);
  }
  return total_log_prob;
}
