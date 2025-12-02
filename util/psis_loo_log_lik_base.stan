//TODO: are closure units individuals, or captures? https://jsocolar.github.io/likelihoodOccupancy/
vector psis_loo_log_lik(int n_individuals, array[] int n_captures,
                        array[,] real capture_times,
                        array[,] int capture_states, real T_end,
                        matrix Q_mod, vector lambda) {
  vector[n_individuals] log_lik;
  for (i in 1 : n_individuals) {
    array[n_captures[i]] real times = capture_times[i, 1 : n_captures[i]];
    array[n_captures[i]] int states = capture_states[i, 1 : n_captures[i]];
    log_lik[i] = observed_individual_lpdf(times | states, T_end, Q_mod,
                   lambda);
  }
  return log_lik;
}
