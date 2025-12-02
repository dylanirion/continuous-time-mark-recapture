// https://discourse.mc-stan.org/t/following-up-on-several-discussions-of-simplex-adjustments-and-ragged-arrays-of-simplexes/36218/4
// https://github.com/bob-carpenter/transforms/blob/ccf7a35695518c24e9433cf01795b681b4b67d32/simplex_transforms/stan/transforms/ILR/ILR_functions.stan

// taken from above, performs performant transformation of vector into simplex, with adjustment to jacobian
vector inv_ilr_simplex_constrain_lp(vector y) {
  int N = rows(y) + 1;
  vector[N - 1] ns = linspaced_vector(N - 1, 1, N - 1);
  vector[N - 1] w = y ./ sqrt(ns .* (ns + 1));
  vector[N] z = append_row(reverse(cumulative_sum(reverse(w))), 0)
                - append_row(0, ns .* w);
  real r = log_sum_exp(z);
  vector[N] x = exp(z - r);
  target += 0.5 * log(N);
  target += sum(z) - N * r;
  return x;
}

// same as above without jacobian adjustment
vector inv_ilr_simplex_constrain(vector y) {
  int N = rows(y) + 1;
  vector[N - 1] ns = linspaced_vector(N - 1, 1, N - 1);
  vector[N - 1] w = y ./ sqrt(ns .* (ns + 1));
  vector[N] z = append_row(reverse(cumulative_sum(reverse(w))), 0)
                - append_row(0, ns .* w);
  real r = log_sum_exp(z);
  return exp(z - r);
}
