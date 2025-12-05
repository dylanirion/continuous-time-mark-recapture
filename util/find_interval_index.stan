int find_interval_index(real t, array[] real grid, int last_idx) {
  int K = size(grid);
  if (t < grid[1])
    return 0; // should not happen if grid covers T
    
  // start search from last_idx since time moves forward
  for (k in last_idx : (K - 1)) {
    if (t >= grid[k] && t < grid[k + 1]) {
      return k;
    }
  }
  return K;
}
