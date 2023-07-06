export function linspace(
  start: number,
  stop: number,
  num: number,
  endpoint = true
): number[] {
  const div = endpoint ? num - 1 : num;
  const step = (stop - start) / div;
  return Array.from({ length: num }, (_, i) => start + step * i);
}

export function gaussian(x: number, mu: number, sigma: number): number {
  return (
    Math.exp(-Math.pow(x - mu, 2) / (2 * Math.pow(sigma, 2))) /
    (sigma * Math.sqrt(2 * Math.PI))
  );
}
