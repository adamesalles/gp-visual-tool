<script lang="ts">
  import svelteLogo from "./assets/svelte.svg";
  import viteLogo from "/vite.svg";
  import Counter from "./lib/Counter.svelte";
  import Katex from "./lib/Katex.svelte";

  function linspace(
    start: number,
    stop: number,
    num: number,
    endpoint = true
  ): number[] {
    const div = endpoint ? num - 1 : num;
    const step = (stop - start) / div;
    return Array.from({ length: num }, (_, i) => start + step * i);
  }

  function gaussian(x: number, mu: number, sigma: number): number {
    return (
      Math.exp(-Math.pow(x - mu, 2) / (2 * Math.pow(sigma, 2))) /
      (sigma * Math.sqrt(2 * Math.PI))
    );
  }

  class Kernel {
    name: string;
    params: number[];
    constructor(name: string, params: number[]) {
      this.name = name;
      this.params = params;
    }
    get math() {
      switch (this.name) {
        case "RBF":
          return `k(x, x') = ${this.params[0]}^2 \\exp \\left( -\\frac{1}{2} \\frac{(x - x')^2}{${this.params[1]}^2} \\right)`;
        case "Matern":
          return `k(x, x') = \\frac{2^{1 - \\nu}}{\\Gamma(\\nu)} \\left( \\sqrt{2 \\nu} \\frac{|x - x'|}{${this.params[1]}} \\right)^\\nu K_\\nu \\left( \\sqrt{2 \\nu} \\frac{|x - x'|}{${this.params[1]}} \\right)`;
        case "Periodic":
          return `k(x, x') = ${this.params[0]}^2 \\exp \\left( -2 \\frac{\\sin^2(\\pi |x - x'| / ${this.params[1]})}{${this.params[2]}^2} \\right)`;
        case "Linear":
          return `k(x, x') = ${this.params[0]} + ${this.params[1]} x x'`;
        case "Polynomial":
          return `k(x, x') = (${this.params[0]} + ${this.params[1]} x x')^${this.params[2]}`;
        case "Cosine":
          return `k(x, x') = ${this.params[0]} \\cos \\left( \\frac{2 \\pi |x - x'|}{${this.params[1]}} \\right)`;
      }
    }
  }

  let x_train = linspace(-5, 5, 10);
  let y_train = x_train.map((x) => gaussian(x, 0, 1));
  let x_test = linspace(-5, 5, 1000);
  let mu: number[];
  let lower: number[];
  let upper: number[];
  let kernel = new Kernel("RBF", [1, 1]);

  function getPosterior(
    x_train: number[],
    y_train: number[],
    x_test: number[],
    kernel: Kernel
  ): Promise<{ mu: number[]; lower: number[]; upper: number[] }> {
    return fetch("http://localhost:5000/api/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        x_train,
        y_train,
        x_test,
        kernel_name: kernel.name,
        kernel_params: kernel.params,
      }),
    }).then((res) => res.json());
  }

  $: {
    getPosterior(x_train, y_train, x_test, kernel).then((res) => {
      mu = res.mu;
      lower = res.lower;
      upper = res.upper;
    });
  }
</script>

<main>
  <h1>Gaussian Processes Visual Tool</h1>
</main>

<style>
  .logo {
    height: 6em;
    padding: 1.5em;
    will-change: filter;
    transition: filter 300ms;
  }
  .logo:hover {
    filter: drop-shadow(0 0 2em #646cffaa);
  }
  .logo.svelte:hover {
    filter: drop-shadow(0 0 2em #ff3e00aa);
  }
  .read-the-docs {
    color: #888;
  }
</style>
