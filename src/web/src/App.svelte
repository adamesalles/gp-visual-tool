<script lang="ts">
  import Katex from "./lib/Katex.svelte";
  import { mu, upper, lower } from "./stores.ts";
  import * as d3 from "d3";
  import { onMount } from "svelte";

  import { gaussian, linspace } from "./funcs.ts";

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

  let x_train = [];
  let y_train = [];
  let x_test_start = -1;
  let x_test_end = 1;
  let amount_test_points = 1000;
  $: x_test = linspace(x_test_start, x_test_end, amount_test_points);
  let kernel = new Kernel("RBF", [1, 1]);

  let width;
  let height;

  let posterior;
  const padding = 30;

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
      $mu = res.mu;
      $lower = res.lower;
      $upper = res.upper;
    });
  }

  $: xScalePosterior = d3
    .scaleLinear()
    .domain([Math.min(...x_test), Math.max(...x_test)])
    .range([0, width]);
  $: yScalePosterior = d3
    .scaleLinear()
    .domain([Math.min(...$lower) - 1, Math.max(...$upper) + 1])
    .range([height, 0]);

  $: {
    const svgPosterior = d3.select("#posterior");
    const svgPosteriorG = d3.select("#posterior-g");

    if (svgPosterior.node() == null) break $;

    let valueLine = d3
      .line()
      .x((d: any) => xScalePosterior(d.x))
      .y((d: any) => yScalePosterior(d.y));

    let area = d3
      .area()
      .x((d: any) => xScalePosterior(d.x))
      .y0((d: any) => yScalePosterior(d.y0))
      .y1((d: any) => yScalePosterior(d.y1));

    const xAxisPosterior = d3.axisBottom(xScalePosterior);
    const yAxisPosterior = d3.axisLeft(yScalePosterior);

    svgPosteriorG
      .selectAll("path.line")
      .data([
        x_test.map((x, i) => {
          return { x, y: $mu[i] };
        }),
      ])
      .join("path")
      .transition()
      .duration(200)
      .attr("class", "line")
      .attr("d", (d) => valueLine(d))
      .attr("fill", "none")
      .attr("stroke", "red")
      .attr("stroke-width", 2);

    svgPosteriorG
      .selectAll("path.area")
      .data([
        x_test.map((x, i) => {
          return { x, y0: $lower[i], y1: $upper[i] };
        }),
      ])
      .join("path")
      .transition()
      .duration(200)
      .attr("class", "area")
      .attr("d", (d) => area(d))
      .attr("fill", "red")
      .attr("opacity", 0.2);

    svgPosteriorG
      .selectAll("circle")
      .data(
        x_train.map((x, i) => {
          return { x, y: y_train[i] };
        })
      )
      .join("circle")
      .attr("cx", (d) => xScalePosterior(d.x))
      .attr("cy", (d) => yScalePosterior(d.y))
      .attr("r", 5)
      .attr("fill", "transparent")
      .attr("stroke", "white")
      .attr("stroke-width", 2)
      .attr("i", (d, i) => i)
      .on("click", (event, d) => {
        event.stopPropagation();
        removePoint(+d3.select(event.target).attr("i"));
      });

    svgPosteriorG
      .select(".posterior-x-axis")
      .attr("transform", `translate(0, ${height})`)
      .call(xAxisPosterior);

    svgPosteriorG.select(".posterior-y-axis").call(yAxisPosterior);
  }

  onMount(() => {
    const svgPosterior = d3.select("#posterior");
    width = svgPosterior.node().getBoundingClientRect().width - 2 * padding;
    height = svgPosterior.node().getBoundingClientRect().height - 2 * padding;

    svgPosterior.on("click", (event) => {
      const [x, y] = d3.pointer(event);
      x_train = [...x_train, xScalePosterior.invert(x - padding)];
      y_train = [...y_train, yScalePosterior.invert(y - padding)];
    });
  });

  function removePoint(index: number) {
    x_train = x_train.filter((_, i) => i !== index);
    y_train = y_train.filter((_, i) => i !== index);
  }
</script>

<main>
  <h1>Gaussian Processes Visual Tool</h1>

  <div class="viz">
    <svg id="posterior" width="100%" height="100%">
      <g id="posterior-g" transform="translate({padding}, {padding})">
        <g class="posterior-x-axis" />
        <g class="posterior-y-axis" />
      </g>
    </svg>

    <svg id="marginal" width="100%" height="100%" />

    <div class="simulation-panel">
      <!-- Button to reset x_train/y_train -->
      <button
        on:click={() => {
          x_train = [];
          y_train = [];
        }}>Clear observations</button
      >

      <!-- Input to x_test_start and x_test_end -->
      <div id="x-input">
        <label for="x_test_start">$x$ start</label>
        <input type="number" id="x_test_start" bind:value={x_test_start} />

        <label for="x_test_end">$x$ end</label>
        <input type="number" id="x_test_end" bind:value={x_test_end} />
      </div>

      <!-- Input to amount of x_test_points -->
      <label for="x_test_points">Amount of $x$ points</label>
      <input type="number" id="x_test_points" bind:value={amount_test_points} />
    </div>
  </div>
</main>

<style lang="scss">
  main {
    display: flex;
    flex-direction: column;
    width: 100vw;
    height: 100vh;
    padding: 2rem;
  }

  .viz {
    display: grid;
    height: 100%;
    grid-template-columns: 2fr 1fr 1fr;
    grid-template-rows: 2fr 1fr;
    gap: 1rem;
    // grid-row: 1fr 1fr;
  }

  #x-input {
    display: flex;
    flex-direction: row;
    gap: 1rem;
  }

  // svg {
  //   user-select: none;
  // }

  #posterior {
    width: 100%;
    height: 100%;
    // background: red;
  }

  #marginal {
    width: 100%;
    height: 100%;
    background: blue;
  }
</style>
