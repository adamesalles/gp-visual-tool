<script lang="ts">
  import Katex from 'svelte-katex';
  import type { Matrix } from "ml-matrix";
  import { mu, upper, lower, covariance_matrix } from "./stores.ts";
  import * as d3 from "d3";
  import { onMount } from "svelte";

  import { gaussian, linspace } from "./funcs.ts";

  class Kernel {
    name: string;
    params: number[];
    constructor(name: string, params: number[]) {
      this.name = name;
      this.params = params;

      // set default params
      switch (this.name) {
        case "RBF":
          if (this.params.length == 0) {
            this.params = [1, 1];
          }
          break;
        case "Matern":
          if (this.params.length == 0) {
            this.params = [1, 1, 1];
          }
          break;
        case "Periodic":
          if (this.params.length == 0) {
            this.params = [1, 1, 1];
          }
          break;
        case "Linear":
          if (this.params.length == 0) {
            this.params = [1, 1];
          }
          break;
        case "Polynomial":
          if (this.params.length == 0) {
            this.params = [1, 1];
          }
          break;
        case "Cosine":
          if (this.params.length == 0) {
            this.params = [1, 1];
          }
          break;
      }
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
          return `k(x, x') = ${this.params[0]}(x x')^\{${this.params[1]}\}`;
        case "Cosine":
          return `k(x, x') = ${this.params[0]} \\cos \\left( \\frac{2 \\pi |x - x'|}{${this.params[1]}} \\right)`;
      }
    }

    get params_names() {
      switch (this.name) {
        case "RBF":
          return ["\\sigma", "l"];
        case "Matern":
          return ["\\sigma", "l", "ν"];
        case "Periodic":
          return ["\\sigma", "l", "p"];
        case "Linear":
          return ["\\sigma", "l"];
        case "Polynomial":
          return ["\\sigma", "p"];
        case "Cosine":
          return ["\\sigma", "l"];
      }
    }
  }

  let x_train = [];
  let y_train = [];
  let x_test_start = -1;
  let x_test_end = 1;
  let amount_test_points = 1000;
  $: x_test = linspace(x_test_start, x_test_end, amount_test_points);
  let sigma = 0.1;
  let kernel = new Kernel("RBF", [1, 1]);

  let width;
  let height;

  let info;
  const padding = 30;

  function getPosterior(
    x_train: number[],
    y_train: number[],
    x_test: number[],
    kernel: Kernel,
    sigma: number
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
        sigma,
        kernel_name: kernel.name,
        kernel_params: kernel.params,
      }),
    }).then((res) => res.json());
  }

  function getCovariance(): Promise<{ covariance_matrix: number[][] }> {
    return fetch("http://localhost:5000/api/kernel", {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    }).then((res) => res.json());
  }

  $: {
    getPosterior(x_train, y_train, x_test, kernel, sigma).then((res) => {
      $mu = res.mu;
      $lower = res.lower;
      $upper = res.upper;
    });

    getCovariance().then((res) => {
      $covariance_matrix = res.covariance_matrix;
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
      .attr("stroke-width", 3);

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
    const svgPosteriorG = d3.select("#posterior-g");
    width = svgPosterior.node().getBoundingClientRect().width - 2 * padding;
    height = svgPosterior.node().getBoundingClientRect().height - 2 * padding;

    svgPosterior.on("click", (event) => {
      const [x, y] = d3.pointer(event);
      x_train = [...x_train, xScalePosterior.invert(x - padding)];
      y_train = [...y_train, yScalePosterior.invert(y - padding)];
    });

    // Show marginal distribution on hover
    svgPosterior.on("mousemove", (event) => {
      const [x, y] = d3.pointer(event);
      const x0 = xScalePosterior.invert(x - padding);
      const y0 = yScalePosterior.invert(y - padding);

      // get closest mu and covariance for x coordinate
      const i = d3.bisectLeft(x_test, x0);
      const marginalMu = $mu[i];
      const marginalLower = $lower[i];
      const marginalUpper = $upper[i];
      const marginalVar = (marginalUpper - marginalLower) / 4;

      const marginalLowerPixel = yScalePosterior(marginalLower);
      const marginalUpperPixel = yScalePosterior(marginalUpper);

      const yMarginalScale = d3
        .scaleLinear()
        .domain([marginalLower, marginalUpper])
        .range([marginalLowerPixel, marginalUpperPixel]);

      const xMarginalScale = d3
        .scaleLinear()
        .domain([0, 1])
        .range([x - padding, x - padding + 200]);

      const marginalLine = d3
        .line()
        .x((d: any) => xMarginalScale(d.x))
        .y((d: any) => yMarginalScale(d.y));

      svgPosteriorG
        .selectAll("path.marginal")
        .data([
          d3.range(marginalLower, marginalUpper, 0.01).map((z) => {
            return { x: gaussian(z, marginalMu, marginalVar), y: z };
          }),
        ])
        .join("path")
        .attr("class", "marginal")
        .attr("d", (d) => marginalLine(d))
        .attr("fill", "none")
        .attr("stroke", "red")
        .attr("opacity", 0.8)
        .attr("stroke-width", 1.5);

      // Add trace line
      svgPosteriorG
        .selectAll("line.trace")
        .data([{ x: x0, y: y0 }])
        .join("line")
        .attr("class", "trace")
        .attr("x1", (d) => xScalePosterior(d.x))
        .attr("y1", (d) => yScalePosterior(marginalLower))
        .attr("x2", (d) => xScalePosterior(d.x))
        .attr("y2", (d) => yScalePosterior(marginalUpper))
        .attr("stroke", "red")
        .attr("stroke-width", 1.5)
        .attr("opacity", 0.8)
        .attr("stroke-dasharray", "5,5");
    });
    svgPosterior.on("mouseleave", () => {
      svgPosteriorG.selectAll("path.marginal").remove();
      svgPosteriorG.selectAll("line.trace").remove();
    });
  });

  function removePoint(index: number) {
    x_train = x_train.filter((_, i) => i !== index);
    y_train = y_train.filter((_, i) => i !== index);
  }

  function updateTrainFromCsv(csv: string) {
    const data = d3.csvParse(csv);
    x_train = data.map((d) => +d.x);
    y_train = data.map((d) => +d.y);
  }

  function changeKernelByName(name: string) {
    kernel = new Kernel(name, []);
  }
</script>

<svelte:head>
  <link
    rel="stylesheet"
    href="https://cdn.jsdelivr.net/npm/katex@0.15.2/dist/katex.min.css"
    integrity="sha384-MlJdn/WNKDGXveldHDdyRP1R4CTHr3FeuDNfhsLPYrq2t0UBkUdK2jyTnXPEK1NQ"
    crossorigin="anonymous"
  />
</svelte:head>

<main>
  <h1>Gaussian Processes Visual Tool</h1>

  <div class="viz">
    <svg id="posterior" width="100%" height="100%">
      <g id="posterior-g" transform="translate({padding}, {padding})">
        <g class="posterior-x-axis" />
        <g class="posterior-y-axis" />
      </g>
    </svg>

    <!-- <svg id="marginal" width="100%" height="100%" /> -->

    <div class="kernel-panel">
      <h3>Kernel options</h3>
      <!-- Dropdown to choose kernel -->
      <label for="kernel">Kernel</label>
      <select
        id="kernel"
        on:click={(event) => changeKernelByName(event.target.value)}
      >
        <option value="RBF">RBF</option>
        <!-- <option value="Matern">Matern</option> -->
        <option value="Periodic">Periodic</option>
        <option value="Linear">Linear</option>
        <option value="Polynomial">Polynomial</option>
      </select>

      <!-- Show math -->
      <div id="kernel-math">
        <Katex>{kernel.math}</Katex>
      </div>

      <!-- Input to kernel parameters -->
      <div id="kernel-params">
        {#each kernel.params as param, i}
          <label for="param{i}"><Katex>{kernel.params_names[i]}</Katex>: {param}</label>
          <input
            type="range"
            id="param{i}"
            bind:value={param}
            on:change={() => {
              kernel = new Kernel(kernel.name, kernel.params);
            }}
            min="0.01"
            max="5"
            step="0.01"
          />
        {/each}
      </div>
      <button
        on:click={() => {
          kernel = new Kernel(kernel.name, kernel.params);
        }}>Force Kernel Update</button
      >

      
    </div>

    <div class="simulation-panel">
      <h3>Simulation panel</h3>
      <!-- Button to reset x_train/y_train -->
      <button
        on:click={() => {
          x_train = [];
          y_train = [];
        }}>Clear observations</button
      >

      <!-- Input to x_test_start and x_test_end -->
      <div id="x-input">
        <label for="x_test_start">
          <Katex>x</Katex> start: {x_test_start}</label
        >
        <input
          type="range"
          id="x_test_start"
          bind:value={x_test_start}
          min="-20"
          max="0"
          step="0.1"
        />

        <label for="x_test_end"><Katex>x</Katex> end: {x_test_end}</label>
        <input
          type="range"
          id="x_test_end"
          bind:value={x_test_end}
          min="0"
          max="20"
          step="0.1"
        />
      </div>

      <!-- Input to amount of x_test_points -->
      <label for="x_test_points"
        >Amount of <Katex>x</Katex> points: {amount_test_points}</label
      >
      <input
        type="range"
        id="x_test_points"
        bind:value={amount_test_points}
        min="1"
        max="2000"
        step="1"
      />

      <!-- Choose likelihood sigma -->
      <label for="likelihood_sigma">Likelihood <Katex>\sigma</Katex>: {sigma}</label>
      <input
        type="range"
        id="likelihood_sigma"
        bind:value={sigma}
        min="0.01"
        max="1"
        step="0.01"
      />

      <!-- Upload your csv -->
      <label for="csv">Upload your csv</label>
      <input
        type="file"
        id="csv"
        accept=".csv"
        on:change={() => {
          const file = document.getElementById("csv").files[0];
          const reader = new FileReader();
          reader.onload = (e) => {
            updateTrainFromCsv(e.target.result);
            console.log(e.target.result);
          };
          reader.readAsText(file);
        }}
      />
    </div>

    <div class="description">
      By Eduardo Adame. Powered with D3.js, Svelte and (G)PyTorch.
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
    grid-template-rows: 3fr 1fr;
    gap: 1rem;
    // grid-row: 1fr 1fr;
  }

  // #x-input {
  //   display: flex;
  //   flex-direction: row;
  //   gap: 1rem;
  // }

  // #kernel-params {
  //   display: flex;
  //   flex-direction: row;
  //   gap: 1rem;
  // }

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
