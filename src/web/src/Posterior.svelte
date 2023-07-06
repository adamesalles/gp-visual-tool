<script lang="ts">
  import { mu, upper, lower } from "./stores.ts";
  import * as d3 from "d3";

  let width = 0;
  let height = 0;

  let posterior;
  const padding = 10;
</script>

<svg id="posterior" width="100%" height="100%">
  <g id="posterior-g" {width} {height}
  transform= "translate({padding}, {padding})"
  />
</svg>
