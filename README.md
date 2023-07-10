# Gaussian Processes Visual Tool

Powered by (G)PyTorch, Flask, D3.js and Svelte. Check the [related paper](https://github.com/adamesalles/gp-visual-tool/blob/main/paper.pdf) for more information.

By Eduardo Adame

## Instructions

### Installation

1. Clone this repository and `cd` to `src/`
2. Install the requirements for Python: `cd gp && pip install -r requirements.txt`
3. Install the requirements for Node: `cd ../web && npm install`

Even though you installed the requirements for Python, it may not work. It happens because of Torch. If you have problems, try to install it manually. You can find the instructions [here](https://pytorch.org/get-started/locally/).

Personally, I create a virtual environment with `conda` and install the requirements there. It works for me. Just using:

```bash
conda create -n gpvt &
conda activate gpvt &
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia &
conda install -c conda-forge pip &
pip install -r requirements.txt
```

### Run

1. `cd` to `src/`
2. Keep the server running: `cd gp/ && python server.py`
3. Run the client: `cd ../web/ && npm run dev`


Code is now kinda messy, but I'll try to clean it up soon. I'm also planning to add more features, like the possibility to change the kernel, the likelihood, the optimizer, etc.
