# Benchmarks the four checkpoints in api_data/image_model.py's MODEL_CHOICES
# (see run_benchmark.py for prompts, seeds, and metrics). Needs CUDA at run
# time to cover all four models - flux2-klein's fp8 checkpoint won't load on
# CPU - so this builds on an NVIDIA CUDA base rather than a plain Python one.
#
# Build:
#   docker build -t reffusion-benchmark benchmark/
#
# Run (full 5-prompt x 100-seed sweep over all four models):
#   docker run --gpus all \
#       -v $(pwd)/benchmark_output:/output \
#       -v $(pwd)/hf_cache:/cache/huggingface \
#       reffusion-benchmark
#
# Quick smoke test (few seeds, one model):
#   docker run --gpus all \
#       -v $(pwd)/benchmark_output:/output \
#       -v $(pwd)/hf_cache:/cache/huggingface \
#       reffusion-benchmark --models sd-turbo --seeds-per-prompt 3

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip curl \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --no-cache-dir --upgrade pip

WORKDIR /benchmark

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# LAION's improved aesthetic predictor checkpoint (~6MB) - a small MLP
# trained on CLIP ViT-L/14 embeddings, loaded by AestheticPredictor in
# run_benchmark.py. Public weight file, fetched once at build time so it's
# baked into the image rather than needed at run time.
RUN curl -fL -o /benchmark/aesthetic-predictor.pth \
    "https://raw.githubusercontent.com/christophschuhmann/improved-aesthetic-predictor/main/sac+logos+ava1-l14-linearMSE.pth"

# lpips.LPIPS(net='alex') otherwise fetches its backbone weights on first
# use; pre-warming at build time means a run never needs network access for
# it, only for the (much larger, cache-mounted) diffusion/CLIP checkpoints.
RUN python3 -c "import lpips; lpips.LPIPS(net='alex')"

COPY run_benchmark.py .

# Mount volumes over these so weights and results survive across runs:
#   /cache/huggingface - diffusion + CLIP checkpoints (multi-GB, else
#                        every run re-downloads all four models)
#   /output             - generated images, per_image_results.csv,
#                        diversity_results.csv, summary.json
ENV HF_HOME=/cache/huggingface
ENV OUTPUT_DIR=/output
VOLUME ["/cache/huggingface", "/output"]

ENTRYPOINT ["python3", "run_benchmark.py", "--output-dir", "/output"]
