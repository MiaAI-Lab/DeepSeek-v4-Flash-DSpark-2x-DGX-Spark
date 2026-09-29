FROM docker.m.daocloud.io/vllm/vllm-openai:v0.28.0
LABEL org.opencontainers.image.version="0.28.0" \
      org.opencontainers.image.description="DGX3/4 DeepSeek V4 Flash 0731 on GB10"
RUN pip install --no-cache-dir flashinfer-python==0.6.18 flashinfer-cubin==0.6.18 \
    --index-url https://flashinfer.ai/whl --extra-index-url https://pypi.org/simple
RUN pip install --no-cache-dir flashinfer-jit-cache==0.6.18 \
    --index-url https://flashinfer.ai/whl/cu130
RUN python3 -c 'import importlib.metadata as m; assert m.version("vllm") == "0.28.0"; print(m.version("vllm"), m.version("flashinfer-python"))'
