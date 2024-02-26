# FROM condaforge/miniforge3:latest

# WORKDIR /model

# COPY . .

# RUN mamba install -y -n base --file environment.yml

# #RUN mamba activate claymodel
# #RUN echo "mamba activate ${ENV_NAME}" >> ~/.bash_profile

# ENTRYPOINT [ "bash", "-l", "-c" ]

# CMD ["jupyter-lab", "--ip", "0.0.0.0", "--port", "8888", "--allow-root"]

FROM mambaorg/micromamba:1.5.6 

WORKDIR /model

COPY --chown=$MAMBA_USER:$MAMBA_USER . .

RUN micromamba create -y -n claymodel --file environment.yml && \
    micromamba clean --all --yes

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]
CMD ["jupyter-lab", "--ip", "0.0.0.0", "--port", "8888", "--allow-root"]