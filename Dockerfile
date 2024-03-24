FROM mambaorg/micromamba:1.5.6

WORKDIR /model

# Address warning about python debugger missing breakpoints
# due to frozen modules being used
ENV PYDEVD_DISABLE_FILE_VALIDATION=1

COPY --chown=$MAMBA_USER:$MAMBA_USER . .

RUN micromamba create -y -n claymodel --file conda-lock.yml && \
    micromamba clean --all --yes

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh"]

CMD ["jupyter-lab", "--ip", "0.0.0.0", "--port", "8888", "--no-browser"]
