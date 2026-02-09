FROM ubuntu:24.04
WORKDIR /root

RUN set -eux; \
    apt_mirror='http://mirrors.tuna.tsinghua.edu.cn/ubuntu'; \
    if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
        sed -i "s|http://archive.ubuntu.com/ubuntu|${apt_mirror}|g; s|http://security.ubuntu.com/ubuntu|${apt_mirror}|g; s|https://archive.ubuntu.com/ubuntu|${apt_mirror}|g; s|https://security.ubuntu.com/ubuntu|${apt_mirror}|g" /etc/apt/sources.list.d/ubuntu.sources; \
    elif [ -f /etc/apt/sources.list ]; then \
        sed -i "s|http://archive.ubuntu.com/ubuntu|${apt_mirror}|g; s|http://security.ubuntu.com/ubuntu|${apt_mirror}|g; s|https://archive.ubuntu.com/ubuntu|${apt_mirror}|g; s|https://security.ubuntu.com/ubuntu|${apt_mirror}|g" /etc/apt/sources.list; \
    fi; \
    apt-get update || { \
        echo 'mirror update failed, falling back to official ubuntu sources'; \
        if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
            sed -i 's|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|http://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list.d/ubuntu.sources; \
        elif [ -f /etc/apt/sources.list ]; then \
            sed -i 's|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|http://archive.ubuntu.com/ubuntu|g' /etc/apt/sources.list; \
        fi; \
        apt-get update; \
    }; \
    apt-get install -y --no-install-recommends python3 python3-pip; \
    rm -rf /var/lib/apt/lists/*

COPY src/requirements.txt src/
RUN pip install --break-system-packages --no-cache-dir --retries 15 --timeout 180 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --trusted-host pypi.tuna.tsinghua.edu.cn \
    -r src/requirements.txt
COPY src/ src/

WORKDIR /root/src

# workaround for https://github.com/BerriAI/litellm/issues/19852
ENV TIKTOKEN_CACHE_DIR=/usr/local/lib/python3.12/dist-packages/litellm/litellm_core_utils/tokenizers

# for quicker debugging
ENV MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT=0

# suppress v2.0 deprecation warning
ENV MSWEA_SILENT_STARTUP=1

ENTRYPOINT [ "python3", "-u", "main.py" ]
