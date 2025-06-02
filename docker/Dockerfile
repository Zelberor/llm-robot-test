FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
  --mount=type=cache,target=/var/lib/apt,sharing=locked \
  apt-get update && apt-get install -y \
  python3.12 python3.12-venv python-is-python3 \
  sudo neovim nano iputils-ping

ADD docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ARG USER_UID=0
ARG USER_GID=0
RUN groupadd -o -g ${USER_GID} python-user
RUN useradd -m -o -u ${USER_UID} -g ${USER_GID} python-user
RUN echo python-user ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/python-user

USER python-user
RUN echo "source /workspace/venv/bin/activate" >> ~/.bashrc

ENTRYPOINT ["/entrypoint.sh"]
