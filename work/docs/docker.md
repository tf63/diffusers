### nvidia-docker の設定など

**GPU の有効化**

- 前準備
- https://matsuand.github.io/docs.docker.jp.onthefly/config/containers/resource_constraints/#gpu

- nvidia-container-runtime をインストール

```
    sudo apt install nvidia-container-runtime
    which nvidia-contaier-runtime-hook # インストール確認
```

- デーモンの再起動

```
    sudo service docker restart
```

**コンテナの起動**

- `docker run`による起動
- `nvidia-smi`できるか確認

```
    docker run -it --rm --gpus all <イメージ名> nvidia-smi
```

**docker compose up で起動**

- GPU 接続したい Docker サービスで deploy を指定する
- https://matsuand.github.io/docs.docker.jp.onthefly/compose/gpu-support/

```
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
            - count: 1
            - capabilities: [gpu]
```

**pytorch で GPU を認識できるか確認**

```
  >>> import torch
  >>> print(torch.cuda.get_device_name())
  NVIDIA RTX A5000
```
