# Axis Neuron BVH middleware

UDP de Axis Neuron entra, ZMQ sale, y hay dos formas de ver el esqueleto.

```
Axis Neuron --UDP--> main.py --ZMQ PUB tcp://*:5556 topic bvh--> tui/
                         |
                         +--> http://127.0.0.1:8080
```

El mensaje es multipart: `[b"bvh", json]` con `{"frame", "avatar", "t", "channels"}`.

## Middleware

```bash
pip install pyzmq
python main.py --src-ip 192.168.0.62 --udp-port 7001
```

Escucha UDP en `0.0.0.0:7001`, publica en `tcp://*:5556` y abre el viewer en `http://127.0.0.1:8080` (`/bvh`, `/skeleton`, `/latest`, `/status`, `/stream`).

Sin `--hierarchy archivo.bvh` usa el esqueleto dummy de Axis Neuron (59 joints). 354/360 canales llevan displacement; 180/186 son solo rotación. Seis canales de más son el bloque de referencia.

## TUI

```bash
cargo run --manifest-path tui/Cargo.toml --release
```

Se conecta a `tcp://127.0.0.1:5556`. El middleware ya hace bind; `--bind` es solo si el publisher hace connect.

```
flechas        orbitar
- / +          zoom
tab            joint
f              dedos
1 3 5 7        frente, lado, arriba, 3/4
espacio        pausa
r              reset de cámara
q              salir
```

`--hierarchy archivo.bvh` usa esa jerarquía. Si no, elige el dummy según el primer frame, igual que `main.py`.
