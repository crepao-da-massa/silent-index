# Silent Index

C++ service with a compact quantized vector index and a tiny FD-passing load
balancer.

## Estratégia

- Build offline lê `references.json.gz` oficial.
- Os vetores são quantizados em `int16` com escala `10000`, preservando os
  valores `round4` da especificação.
- O builder treina um IVF com K-means sobre amostra das referências e grava um
  `index.bin` clusterizado.
- A API faz parse manual do payload, monta o vetor de 14 dimensões e executa a
  busca nos clusters candidatos com reparo por bounding box.
- O load balancer é `fd-lb`, um processo C mínimo que aceita TCP em `:9999`,
  faz round-robin e passa o socket aceito para as APIs via `SCM_RIGHTS`.
- O LB não faz proxy de bytes e não inspeciona payload.

## Rodar

```sh
cp /path/to/references.json.gz bench/references.json.gz
make docker-up
curl http://localhost:9999/ready
```

Os arquivos de dados locais em `bench/` e os artefatos em `build/` são ignorados
pelo Git.

## Interface

- `GET /ready`
- `POST /fraud-score`
- Porta `9999`
- 1 load balancer + 2 instâncias da API
- `docker-compose.yml` na raiz
- Limite declarado: `1.0 CPU` e `350 MB`
- Sem lookup dos payloads de teste
