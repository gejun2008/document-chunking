# Chunking benchmark results

Metrics in %, higher is better. R@2k tok = recall when retrieving chunks up to a 2,000-token budget (fair across chunk sizes).

## chroma (5 docs, 472 queries)

| strategy | option | R@5 | P@5 | IoU@5 | R@1k tok | R@2k tok | chunks | avg tok | chunk s |
|---|---|---|---|---|---|---|---|---|---|
| fixed_size | selfbuilt | 91.5 | 3.8 | 3.8 | 83.9 | 93.7 | 1424 | 241 | 0.6 |
| fixed_size | chonkie | 91.5 | 3.1 | 3.1 | 75.6 | 91.5 | 963 | 369 | 0.3 |
| fixed_size | langchain | 92.3 | 3.7 | 3.7 | 84.5 | 93.1 | 1209 | 281 | 0.3 |
| layout_aware | selfbuilt | 90.3 | 3.7 | 3.7 | 83.4 | 93.9 | 1372 | 239 | 0.4 |
| layout_aware | chonkie | 91.9 | 3.6 | 3.6 | 82.8 | 94.6 | 1181 | 278 | 0.2 |
| layout_aware | langchain | 91.4 | 3.5 | 3.5 | 81.8 | 92.7 | 1144 | 286 | 0.8 |
| semantic | selfbuilt | 88.3 | 3.9 | 3.8 | 80.7 | 91.4 | 1284 | 255 | 98.6 |
| semantic | chonkie | 89.4 | 3.8 | 3.8 | 79.3 | 92.4 | 1351 | 243 | 125.3 |
| semantic | langchain | 88.3 | 3.9 | 3.8 | 80.7 | 91.4 | 1284 | 255 | 0.2 |

## qasper (80 docs, 226 queries)

| strategy | option | R@5 | P@5 | IoU@5 | R@1k tok | R@2k tok | chunks | avg tok | chunk s |
|---|---|---|---|---|---|---|---|---|---|
| fixed_size | selfbuilt | 74.0 | 8.0 | 7.8 | 53.1 | 78.6 | 1121 | 325 | 0.3 |
| fixed_size | chonkie | 68.9 | 7.1 | 7.0 | 42.7 | 69.4 | 1067 | 362 | 0.2 |
| fixed_size | langchain | 74.0 | 8.0 | 7.8 | 53.1 | 78.6 | 1121 | 325 | 0.2 |
| layout_aware | selfbuilt | 74.2 | 9.4 | 9.2 | 61.0 | 81.2 | 1324 | 269 | 0.3 |
| layout_aware | chonkie | 72.2 | 9.4 | 9.2 | 60.9 | 79.3 | 1315 | 271 | 0.2 |
| layout_aware | langchain | 69.5 | 10.0 | 9.7 | 61.8 | 82.6 | 1544 | 231 | 1.1 |
| semantic | selfbuilt | 64.5 | 8.7 | 8.4 | 55.5 | 76.7 | 1569 | 227 | 151.9 |
| semantic | chonkie | 63.4 | 8.0 | 7.8 | 48.8 | 71.5 | 1558 | 229 | 209.8 |
| semantic | langchain | 64.5 | 8.7 | 8.4 | 55.5 | 76.7 | 1569 | 227 | 0.2 |

## ohr (70 docs, 731 queries)

| strategy | option | R@5 | P@5 | IoU@5 | R@1k tok | R@2k tok | chunks | avg tok | chunk s |
|---|---|---|---|---|---|---|---|---|---|
| fixed_size | selfbuilt | 85.6 | 2.0 | 2.0 | 76.5 | 86.6 | 1504 | 294 | 0.4 |
| fixed_size | chonkie | 84.6 | 1.7 | 1.7 | 72.3 | 84.6 | 1268 | 366 | 0.2 |
| fixed_size | langchain | 85.4 | 2.0 | 2.0 | 76.3 | 86.6 | 1495 | 296 | 0.3 |
| layout_aware | selfbuilt | 84.8 | 1.9 | 1.9 | 76.2 | 86.3 | 1479 | 289 | 0.5 |
| layout_aware | chonkie | 85.0 | 2.0 | 2.0 | 76.5 | 86.6 | 1484 | 287 | 0.3 |
| layout_aware | langchain | 85.5 | 1.8 | 1.8 | 72.3 | 85.9 | 1251 | 342 | 1.2 |
| semantic | selfbuilt | 82.8 | 2.2 | 2.2 | 77.4 | 87.1 | 1888 | 226 | 159.2 |
| semantic | chonkie | 83.9 | 2.1 | 2.1 | 75.4 | 86.7 | 1935 | 221 | 210.1 |
| semantic | langchain | 82.8 | 2.2 | 2.2 | 77.4 | 87.1 | 1888 | 226 | 0.2 |

## Context enrichment (lite subsets, base = each option's fixed_size)

| dataset | option | R@5 base | R@5 enriched | Δ | R@2k base | R@2k enriched | Δ | queries |
|---|---|---|---|---|---|---|---|---|
| chroma-lite | selfbuilt | 96.7 | 97.1 | +0.4 | 97.4 | 97.2 | -0.3 | 276 |
| chroma-lite | chonkie | 97.2 | 96.3 | -0.9 | 97.2 | 96.3 | -0.9 | 276 |
| chroma-lite | langchain | 97.4 | 96.8 | -0.7 | 97.9 | 97.2 | -0.7 | 276 |
| qasper-lite | selfbuilt | 75.1 | 75.6 | +0.5 | 77.4 | 80.2 | +2.7 | 44 |
| ohr-lite | selfbuilt | 84.4 | 88.0 | +3.6 | 84.9 | 90.1 | +5.2 | 186 |

## Macro average R@2k tok over datasets

| strategy | selfbuilt | chonkie | langchain |
|---|---|---|---|
| fixed_size | 86.3 (n=3) | 81.8 (n=3) | 86.1 (n=3) |
| layout_aware | 87.1 (n=3) | 86.8 (n=3) | 87.1 (n=3) |
| semantic | 85.1 (n=3) | 83.5 (n=3) | 85.1 (n=3) |

## Chunk-size sweep (R@2k tok at chunk_size 200 / 300 / 400; avg tokens in brackets)

| dataset | strategy | option | 200 | 300 | 400 |
|---|---|---|---|---|---|
| chroma | fixed_size | selfbuilt | 95.3 (115) | 95.4 (178) | 93.7 (241) |
| chroma | fixed_size | chonkie | 95.4 (175) | 93.6 (271) | 91.5 (369) |
| chroma | fixed_size | langchain | 95.3 (137) | 95.4 (209) | 93.1 (281) |
| chroma | layout_aware | selfbuilt | 94.9 (115) | 95.3 (175) | 93.9 (239) |
| chroma | layout_aware | chonkie | 95.3 (137) | 94.1 (207) | 94.6 (278) |
| chroma | layout_aware | langchain | 95.5 (140) | 93.8 (211) | 92.7 (286) |
| qasper | fixed_size | selfbuilt | 82.2 (142) | 79.7 (230) | 78.6 (325) |
| qasper | fixed_size | chonkie | 77.5 (175) | 73.9 (268) | 69.4 (362) |
| qasper | fixed_size | langchain | 82.2 (141) | 79.7 (230) | 78.6 (325) |
| qasper | layout_aware | selfbuilt | 81.8 (128) | 82.7 (198) | 81.2 (269) |
| qasper | layout_aware | chonkie | 82.3 (130) | 82.6 (201) | 79.3 (271) |
| qasper | layout_aware | langchain | 82.2 (125) | 81.4 (184) | 82.6 (231) |
| ohr | fixed_size | selfbuilt | 89.4 (139) | 89.3 (215) | 86.6 (294) |
| ohr | fixed_size | chonkie | 89.9 (176) | 88.3 (272) | 84.6 (366) |
| ohr | fixed_size | langchain | 89.5 (141) | 89.3 (218) | 86.6 (296) |
| ohr | layout_aware | selfbuilt | 89.3 (137) | 88.1 (211) | 86.3 (289) |
| ohr | layout_aware | chonkie | 89.9 (140) | 87.7 (213) | 86.6 (287) |
| ohr | layout_aware | langchain | 90.3 (158) | 88.6 (250) | 85.9 (342) |
