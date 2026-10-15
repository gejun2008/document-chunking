# Chunking benchmark results

Metrics in %, higher is better. R@2k tok = recall when retrieving chunks up to a 2,000-token budget (fair across chunk sizes).

## chroma (5 docs, 472 queries)

| strategy | option | R@5 | P@5 | IoU@5 | R@1k tok | R@2k tok | chunks | avg tok | chunk s |
|---|---|---|---|---|---|---|---|---|---|
| fixed_size | selfbuilt | 89.9 | 3.1 | 3.1 | 74.5 | 90.1 | 1000 | 346 | 0.3 |
| fixed_size | selfbuilt_v2 | 90.5 | 3.2 | 3.2 | 76.2 | 91.5 | 1069 | 326 | 0.4 |
| fixed_size | selfbuilt_v3 | 91.5 | 3.8 | 3.8 | 83.9 | 93.7 | 1424 | 241 | 0.6 |
| fixed_size | chonkie | 91.5 | 3.1 | 3.1 | 75.6 | 91.5 | 963 | 369 | 0.3 |
| fixed_size | langchain | 92.3 | 3.7 | 3.7 | 84.5 | 93.1 | 1209 | 281 | 0.3 |
| layout_aware | selfbuilt | 90.7 | 3.1 | 3.1 | 76.0 | 91.3 | 1012 | 324 | 0.4 |
| layout_aware | chonkie | 91.9 | 3.6 | 3.6 | 82.8 | 94.6 | 1181 | 278 | 0.2 |
| layout_aware | langchain | 91.4 | 3.5 | 3.5 | 81.8 | 92.7 | 1144 | 286 | 0.8 |
| semantic | selfbuilt | 88.3 | 3.9 | 3.8 | 80.7 | 91.4 | 1284 | 255 | 98.6 |
| semantic | chonkie | 89.4 | 3.8 | 3.8 | 79.3 | 92.4 | 1351 | 243 | 125.3 |
| semantic | langchain | 88.3 | 3.9 | 3.8 | 80.7 | 91.4 | 1284 | 255 | 0.2 |

## qasper (80 docs, 226 queries)

| strategy | option | R@5 | P@5 | IoU@5 | R@1k tok | R@2k tok | chunks | avg tok | chunk s |
|---|---|---|---|---|---|---|---|---|---|
| fixed_size | selfbuilt | 68.1 | 7.0 | 6.9 | 43.5 | 69.2 | 1078 | 360 | 0.2 |
| fixed_size | selfbuilt_v2 | 70.8 | 8.0 | 7.8 | 49.2 | 74.0 | 1221 | 321 | 0.3 |
| fixed_size | selfbuilt_v3 | 74.0 | 8.0 | 7.8 | 53.1 | 78.6 | 1121 | 325 | 0.3 |
| fixed_size | chonkie | 68.9 | 7.1 | 7.0 | 42.7 | 69.4 | 1067 | 362 | 0.2 |
| fixed_size | langchain | 74.0 | 8.0 | 7.8 | 53.1 | 78.6 | 1121 | 325 | 0.2 |
| layout_aware | selfbuilt | 74.2 | 9.4 | 9.2 | 61.0 | 81.2 | 1317 | 270 | 0.3 |
| layout_aware | chonkie | 72.2 | 9.4 | 9.2 | 60.9 | 79.3 | 1315 | 271 | 0.2 |
| layout_aware | langchain | 69.5 | 10.0 | 9.7 | 61.8 | 82.6 | 1544 | 231 | 1.1 |
| semantic | selfbuilt | 64.5 | 8.7 | 8.4 | 55.5 | 76.7 | 1569 | 227 | 151.9 |
| semantic | chonkie | 63.4 | 8.0 | 7.8 | 48.8 | 71.5 | 1558 | 229 | 209.8 |
| semantic | langchain | 64.5 | 8.7 | 8.4 | 55.5 | 76.7 | 1569 | 227 | 0.2 |

## ohr (70 docs, 731 queries)

| strategy | option | R@5 | P@5 | IoU@5 | R@1k tok | R@2k tok | chunks | avg tok | chunk s |
|---|---|---|---|---|---|---|---|---|---|
| fixed_size | selfbuilt | 86.9 | 1.7 | 1.7 | 72.1 | 87.1 | 1277 | 365 | 0.2 |
| fixed_size | selfbuilt_v2 | 85.3 | 1.9 | 1.9 | 73.6 | 86.1 | 1407 | 334 | 0.4 |
| fixed_size | selfbuilt_v3 | 85.6 | 2.0 | 2.0 | 76.5 | 86.6 | 1504 | 294 | 0.4 |
| fixed_size | chonkie | 84.6 | 1.7 | 1.7 | 72.3 | 84.6 | 1268 | 366 | 0.2 |
| fixed_size | langchain | 85.4 | 2.0 | 2.0 | 76.3 | 86.6 | 1495 | 296 | 0.3 |
| layout_aware | selfbuilt | 84.5 | 1.8 | 1.8 | 72.8 | 84.9 | 1278 | 334 | 0.5 |
| layout_aware | chonkie | 85.0 | 2.0 | 2.0 | 76.5 | 86.6 | 1484 | 287 | 0.3 |
| layout_aware | langchain | 85.5 | 1.8 | 1.8 | 72.3 | 85.9 | 1251 | 342 | 1.2 |
| semantic | selfbuilt | 82.8 | 2.2 | 2.2 | 77.4 | 87.1 | 1888 | 226 | 159.2 |
| semantic | chonkie | 83.9 | 2.1 | 2.1 | 75.4 | 86.7 | 1935 | 221 | 210.1 |
| semantic | langchain | 82.8 | 2.2 | 2.2 | 77.4 | 87.1 | 1888 | 226 | 0.2 |

## Context enrichment (lite subsets, base = each option's fixed_size)

| dataset | option | R@5 base | R@5 enriched | Δ | R@2k base | R@2k enriched | Δ | queries |
|---|---|---|---|---|---|---|---|---|
| chroma-lite | selfbuilt | 96.0 | 95.2 | -0.8 | 96.0 | 95.2 | -0.8 | 276 |
| chroma-lite | chonkie | 97.2 | 96.3 | -0.9 | 97.2 | 96.3 | -0.9 | 276 |
| chroma-lite | langchain | 97.4 | 96.8 | -0.7 | 97.9 | 97.2 | -0.7 | 276 |
| qasper-lite | selfbuilt | 61.0 | 67.2 | +6.3 | 63.2 | 67.2 | +4.0 | 44 |
| ohr-lite | selfbuilt | 88.3 | 88.6 | +0.4 | 88.3 | 88.6 | +0.4 | 186 |

## Macro average R@2k tok over datasets

| strategy | selfbuilt | selfbuilt_v2 | selfbuilt_v3 | chonkie | langchain |
|---|---|---|---|---|---|
| fixed_size | 82.1 (n=3) | 83.9 (n=3) | 86.3 (n=3) | 81.8 (n=3) | 86.1 (n=3) |
| layout_aware | 85.8 (n=3) | - | - | 86.8 (n=3) | 87.1 (n=3) |
| semantic | 85.1 (n=3) | - | - | 83.5 (n=3) | 85.1 (n=3) |

## Chunk-size sweep (R@2k tok at chunk_size 200 / 300 / 400; avg tokens in brackets)

| dataset | strategy | option | 200 | 300 | 400 |
|---|---|---|---|---|---|
| chroma | fixed_size | selfbuilt | 93.2 (171) | 91.8 (257) | 90.1 (346) |
| chroma | fixed_size | selfbuilt_v2 | 94.6 (165) | 92.7 (247) | 91.5 (326) |
| chroma | fixed_size | selfbuilt_v3 | 95.3 (115) | 95.4 (178) | 93.7 (241) |
| chroma | fixed_size | chonkie | 95.4 (175) | 93.6 (271) | 91.5 (369) |
| chroma | fixed_size | langchain | 95.3 (137) | 95.4 (209) | 93.1 (281) |
| chroma | layout_aware | selfbuilt | 94.3 (167) | 92.4 (245) | 91.3 (324) |
| chroma | layout_aware | selfbuilt_v2 | - | - | - |
| chroma | layout_aware | selfbuilt_v3 | - | - | - |
| chroma | layout_aware | chonkie | 95.3 (137) | 94.1 (207) | 94.6 (278) |
| chroma | layout_aware | langchain | 95.5 (140) | 93.8 (211) | 92.7 (286) |
| qasper | fixed_size | selfbuilt | 76.6 (175) | 75.4 (269) | 69.2 (360) |
| qasper | fixed_size | selfbuilt_v2 | 78.0 (158) | 76.8 (238) | 74.0 (321) |
| qasper | fixed_size | selfbuilt_v3 | 82.2 (142) | 79.7 (230) | 78.6 (325) |
| qasper | fixed_size | chonkie | 77.5 (175) | 73.9 (268) | 69.4 (362) |
| qasper | fixed_size | langchain | 82.2 (141) | 79.7 (230) | 78.6 (325) |
| qasper | layout_aware | selfbuilt | 81.6 (140) | 83.2 (204) | 81.2 (270) |
| qasper | layout_aware | selfbuilt_v2 | - | - | - |
| qasper | layout_aware | selfbuilt_v3 | - | - | - |
| qasper | layout_aware | chonkie | 82.3 (130) | 82.6 (201) | 79.3 (271) |
| qasper | layout_aware | langchain | 82.2 (125) | 81.4 (184) | 82.6 (231) |
| ohr | fixed_size | selfbuilt | 88.8 (177) | 87.7 (272) | 87.1 (365) |
| ohr | fixed_size | selfbuilt_v2 | 89.2 (165) | 88.4 (250) | 86.1 (334) |
| ohr | fixed_size | selfbuilt_v3 | 89.4 (139) | 89.3 (215) | 86.6 (294) |
| ohr | fixed_size | chonkie | 89.9 (176) | 88.3 (272) | 84.6 (366) |
| ohr | fixed_size | langchain | 89.5 (141) | 89.3 (218) | 86.6 (296) |
| ohr | layout_aware | selfbuilt | 90.9 (166) | 87.7 (251) | 84.9 (334) |
| ohr | layout_aware | selfbuilt_v2 | - | - | - |
| ohr | layout_aware | selfbuilt_v3 | - | - | - |
| ohr | layout_aware | chonkie | 89.9 (140) | 87.7 (213) | 86.6 (287) |
| ohr | layout_aware | langchain | 90.3 (158) | 88.6 (250) | 85.9 (342) |
