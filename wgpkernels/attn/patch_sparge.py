# Copyright (c) 2026 DeepBeepMeep - Apache License 2.0 (see LICENSE.txt)
"""Derive the WanGP VSA kernels from the vendored SpargeAttn sources (sparge/qattn/*.cuh, Apache-2.0):

- every key block carries a valid-token count (`Valid_Count`), masked in every iteration: partial 64-token tiles of a video grid
  attend only their real rows;
- the dispatcher takes the CUDA stream of the call.

Run once after refreshing the vendored sources; the patched headers are what attn.cu includes."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = {"qk_int_sv_f8_cuda_sm89.cuh": "vsa_sm89.cuh", "qk_int_sv_f16_cuda_sm80.cuh": "vsa_sm80.cuh"}
MASK = "apply_out_of_bound_mask<num_tiles_q, num_tiles_k>({base}, RS_f32, cur_block * CTA_K + Valid_Count[cur_block]);"


def replace_once(text, old, new, name):
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{name}: expected one occurrence of {old[:60]!r}, found {count}")
    return text.replace(old, new)


def patch(source, target):
    text = (HERE / "sparge" / "qattn" / source).read_text(encoding="utf-8")
    name = source
    text = text.replace('#include "attn_utils.cuh"', '#include "attn_utils.cuh"  // WanGP VSA variant: per-block valid counts, stream argument')
    text = text.replace("int32_t *__restrict__ Valid_Block_Num, float *__restrict__ PV_Threshold,",
                        "int32_t *__restrict__ Valid_Block_Num, int32_t *__restrict__ Valid_Count, float *__restrict__ PV_Threshold,")
    text = replace_once(text, "  uint32_t KV_block_increm = 0;", "  uint32_t KV_block_increm = 0;\n  uint32_t cur_block = 0;  // key block whose scores are in RS", name)
    text = replace_once(text, "  KV_block_increm = Lut[0];", "  KV_block_increm = Lut[0];\n  cur_block = KV_block_increm;", name)
    # threshold branch: the next K load (and K_idx_lane_base) was advanced before the softmax update
    text = replace_once(text, "      float local_max_diff = update_mo<",
                        "      " + MASK.format(base="K_idx_lane_base - KV_block_increm * CTA_K") + "\n      cur_block += KV_block_increm;\n      float local_max_diff = update_mo<", name)
    # plain branch: scores of cur_block, masked before the update; the next block is loaded afterwards
    plain = re.search(r"      update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, false, (?:true|false), false>\(RS_f32, RO, m, d, sm_scale\);", text)
    if plain is None:
        raise ValueError(f"{name}: plain update_mdo not found")
    text = text[:plain.start()] + "      " + MASK.format(base="K_idx_lane_base") + "\n" + text[plain.start():]
    text = replace_once(text, "      // load K\n      KV_block_increm = Lut[iter];", "      // load K\n      KV_block_increm = Lut[iter];\n      cur_block += KV_block_increm;", name)
    # second last iteration: K_idx_lane_base still addresses the scored block until it is advanced
    second = re.search(r"    K_idx_lane_base \+= KV_block_increm \* CTA_K;\n\n    update_mdo<", text)
    if second is None:
        raise ValueError(f"{name}: second-last update not found")
    text = text[:second.start()] + "    " + MASK.format(base="K_idx_lane_base") + "\n    cur_block += KV_block_increm;\n" + text[second.start():]
    text = replace_once(text, "apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, kv_len);",
                        "apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, RS_f32, min(kv_len, cur_block * CTA_K + Valid_Count[cur_block]));", name)
    # dispatcher: valid counts and the stream
    text, count = re.subn(r"  float sm_scale\)\s*\n\{\n  constexpr MaskMode mask_mode", "  float sm_scale, cudaStream_t stream)\n{\n  constexpr MaskMode mask_mode", text)
    if count != 1:
        raise ValueError(f"{name}: dispatcher signature not found")
    text = replace_once(text, "<<<grid, block, smem_max>>>(", "<<<grid, block, smem_max, stream>>>(", name)
    text = replace_once(text, "    Valid_Block_Num,\n    PV_Threshold,", "    Valid_Block_Num,\n    Valid_Count,\n    PV_Threshold,", name)
    (HERE / "sparge" / "qattn" / target).write_text(text, encoding="utf-8")
    print(f"wrote {target}")


for source, target in SOURCES.items():
    patch(source, target)
