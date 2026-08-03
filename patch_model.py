#!/usr/bin/env python
"""Install the MiniMax H3 "block_loop" hooks into ComfyUI's model file.

The TE-Speed-MiniMaxH3-OSS node accelerates the MiniMax H3 DiT by patching
the ("block_loop", 0) model hook. Stock ComfyUI's comfy/ldm/minimax/model.py
does not expose that hook yet, so this script surgically adds it (same hooks
the original TE-Speed patch ships):

  1. a _run_blocks(start, end) method (block-loop execution with slicing), and
  2. a ("block_loop", 0) branch in the model forward that hands the loop to a
     custom node when one is installed.

Usage:
    python patch_model.py                      # auto-locate ComfyUI and patch
    python patch_model.py --comfy-ui <dir>     # explicit ComfyUI root
    python patch_model.py --revert             # restore the .bak backup
    python patch_model.py --check              # only report the hook status

A backup is written next to the file (model.py.te_speed.bak) before patching.
"""

import argparse
import ast
import re
import shutil
import sys
from pathlib import Path

TARGET = Path("comfy/ldm/minimax/model.py")
BACKUP_SUFFIX = ".te_speed.bak"

COMMON_ROOTS = [
    Path("."),
    Path(".."),
    Path("ComfyUI"),
    Path("../ComfyUI"),
    Path("G:/ComfyUI-aki-v3/ComfyUI"),
    Path("C:/ComfyUI"),
    Path("D:/ComfyUI"),
    Path("E:/ComfyUI"),
    Path("C:/Users/Administrator/ComfyUI"),
]

RUN_BLOCKS_METHOD = '''    def _run_blocks(self, h, t_emb, mod_segments, rope_freqs, transformer_options, start=0, end=None):
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        end = len(self.blocks) if end is None else end
        prefetch_queue = comfy.model_prefetch.make_prefetch_queue(list(self.blocks[start:end]), h.device, transformer_options)
        for i in range(start, end):
            block = self.blocks[i]
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, h.device, block)
            if ("double_block", i) in blocks_replace:
                def block_wrap(args):
                    return {"img": block(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                         transformer_options=args["transformer_options"])}
                h = blocks_replace[("double_block", i)](
                    {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                     "transformer_options": transformer_options},
                    {"original_block": block_wrap})["img"]
            else:
                h = block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)
        if prefetch_queue is not None:
            comfy.model_prefetch.prefetch_queue_pop(prefetch_queue, h.device, None)
        return h

'''

HOOK_LOOP = '''        # blocks (TE-Speed-MiniMaxH3-OSS hook)
        patches_replace = transformer_options.get("patches_replace", {})
        blocks_replace = patches_replace.get("dit", {})
        cache_ranges = [(a, b) for a, b, kind in layout.segments if kind in ("audio", "video")]
        if ("block_loop", 0) in blocks_replace:
            def block_loop_wrap(args):
                return {"img": self._run_blocks(args["img"], args["t_emb"], args["mod_segments"], args["rope_freqs"],
                                                args["transformer_options"], args.get("start", 0), args.get("end"))}
            h = blocks_replace[("block_loop", 0)](
                {"img": h, "t_emb": t_emb, "mod_segments": mod_segments, "rope_freqs": rope_freqs,
                 "transformer_options": transformer_options, "cache_ranges": cache_ranges, "block_count": len(self.blocks)},
                {"original_block": block_loop_wrap})["img"]
        else:
            h = self._run_blocks(h, t_emb, mod_segments, rope_freqs, transformer_options)
'''

# The stock block loop, from `patches_replace = ...` through the final
# prefetch_queue_pop call. Anchored on the two stable end lines.
LOOP_RE = re.compile(
    r'        patches_replace = transformer_options\.get\("patches_replace", \{\}\)\n'
    r'        blocks_replace = patches_replace\.get\("dit", \{\}\)\n'
    r'        prefetch_queue = comfy\.model_prefetch\.make_prefetch_queue\('
    r'list\(self\.blocks\), device, transformer_options\)\n'
    r'.*?'
    r'        if prefetch_queue is not None:\n'
    r'            comfy\.model_prefetch\.prefetch_queue_pop\(prefetch_queue, device, None\)\n',
    re.DOTALL)

FORWARD_ANCHOR = "    def forward(self, x, timestep, context,"
RUN_BLOCKS_ANCHOR = "    def _run_blocks(self, h, t_emb, mod_segments, rope_freqs, transformer_options, start=0, end=None):"


def find_model_file(comfy_ui=None):
    if comfy_ui is not None:
        p = Path(comfy_ui)
        if not p.is_dir():
            p = Path(comfy_ui) / "ComfyUI"
        for candidate in (p / TARGET, p / "ComfyUI" / TARGET):
            if candidate.is_file():
                return candidate
        raise SystemExit(f"error: no model file at {p / TARGET}")
    for root in COMMON_ROOTS:
        for candidate in (root / TARGET, root / "ComfyUI" / TARGET):
            if candidate.is_file():
                return candidate
    raise SystemExit(
        "error: could not locate ComfyUI. Pass --comfy-ui <ComfyUI root dir>, "
        "e.g. --comfy-ui G:/ComfyUI-aki-v3/ComfyUI")


def hook_status(text):
    has_run_blocks = RUN_BLOCKS_ANCHOR in text
    has_loop_hook = '("block_loop", 0) in blocks_replace' in text
    return has_run_blocks and has_loop_hook


def apply_patch(text):
    if not LOOP_RE.search(text):
        raise SystemExit(
            "error: could not find the stock block loop in the model file. "
            "This ComfyUI version may differ; report it or patch manually.")
    text = LOOP_RE.sub(HOOK_LOOP, text, count=1)
    if RUN_BLOCKS_ANCHOR not in text:
        if FORWARD_ANCHOR not in text:
            raise SystemExit("error: could not find the MiniMaxH3Model.forward anchor.")
        text = text.replace(FORWARD_ANCHOR, RUN_BLOCKS_METHOD + FORWARD_ANCHOR, 1)
    return text


def main():
    parser = argparse.ArgumentParser(description="Install MiniMax H3 block_loop hooks into ComfyUI's minimax model.py")
    parser.add_argument("--comfy-ui", help="ComfyUI root directory (auto-detected when omitted)")
    parser.add_argument("--revert", action="store_true", help="restore the pre-patch backup")
    parser.add_argument("--check", action="store_true", help="only report hook status")
    args = parser.parse_args()

    target = find_model_file(args.comfy_ui)
    text = target.read_text(encoding="utf-8")

    if args.revert:
        backup = target.with_name(target.name + BACKUP_SUFFIX)
        if not backup.is_file():
            raise SystemExit(f"error: no backup at {backup}")
        shutil.copy2(backup, target)
        print(f"TE-Speed-MiniMaxH3-OSS: restored {target} from backup.")
        return

    if hook_status(text):
        print(f"TE-Speed-MiniMaxH3-OSS: hooks already present in {target}")
        return

    if args.check:
        print(f"TE-Speed-MiniMaxH3-OSS: hooks MISSING in {target}")
        return

    backup = target.with_name(target.name + BACKUP_SUFFIX)
    if not backup.is_file():
        shutil.copy2(target, backup)
        print(f"TE-Speed-MiniMaxH3-OSS: backup written to {backup}")

    patched = apply_patch(text)
    ast.parse(patched)  # syntax gate
    assert hook_status(patched)
    target.write_text(patched, encoding="utf-8")
    print(f"TE-Speed-MiniMaxH3-OSS: patched {target}")
    print("Restart ComfyUI (or reload custom nodes), then use the "
          "TESpeedMiniMaxH3 node. Revert any time with --revert.")


if __name__ == "__main__":
    main()
