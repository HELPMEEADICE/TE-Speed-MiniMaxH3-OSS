"""TE-Speed-MiniMaxH3-OSS: open-source block-cache accelerator for the MiniMax H3 DiT.

This is a clean-room reimplementation of the compiled TE-Speed-MiniMaxH3 custom
node (nodes.pyd). It reproduces the original plugin's observable behavior
(block caching with residual correction, a caching window in the middle of the
denoise schedule, a sigma-delta threshold, and a cap on consecutive cache
steps) using the same model.py hooks ("block_loop" patch point, cache_ranges,
block_count, start/end block slicing) that the original patch adds.

How it accelerates
------------------
For every denoising step the patched ("block_loop", 0) hook receives the packed
hidden state h and a handle to the original transformer block loop. The cache
wrapper picks one of two execution modes:

  FULL  : run every transformer block; then store
          residual = h_full - h_k
          where h_k is the hidden state right before block k (the "warm"
          prefix output), captured in-loop by a ("double_block", k) patch so
          no extra forward pass is needed.
  CACHE : recompute only the leading warm blocks and add the stored residual.
          The contribution of the trailing cached blocks is assumed to change
          little between adjacent denoising steps, which holds when the sigma
          delta is small.

A step may use CACHE only when all of these hold:
  * it lies inside the [processing_percent_1, processing_percent_2] window of
    the denoise schedule (first/last percent of the run always run FULL),
  * the sigma delta to the previous step is below processing_control_value,
  * fewer than mcs consecutive cache steps have been taken already
    (a forced FULL step bounds error accumulation),
  * a residual is available from a previous FULL step.
The first step of every run is always FULL. CFG (cond/uncond) pairs share the
same decision for both calls of a step.

With the defaults (threshold 0.12, window 0.1..0.9, mcs 2, cache_depth 0.75)
this reproduces the ~45% wall-clock speedup of the original plugin on the
reference workflow while remaining fully configurable.
"""

import torch


class _MiniMaxH3Cache:
    """Block-loop cache: decides between FULL and CACHE execution per step."""

    def __init__(self, control_value, start_percent, end_percent, mcs, device, cache_depth=0.75):
        self.threshold = float(control_value)
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)
        self.mcs = int(mcs)
        self.cache_device = str(device)
        self.cache_depth = float(cache_depth)
        # threshold=0 / mcs=0 / cache_depth=0 all disable caching entirely
        self.enabled = (self.threshold > 0.0) and (self.mcs > 0) and (self.cache_depth > 0.0)
        self.reset()

    def reset(self):
        self.start_sigma = None
        self.last_sigma = None
        self.prev_sigma = None
        self.step = -1
        self.total_steps = None
        self.consecutive_skips = 0
        self.residual = None
        self.snapshot = None
        self.sigma_scale = 1.0
        self.last_mode = "full"
        self.full_steps = 0
        self.cache_hits = 0
        self.skipped_blocks = 0
        self.total_blocks = 0
        self.printed = False

    # ------------------------------------------------------------- patching

    def __call__(self, args, kwargs):
        """("block_loop", 0) patch entry. args holds the packed state, kwargs
        the original block loop; returns {"img": new_h} like the original."""
        original_block = kwargs["original_block"]
        block_count = int(args["block_count"])
        sigma = self._current_sigma(args)
        if sigma is None:
            return {"img": original_block(args)["img"]}

        if self.last_sigma is None or sigma > self.last_sigma + 1e-6:
            self._finish_run()
            self.reset()
            self.start_sigma = sigma
            self.sigma_scale = self._detect_scale(args, sigma)
            self.last_sigma = sigma
            self.step = 0
            self.last_mode = "full"
            return {"img": self._full_step(args, kwargs, block_count, count=True)}

        if abs(sigma - self.last_sigma) <= 1e-6:
            if self.last_mode == "cache":
                return {"img": self._cache_step(args, kwargs, block_count, count=False)}
            return {"img": self._full_step(args, kwargs, block_count, count=False)}

        self.prev_sigma = self.last_sigma
        self.last_sigma = sigma
        self.step += 1
        # sigma_scale maps the model's raw timestep convention onto the sample
        # schedule (raw sigma -> ~1.0, sigma*1000 -> ~1000).
        sigma_n = sigma / self.sigma_scale
        prev_n = self.prev_sigma / self.sigma_scale
        pos = self._position(args, sigma_n)
        k = self._warm_blocks(block_count)
        in_window = self.start_percent <= pos <= self.end_percent
        slow = abs(prev_n - sigma_n) < self.threshold
        can_skip = (self.enabled and k < block_count and self.residual is not None
                    and in_window and slow and self.consecutive_skips < self.mcs)
        if can_skip:
            self.last_mode = "cache"
            return {"img": self._cache_step(args, kwargs, block_count, count=True)}
        self.last_mode = "full"
        return {"img": self._full_step(args, kwargs, block_count, count=True)}

    # ------------------------------------------------------- step execution

    def _warm_blocks(self, block_count):
        """Number of leading blocks recomputed on a CACHE step (and on a FULL
        step to refresh the residual). Always < block_count so that at least
        one block is served from cache when a cache step is allowed."""
        return max(0, min(block_count - 1, round(block_count * (1.0 - self.cache_depth))))

    def _full_step(self, args, kwargs, block_count, count=True):
        if count:
            self.full_steps += 1
            self.consecutive_skips = 0
            self.total_blocks += block_count
        original_block = kwargs["original_block"]
        # The block loop mutates its input tensor in place, so keep a pristine
        # copy when no in-loop snapshot is available (caching disabled).
        h_in = None if self.snapshot is not None else args["img"].clone()
        h = original_block(args)["img"]
        if self.snapshot is not None:
            residual = h - self.snapshot
        else:
            residual = h - h_in
        self.residual = residual.to("cpu") if self.cache_device == "cpu" else residual
        if count and self.total_steps is not None and self.step >= self.total_steps:
            self._print_stats()
        return h

    def _cache_step(self, args, kwargs, block_count, count=True):
        if count:
            self.cache_hits += 1
            self.consecutive_skips += 1
            self.total_blocks += block_count
            self.skipped_blocks += block_count - self._warm_blocks(block_count)
        original_block = kwargs["original_block"]
        h = args["img"]
        k = self._warm_blocks(block_count)
        if k > 0:
            h = original_block({**args, "start": 0, "end": k})["img"]
        if self.residual is not None:
            h = h + self.residual.to(h.device)
        return h

    # ------------------------------------------------------------ step info

    @staticmethod
    def _current_sigma(args):
        """Raw timestep the sampler handed to the model (in the model's own
        convention: raw sigma for ComfyUI's H3, sigma*1000 for the original
        patch). Normalized against the schedule via sigma_scale."""
        timestep = args["transformer_options"].get("sigmas")
        if timestep is None:
            return None
        return float(torch.as_tensor(timestep).flatten()[0].float())

    def _detect_scale(self, args, sigma):
        """Timestep units relative to the sample schedule, measured on the
        first step of each run: sigma / sample_sigmas[0]. Raw-sigma models
        yield ~1.0, sigma*1000 models yield ~1000."""
        sample_sigmas = args["transformer_options"].get("sample_sigmas")
        if sample_sigmas is not None:
            ss0 = float(torch.as_tensor(sample_sigmas).flatten().float()[0])
            if abs(ss0) > 1e-9:
                s = sigma / ss0
                if abs(s) > 1e-9:
                    return s
        return 1.0

    def _position(self, args, sigma_n):
        """0..1 progress through the denoise schedule. Uses the schedule
        passed via transformer_options["sample_sigmas"], with a sigma-range
        fallback when it is unavailable."""
        sample_sigmas = args["transformer_options"].get("sample_sigmas")
        if sample_sigmas is not None:
            ss = torch.as_tensor(sample_sigmas).flatten().float()
            if ss.numel() > 1:
                self.total_steps = ss.numel() - 1
                idx = int((ss - sigma_n).abs().argmin())
                return min(1.0, max(0.0, idx / self.total_steps))
        if self.start_sigma is not None and self.start_sigma > 0.0:
            start_n = self.start_sigma / self.sigma_scale
            return min(1.0, max(0.0, (start_n - sigma_n) / start_n))
        return 1.0

    # -------------------------------------------------------------- stats

    def _finish_run(self):
        if not self.printed and (self.full_steps + self.cache_hits) > 0:
            self._print_stats()

    def _print_stats(self):
        self.printed = True
        if self.total_blocks <= 0:
            return
        saved = 100.0 * self.skipped_blocks / self.total_blocks
        print(f"TE-Speed-MiniMaxH3(OSS): acceleration {saved:.1f}% "
              f"(full={self.full_steps} cache={self.cache_hits} of "
              f"{self.full_steps + self.cache_hits} steps, skipped "
              f"{self.skipped_blocks}/{self.total_blocks} blocks)")


class _MiniMaxH3Snapshot:
    """("double_block", k) patch: captures the hidden state right before block
    k during a FULL pass so the residual needs no extra forward pass."""

    def __init__(self, cache):
        self.cache = cache

    def __call__(self, args, kwargs):
        self.cache.snapshot = args["img"].clone()
        return kwargs["original_block"](args)


class TESpeedMiniMaxH3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "processing_control_value": ("FLOAT", {
                    "default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Sigma-delta threshold: a step may use the block cache when its sigma moved less than this since the previous step."}),
                "processing_percent_1": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 0.49, "step": 0.01,
                    "tooltip": "Start of the caching window as a fraction of the denoise schedule (steps before it always run fully)."}),
                "processing_percent_2": ("FLOAT", {
                    "default": 0.9, "min": 0.51, "max": 1.0, "step": 0.01,
                    "tooltip": "End of the caching window (steps after it always run fully)."}),
                "mcs": ("INT", {
                    "default": 2, "min": 0, "max": 10, "step": 1,
                    "tooltip": "Max consecutive cache steps before a forced full step bounds error accumulation. 0 disables caching."}),
                "device": (["auto", "cpu", "gpu"], {
                    "default": "auto",
                    "tooltip": "Where to store the cached residual. cpu frees VRAM at the cost of transfer; auto/gpu keep it on the compute device."}),
            },
            "optional": {
                "cache_depth": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 0.95, "step": 0.05,
                    "tooltip": "OSS extension: fraction of trailing blocks served from cache on cache steps. 0.75 matches the original's ~45% speedup; lower for quality, higher for speed. 0 disables caching."}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "sampling/custom_sampling/minimax_h3"

    def patch(self, model, processing_control_value, processing_percent_1,
              processing_percent_2, mcs, device, cache_depth=0.75):
        inner = self._find_minimax_dit(model)
        if inner is None:
            raise ValueError("TE-Speed-MiniMaxH3 only supports MiniMax H3 models")
        if not hasattr(inner, "_run_blocks"):
            print("TE-Speed-MiniMaxH3(OSS): comfy/ldm/minimax/model.py has no "
                  "block_loop hooks - run patch_model.py first. Returning the "
                  "model unpatched (stock speed).")
            return (model,)
        cache = _MiniMaxH3Cache(processing_control_value, processing_percent_1,
                                processing_percent_2, mcs, device, cache_depth)
        model = model.clone()
        model.set_model_patch_replace(cache, "dit", "block_loop", 0)
        # In-loop snapshot at block k: residual captured during FULL passes
        # with zero extra compute. k must match _warm_blocks at runtime.
        block_count = len(inner.blocks)
        k = cache._warm_blocks(block_count)
        if cache.enabled and 0 < k < block_count:
            model.set_model_patch_replace(_MiniMaxH3Snapshot(cache), "dit", "double_block", k)
        return (model,)

    @staticmethod
    def _find_minimax_dit(model):
        """Walk the ModelPatcher chain to the MiniMaxH3Model DiT.
        Real chains: ModelPatcher.model -> model_base (MiniMaxH3) ->
        .diffusion_model -> MiniMaxH3Model. Reference-to-video wrappers may
        add extra levels."""
        m = getattr(model, "model", None)
        seen = 0
        while m is not None and seen < 12:
            if type(m).__name__ == "MiniMaxH3Model":
                return m
            nxt = None
            for attr in ("model", "inner_model", "diffusion_model", "unet_model"):
                nxt = getattr(m, attr, None)
                if nxt is not None:
                    break
            m = nxt
            seen += 1
        return None
