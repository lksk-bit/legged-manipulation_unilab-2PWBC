"""Device-resident time-major replay; histories stop at episode boundaries."""

from __future__ import annotations

import torch


class Replay:
    def __init__(self, capacity, envs, device, seed, storage_device=None):
        self.capacity = capacity
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device or device)
        self.rng = torch.Generator(device=self.storage_device).manual_seed(seed)
        self.cursor = 0
        self.episode = torch.zeros(envs, device=self.storage_device, dtype=torch.long)
        self.data = {}

    def append(self, done, **batch):
        self.append_many(done.unsqueeze(0), **{k: v.unsqueeze(0) for k, v in batch.items()})

    def append_many(self, done, **batch):
        """Write a time chunk with at most two contiguous writes per field."""
        done = done.to(self.storage_device)
        increments = done.long()
        episodes = self.episode + increments.cumsum(dim=0) - increments
        batch.update(episode=episodes, done=done)
        length = len(done)
        keep = min(length, self.capacity)
        offset = (self.cursor + length - keep) % self.capacity
        head = min(keep, self.capacity - offset)
        for k, value in batch.items():
            if k not in self.data:
                self.data[k] = torch.empty(
                    (self.capacity, *value.shape[1:]), device=self.storage_device, dtype=value.dtype
                )
            value = value[-keep:].to(self.storage_device)
            self.data[k][offset : offset + head].copy_(value[:head])
            if head < keep:
                self.data[k][: keep - head].copy_(value[head:])
        self.episode += increments.sum(dim=0)
        self.cursor += length

    def sample(self, window, count, history=1, *, policy=None):
        if policy is None:
            policy = window == 1
        oldest = max(0, self.cursor - self.capacity)
        if self.cursor - oldest < window:
            return None

        device = self.storage_device
        start = torch.randint(
            oldest, self.cursor - window + 1, (count * 8,), generator=self.rng, device=device
        )
        env = torch.randint(len(self.episode), (count * 8,), generator=self.rng, device=device)
        ids = self.data["episode"]
        if window > 1:
            valid = (
                ids[start % self.capacity, env] == ids[(start + window - 1) % self.capacity, env]
            )
            # Keep the common path fixed-size on CUDA. Invalid candidates are
            # sorted behind valid ones, with no device-to-host boolean indexing.
            choices = valid.long().argsort(descending=True, stable=True)[:count]
            available = valid[choices].sum()
            if not available.item():
                return None
            slots = torch.arange(count, device=device)
            choices = choices[slots % available]
            start, env = start[choices], env[choices]
        else:
            start, env = start[:count], env[:count]
        slots = (start + torch.arange(window, device=device)[:, None]) % self.capacity
        keys = (
            ("state", "controller", "context", "elapsed", "command_state")
            if policy
            else ("state", "next_state", "target", "context")
        )
        result = {k: self.data[k][slots, env].to(self.device) for k in keys if k in self.data}
        if policy:
            prefix = start + torch.arange(-history, 0, device=device)[:, None]
            prev_slots = prefix.clamp_min(oldest) % self.capacity
            same = (prefix >= oldest) & (ids[prev_slots, env] == ids[start % self.capacity, env])
            for key in ("leg_frame", "arm_frame"):
                frames = self.data[key][prev_slots, env] * same.unsqueeze(-1)
                result[key] = frames.transpose(0, 1).to(self.device)
        return result
