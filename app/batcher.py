"""Dynamic batching engine.

Individual requests are submitted to a queue. A single background coroutine
drains the queue, forming a batch that is flushed when EITHER:

  * ``max_batch`` (32) items have accumulated, OR
  * ``window`` (10ms) has elapsed since the first item of the batch arrived,

whichever comes first. The batch is run through ``model.predict`` in one call,
and each caller's ``asyncio.Future`` is resolved with its own result.

This trades a few milliseconds of latency for much higher throughput: the model
processes many texts per forward pass instead of one at a time, which is the
whole point of batching CPU inference.
"""
from __future__ import annotations

import asyncio
import logging

from app import metrics, model

logger = logging.getLogger(__name__)

# Constraints fixed by the platform design.
WINDOW_SECONDS = 0.010  # 10ms batch window
MAX_BATCH = 32


class Batcher:
    """Collects single-text requests into batches for the model."""

    def __init__(self, window: float = WINDOW_SECONDS, max_batch: int = MAX_BATCH):
        self.window = window
        self.max_batch = max_batch
        self._queue: asyncio.Queue[tuple[str, asyncio.Future]] = asyncio.Queue()
        self._task: asyncio.Task | None = None

    async def submit(self, text: str) -> dict:
        """Submit one text and await its prediction result.

        Returns a ``{"label", "score"}`` dict once the batcher has processed it.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put((text, future))
        return await future

    def start(self) -> None:
        """Start the background batching loop."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="batcher")
            logger.info("Batcher started (window=%sms, max_batch=%d)",
                        self.window * 1000, self.max_batch)

    async def stop(self) -> None:
        """Stop the background batching loop and fail any pending futures."""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
            logger.info("Batcher stopped.")

    async def _collect_batch(self) -> list[tuple[str, asyncio.Future]]:
        """Block for the first item, then gather up to max_batch within window."""
        loop = asyncio.get_running_loop()

        # Block until at least one item is available — no busy-waiting.
        first = await self._queue.get()
        batch = [first]
        deadline = loop.time() + self.window

        while len(batch) < self.max_batch:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                batch.append(item)
            except TimeoutError:
                break

        return batch

    async def _run(self) -> None:
        """Background loop: collect a batch, run inference, resolve futures."""
        while True:
            batch = await self._collect_batch()
            texts = [text for text, _ in batch]
            futures = [fut for _, fut in batch]

            metrics.batch_size.observe(len(batch))

            try:
                # Model inference is blocking/CPU-bound — run it off the event
                # loop so we don't stall other coroutines while it computes.
                results = await asyncio.to_thread(model.predict, texts)
                for fut, result in zip(futures, results, strict=True):
                    if not fut.done():
                        fut.set_result(result)
            except Exception as exc:  # noqa: BLE001 — propagate to every caller
                logger.exception("Batch inference failed for %d items", len(batch))
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)
