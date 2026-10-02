"""Adaptive arithmetic coder for quantized latents, built on `constriction`.

Why a library
-------------
An earlier hand-written rANS/E1E2E3 implementation desynchronised and hung. A
subtly incorrect entropy coder is the worst thing that could be published in a
BPS table, so this uses `constriction` (a tested, production stream-coding
library) and verifies a full encode -> decode round trip on every trace.

Why not a general-purpose byte compressor
-----------------------------------------
LZMA/bz2 were measured and rejected: on a dense quantized latent they save only
~12% over fixed-width at large code lengths, and the container header dominates
entirely for short latents. They model repeated byte patterns, not symbol
statistics, so they are the wrong tool and would understate how compressible the
latent is. A context-modelling coder is required.

Reported quantities, per trace
-------------------------------
* `fixed_bits`    - b bits per symbol, no compression. Baseline.
* `entropy_bound` - zeroth-order empirical entropy. Theoretical LOWER bound.
* `coded_bits`    - measured rate of the adaptive arithmetic coder. This is the
                    headline number, and it always sits above `entropy_bound`.
                    A byte-aligned container rounds this up by at most 7 bits.

The model is adaptive and driven only by the symbols of that trace, reset per
trace, so no frequency table is transmitted. The decoder replays the same
symbols and therefore tracks an identical model; `get_decoder()` builds the
decoder from the encoder's real final state and the round trip is asserted.
"""

import numpy as np
import constriction

_Cat = constriction.stream.model.Categorical
_q = constriction.stream.queue


def _probs(counts):
    return counts / counts.sum()


class AdaptiveCoder:
    def __init__(self, alphabet, perfect=True):
        self.A = alphabet
        self.perfect = perfect

    def encode(self, seq):
        """seq: 1-D array of symbols in [0, A). Returns (num_bits, decoder)."""
        counts = np.ones(self.A)
        enc = _q.RangeEncoder()
        for s in seq:
            enc.encode(int(s), _Cat(_probs(counts), perfect=self.perfect))
            counts[int(s)] += 1
        return enc.num_bits(), enc.get_decoder()

    def decode(self, decoder, n):
        counts = np.ones(self.A)
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            v = int(decoder.decode(_Cat(_probs(counts), perfect=self.perfect)))
            out[i] = v
            counts[v] += 1
        return out


def zeroth_order_entropy(syms, A):
    counts = np.bincount(np.asarray(syms, dtype=np.int64), minlength=A).astype(np.float64)
    p = counts / counts.sum()
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def measure(q, bits, A, verify=True):
    """Rate and entropy for a (n_traces, latent_dim) array of quantized symbols.

    Returns bits per trace for the fixed-width baseline, the entropy lower bound
    and the measured adaptive-coder rate, averaged over traces. With verify=True
    every trace is decoded back and compared, so a rate is never reported for
    symbols a decoder could not recover.
    """
    n, lat = q.shape
    coder = AdaptiveCoder(A)
    coded = 0.0
    ent = 0.0
    for t in range(n):
        seq = q[t]
        nb, dec = coder.encode(seq)
        coded += nb
        ent += zeroth_order_entropy(seq, A) * lat
        if verify:
            back = coder.decode(dec, lat)
            if not np.array_equal(back, seq):
                raise RuntimeError(f"coder round trip failed: trace {t}/{n} at {bits} bits")
    return {
        "fixed_bits": float(bits * lat),
        "entropy_bound_bits": ent / n,
        "coded_bits": coded / n,
    }
