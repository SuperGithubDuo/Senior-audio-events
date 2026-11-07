"""TEST ONLY: build a stand-in for the YAMNet SavedModel.

It reproduces YAMNet's exact waveform padding, log-mel features and 0.96 s /
0.48 s patch framing, and the TF Hub call signature
    model(waveform) -> (scores [T, 521], embeddings [T, 1024], log_mel [F, 64])
    model.class_map_path()
but the network is a fixed random projection, NOT the trained YAMNet. Use it to
exercise the pipeline where the real model cannot be downloaded; never for
real detection. Feature code follows tensorflow/models research/audioset/yamnet
(Apache 2.0).

usage: python tools/make_standin_yamnet.py --out models/yamnet_standin --class-map yamnet_class_map.csv
"""

import argparse
import os
import shutil

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf

SR, WIN, HOP, MELS = 16000.0, 0.025, 0.010, 64
PATCH_S, PATCH_HOP_S = 0.96, 0.48


def pad_waveform(w):
    min_n = tf.cast((PATCH_S + WIN - HOP) * SR, tf.int32)
    n = tf.shape(w)[0]
    pad = tf.maximum(0, min_n - n)
    n = tf.maximum(n, min_n)
    after = n - min_n
    hop = tf.cast(PATCH_HOP_S * SR, tf.int32)
    hops = tf.cast(tf.math.ceil(tf.cast(after, tf.float32) / tf.cast(hop, tf.float32)), tf.int32)
    pad += hop * hops - after
    return tf.pad(w, [[0, pad]])


def log_mel_patches(w):
    wl, hl = int(round(SR * WIN)), int(round(SR * HOP))
    fft = 2 ** int(np.ceil(np.log2(wl)))
    mag = tf.abs(tf.signal.stft(w, wl, hl, fft))
    mel = tf.signal.linear_to_mel_weight_matrix(MELS, fft // 2 + 1, SR, 125.0, 7500.0)
    lm = tf.math.log(tf.matmul(mag, mel) + 0.001)
    return lm, tf.signal.frame(lm, 96, 48, axis=0)


class StandIn(tf.Module):
    def __init__(self, class_map):
        super().__init__()
        rng = np.random.default_rng(1234)
        self.A = tf.Variable(rng.normal(0, 1 / np.sqrt(256), (256, 1024)).astype(np.float32))
        self.c = tf.Variable(rng.normal(0, 0.1, 1024).astype(np.float32))
        self.B = tf.Variable(rng.normal(0, 1 / 32, (1024, 521)).astype(np.float32))
        self.d = tf.Variable(np.full(521, -3.0, np.float32))
        self.asset = tf.saved_model.Asset(class_map)

    @tf.function(input_signature=[tf.TensorSpec([None], tf.float32)])
    def __call__(self, waveform):
        lm, patches = log_mel_patches(pad_waveform(waveform))
        # per-patch summary: band mean/std/max + max frame-to-frame rise (onsets)
        rise = tf.reduce_max(patches[:, 1:, :] - patches[:, :-1, :], axis=1)
        f = tf.concat([tf.reduce_mean(patches, 1), tf.math.reduce_std(patches, 1),
                       tf.reduce_max(patches, 1), rise], axis=1)
        emb = tf.nn.relu(tf.matmul(f / 5.0, self.A) + self.c)
        scores = tf.sigmoid(tf.matmul(emb, self.B) + self.d)
        return scores, emb, lm

    @tf.function(input_signature=[])
    def class_map_path(self):
        return self.asset.asset_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--class-map", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    m = StandIn(a.class_map)
    tf.saved_model.save(m, a.out)
    print(f"stand-in (random weights, test only) saved to {a.out}")


if __name__ == "__main__":
    main()
