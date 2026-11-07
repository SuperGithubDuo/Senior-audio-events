# Self-hosting YAMNet for the test page (optional)

By default the page downloads the TensorFlow.js version of YAMNet from
TensorFlow Hub / Kaggle. To serve it from this site instead (no third-party
model download, works if Kaggle is blocked on your network):

1. Download "YAMNet, TF.js" from https://www.kaggle.com/models/google/yamnet
   (framework TensorFlow.js, variation `tfjs`).
2. Extract `model.json` and all `group1-shard*.bin` files into `docs/models/yamnet/`.
3. Commit and push. The page tries `./models/yamnet/model.json` first.

YAMNet is released by Google under the Apache 2.0 license.
