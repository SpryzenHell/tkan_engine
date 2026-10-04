# T-KAN Microstructure Engine — end-to-end notebook-style script

import pandas as pd

from tkan_engine.data import SyntheticLOBStream
from tkan_engine.features import LOBFeatureExtractor

frames = [
    LOBFeatureExtractor.transform(frame, 10)
    for frame in SyntheticLOBStream(5000, batch_size=1000).batches()
]
df = pd.concat(frames, ignore_index=True)
df["mid_return"] = df["mid"].pct_change().fillna(0.0)
df.head()
