"""A checkpoint without the Clay encoder weights must fail, not train or embed
with a random encoder.

    python -m unittest discover -s tests -v

Runs on CPU; the test that needs checkpoints/clay-v1.5.ckpt skips when it is absent.
"""

import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints/clay-v1.5.ckpt"


class LoadEncoderWeightsTest(unittest.TestCase):
    def test_wrong_checkpoint_raises(self):
        from claymodel.finetune.embedder.factory import Embedder
        from claymodel.finetune.segment.factory import Segmentor

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "wrong.ckpt"
            torch.save({"state_dict": {"model.decoder.x": torch.zeros(1)}}, ckpt)
            with self.assertRaisesRegex(ValueError, "265 of 265 encoder parameters"):
                Embedder(img_size=16, ckpt_path=ckpt)
            with self.assertRaisesRegex(ValueError, "265 of 265 encoder parameters"):
                Segmentor(num_classes=2, ckpt_path=ckpt)

    @unittest.skipUnless(CKPT.exists(), f"{CKPT} not downloaded")
    def test_clay_checkpoint_loads_and_freezes(self):
        from claymodel.finetune.embedder.factory import Embedder

        encoder = Embedder(img_size=16, ckpt_path=CKPT).clay_encoder
        sd = torch.load(CKPT, map_location="cpu", mmap=True)["state_dict"]
        weight = encoder.transformer.layers[0][0].to_qkv.weight
        self.assertTrue(
            torch.equal(
                weight, sd["model.encoder.transformer.layers.0.0.to_qkv.weight"]
            )
        )
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
