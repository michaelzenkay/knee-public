import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from knee_mri.anatomy import FusedKneeAnatomyResNet34Unet2p5D
from knee_mri.anatomy_training import anatomy_loss, attachment_batch
from knee_mri.inference import load_model, predict
from knee_mri.model import Config, NativeKnee
from knee_mri.preprocessing import ExamPreprocessor
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage, generate_uid
from transformers import Dinov2Config, Dinov2Model


class KneeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(11)

    def test_dicom_to_physical_model_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "study" / "series"
            folder.mkdir(parents=True)
            for index, z in enumerate((6, 0, 3)):
                meta = FileMetaDataset()
                meta.TransferSyntaxUID = ExplicitVRLittleEndian
                meta.MediaStorageSOPClassUID = MRImageStorage
                meta.MediaStorageSOPInstanceUID = generate_uid()
                ds = FileDataset(
                    str(folder / f"{index}.dcm"),
                    {},
                    file_meta=meta,
                    preamble=b"\0" * 128,
                )
                ds.SOPClassUID, ds.SOPInstanceUID = (
                    MRImageStorage,
                    meta.MediaStorageSOPInstanceUID,
                )
                ds.Rows, ds.Columns, ds.SamplesPerPixel = 31, 47, 1
                ds.PhotometricInterpretation = "MONOCHROME2"
                ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = (
                    16,
                    16,
                    15,
                    0,
                )
                ds.PixelSpacing = [0.4, 0.6]
                ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
                ds.ImagePositionPatient = [0, 0, z]
                ds.RescaleSlope, ds.RescaleIntercept = 2, -5
                ds.PixelData = (
                    np.arange(31 * 47, dtype=np.uint16).reshape(31, 47) + index
                ).tobytes()
                ds.save_as(ds.filename, enforce_file_format=True)
            rows = [
                dict(
                    StudyInstanceUID="study",
                    SeriesInstanceUID="series",
                    Anatomical_Plane="Sagittal",
                    Fluid_Sensitive="1",
                    Fat_Suppression="0",
                )
            ]
            prep = ExamPreprocessor()
            processed, _ = prep.process(directory, "study", rows)
            arrays, _ = processed[0]
            np.testing.assert_array_equal(arrays["positions_mm"], [0, 3, 6])
            self.assertEqual(arrays["pixels"][0, 0, 0], -3)
            images, _ = prep.model_inputs(directory, "study", rows)
            self.assertEqual(tuple(images[0]["pixels"].shape), (3, 31, 47))
            self.assertTrue(torch.isfinite(images[0]["pixels"]).all())

    def test_classifier_gradients_and_checkpoint(self):
        backbone = Dinov2Model(
            Dinov2Config(
                hidden_size=384,
                num_hidden_layers=1,
                num_attention_heads=6,
                image_size=28,
                patch_size=14,
            )
        )
        model = NativeKnee(
            backbone,
            Config(width=24, heads=3, spatial_queries=2, depth_layers=1, dropout=0),
        )
        images = [
            dict(
                pixels=torch.rand(depth, 29, 43),
                positions_mm=torch.arange(depth).float() * 3,
                spacing_mm=torch.tensor([0.4, 0.6]),
                orientation=torch.tensor([1.0, 0, 0, 0, 1, 0]),
                protocol=protocol,
            )
            for depth, protocol in ((3, 2), (2, 5))
        ]
        model.train()
        model.set_phase(True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(images), torch.zeros(12)
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.patch_projection.weight.grad.abs().sum()), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            backbone.config.to_json_file(path / "dino.json")
            torch.save(
                dict(
                    identity={"architecture": model.architecture()},
                    model=model.state_dict(),
                ),
                path / "model.pt",
            )
            restored = load_model(path / "model.pt", path / "dino.json")
            np.testing.assert_allclose(
                predict(model, images), predict(restored, images), atol=1e-6, rtol=0
            )

    def test_anatomy_heads_and_point_supervision(self):
        model = FusedKneeAnatomyResNet34Unet2p5D().train()
        volume = np.random.default_rng(3).random((3, 32, 40), dtype=np.float32)
        batch = attachment_batch(volume, [[0.3, 0.4, 0], [0.7, 0.6, 2]], size=64)
        self.assertEqual(batch["target_presence"].argmax(0).tolist(), [0, 2])
        output = model(batch["image"])
        self.assertEqual(tuple(output["segmentation_logits"].shape), (3, 4, 64, 64))
        seg = dict(
            image=batch["image"],
            target=torch.zeros(3, 4, 64, 64),
            supervised_classes=torch.ones(3, 4),
        )
        total = anatomy_loss(model, seg, "segmentation")
        for task in ("acl", "mcl"):
            total = total + anatomy_loss(model, batch, task)
        total.backward()
        self.assertTrue(torch.isfinite(total))
        for task in ("acl", "mcl"):
            self.assertGreater(
                float(model.attachment_outputs[task].weight.grad.abs().sum()), 0
            )
