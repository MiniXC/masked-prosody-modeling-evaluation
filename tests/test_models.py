import sys

sys.path.append(".")  # add root of project to path

import torch
import onnxruntime as ort

from model.burn_classifiers import BreakProminenceClassifier
from model.ravdess_classifiers import EmotionClassifier
from model.timit_classifiers import PhonemeWordBoundaryClassifier
from configs.args import BURNModelArgs, TIMITModelArgs, RAVDESSModelArgs

model_burn = BreakProminenceClassifier(BURNModelArgs())
model_ravdess = EmotionClassifier(RAVDESSModelArgs())
model_timit = PhonemeWordBoundaryClassifier(TIMITModelArgs())

BURN_OUT_SHAPE = (1, 256, 2)
RAVDESS_OUT_SHAPE = (1, 8)
TIMIT_OUT_SHAPE = (1, 256, 2)


def test_forward_pass_burn():
    x = model_burn.dummy_input
    y = model_burn(x)
    assert y.shape == BURN_OUT_SHAPE


def test_forward_pass_ravdess():
    x = model_ravdess.dummy_input
    y = model_ravdess(x)
    assert y.shape == RAVDESS_OUT_SHAPE


def test_forward_pass_timit():
    x = model_timit.dummy_input
    y = model_timit(x)
    assert y.shape == TIMIT_OUT_SHAPE


def test_save_load_model_burn(tmp_path):
    model_burn.save_model(tmp_path / "test")
    model_burn.from_pretrained(tmp_path / "test")
    x = model_burn.dummy_input
    y = model_burn(x)
    assert y.shape == BURN_OUT_SHAPE


def test_save_load_model_ravdess(tmp_path):
    model_ravdess.save_model(tmp_path / "test")
    model_ravdess.from_pretrained(tmp_path / "test")
    x = model_ravdess.dummy_input
    y = model_ravdess(x)
    assert y.shape == RAVDESS_OUT_SHAPE


def test_save_load_model_timit(tmp_path):
    model_timit.save_model(tmp_path / "test")
    model_timit.from_pretrained(tmp_path / "test")
    x = model_timit.dummy_input
    y = model_timit(x)
    assert y.shape == TIMIT_OUT_SHAPE


def test_onnx_burn(tmp_path):
    with torch.no_grad():
        model_burn.export_onnx(tmp_path / "test" / "model_burn.onnx")
        ort_session = ort.InferenceSession(tmp_path / "test" / "model_burn.onnx")
        dummy_input = model_burn.dummy_input
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        assert ort_outs[0].shape == BURN_OUT_SHAPE
        regular_outs = model_burn(dummy_input)
        mean_abs_error = torch.abs(regular_outs - torch.tensor(ort_outs[0])).mean()
        assert mean_abs_error < 0.01


def test_onnx_ravdess(tmp_path):
    with torch.no_grad():
        model_ravdess.export_onnx(tmp_path / "test" / "model_ravdess.onnx")
        ort_session = ort.InferenceSession(tmp_path / "test" / "model_ravdess.onnx")
        dummy_input = model_ravdess.dummy_input
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        assert ort_outs[0].shape == RAVDESS_OUT_SHAPE
        regular_outs = model_ravdess(dummy_input)
        mean_abs_error = torch.abs(regular_outs - torch.tensor(ort_outs[0])).mean()
        assert mean_abs_error < 0.02


def test_onnx_timit(tmp_path):
    with torch.no_grad():
        model_timit.export_onnx(tmp_path / "test" / "model_timit.onnx")
        ort_session = ort.InferenceSession(tmp_path / "test" / "model_timit.onnx")
        dummy_input = model_timit.dummy_input
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        assert ort_outs[0].shape == TIMIT_OUT_SHAPE
        regular_outs = model_timit(dummy_input)
        mean_abs_error = torch.abs(regular_outs - torch.tensor(ort_outs[0])).mean()
        assert mean_abs_error < 0.01
