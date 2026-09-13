"""Exercise the real VAE interface without fetching pretrained weights."""

import pytest


def test_codec_supervision_uses_real_vae_normalization_and_scale(monkeypatch):
    torch = pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    from moebius_finetune.students import LightweightCodec
    from moebius_finetune.training.codec import train_codec

    torch.manual_seed(19)
    vae = diffusers.AutoencoderKL(
        in_channels=3, out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(8, 8, 8, 8), layers_per_block=1,
        latent_channels=4, norm_num_groups=4, sample_size=32,
        scaling_factor=0.5,
    )
    rgb = torch.linspace(0, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
    seen = {}
    original_encode = vae.encode
    original_mse = torch.nn.functional.mse_loss

    def encode(image, *args, **kwargs):
        seen["image"] = image.detach().clone()
        result = original_encode(image, *args, **kwargs)
        seen["mode"] = result.latent_dist.mode().detach().clone()
        return result

    def mse(pred, target, *args, **kwargs):
        seen["target"] = target.detach().clone()
        return original_mse(pred, target, *args, **kwargs)

    monkeypatch.setattr(vae, "encode", encode)
    monkeypatch.setattr(torch.nn.functional, "mse_loss", mse)
    trainer = train_codec(
        LightweightCodec(), vae, steps=1,
        cfg={"device": "cpu", "batches": [rgb]},
    )
    torch.testing.assert_close(seen["image"], 2.0 * rgb - 1.0)
    torch.testing.assert_close(seen["target"], 0.5 * seen["mode"])
    assert torch.isfinite(torch.tensor(trainer.final_loss))
    assert all(p.grad is None and not p.requires_grad for p in vae.parameters())
