from pathlib import Path
from typing import Dict, cast

import ai_edge_torch
import ai_edge_torch.generative.layers.model_config as cfg
import ai_edge_torch.generative.utilities.loader as loading_utils
import download
import torch
from ai_edge_torch.generative.layers.attention import TransformerBlock
from ai_edge_torch.generative.quantize import quant_recipes
from const import OUTPUTS
from torch import nn
from train import dataset

MANGA_OCR_BASE = "kha-white/manga-ocr-base"


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=stride, padding=1
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


def create_cnn_frontend(in_channels: int = 3):
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU6(),
        DepthwiseSeparableConv(16, 32, stride=2),
        nn.BatchNorm2d(32),
        nn.ReLU6(),
        DepthwiseSeparableConv(32, 64, stride=2),
    )


class MobileViTEncoder(nn.Module):
    def __init__(self, config: cfg.ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.cnn_frontend = create_cnn_frontend()

        # Patch embedding
        patch_size = 4  # Smaller patch size after downsampling
        self.patch_embed = nn.Conv2d(
            64, config.embedding_dim, kernel_size=patch_size, stride=patch_size
        )

        # image_size = 224  # TODO: config?
        # effective_img_size = image_size // 8
        # num_patches = effective_img_size // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    cast(cfg.TransformerBlockConfig, config.block_configs),
                    config,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.norm = nn.LayerNorm(config.embedding_dim)

    def forward(self, pixel_values):
        pixel_values = self.cnn_frontend(pixel_values)

        pixel_values = self.patch_embed(pixel_values)
        pixel_values = pixel_values.flatten(2).transpose(1, 2)

        # Add class token
        cls_token = self.cls_token.expand(pixel_values.shape[0], -1, -1)
        pixel_values = torch.cat((cls_token, pixel_values), dim=1)

        # NOTE: The original encoder doesn't *seem* to have position
        # embeddings? Not sure whether this is important...
        pos_embed = self.pos_embed.expand(-1, pixel_values.shape[1], -1)
        pixel_values = pixel_values + pos_embed

        for block in self.blocks:
            pixel_values = block(pixel_values)

        pixel_values = self.norm(pixel_values)
        return pixel_values


class LoadedModelLoader(loading_utils.ModelLoader):
    """Google's version barfs if we try to load without CUDA
    (eg, on a mac). This version lets us pass in the result
    from torch.load, which is what it uses anyway."""

    def __init__(
        self,
        path: Path,
        state_dict: Dict[str, torch.Tensor],
        names: loading_utils.ModelLoader.TensorNames,
    ):
        super().__init__(str(path), names)
        self._state_dict = state_dict

    def get_state(self):
        return self._state_dict


def _reauthor_encoder(original_bin: Path, encoder_path: Path):
    config, state_dict = _prepare_config(original_bin, is_encoder=True)
    encoder = MobileViTEncoder(config)

    print("Loading weights onto the encoder")
    tensor_names = loading_utils.ModelLoader.TensorNames(
        ff_up_proj="encoder.encoder.layer.{}.intermediate.dense",
        ff_down_proj="encoder.encoder.layer.{}.output.dense",
        attn_query_proj="encoder.encoder.layer.{}.attention.attention.query",
        attn_key_proj="encoder.encoder.layer.{}.attention.attention.key",
        attn_value_proj="encoder.encoder.layer.{}.attention.attention.value",
        attn_output_proj="encoder.encoder.layer.{}.attention.output.dense",
        pre_attn_norm="encoder.encoder.layer.{}.layernorm_before",
        post_attn_norm="encoder.encoder.layer.{}.layernorm_after",
        embedding="encoder.embeddings.patch_embeddings.projection",
        final_norm="encoder.layernorm",
    )
    loader = LoadedModelLoader(original_bin, state_dict, names=tensor_names)
    loader.load(encoder, strict=False)

    image_tensor = torch.randn((1, 3, 224, 224), dtype=torch.float32)
    # tokens_tensor = torch.tensor([[2]], dtype=torch.int)

    quant_config = quant_recipes.full_int8_dynamic_recipe()
    edge_model = ai_edge_torch.convert(
        encoder.eval(), (image_tensor,), quant_config=quant_config
    )
    edge_model.export(str(encoder_path))
    print(encoder_path, "@", encoder_path.stat().st_size)

    # TODO: Attempt to verify
    # original_model = VisionEncoderDecoderModel.from_pretrained(MANGA_OCR_BASE)
    # verifier.verify_reauthored_model(
    #     original_model=transformers_verifier.TransformersModelWrapper(original_model),
    #     reauthored_model=verifier.ReauthoredModelWrapper(encoder),
    #     tokenizer=verifier.TokenizerWrapper(
    #         AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-char-v2")
    #     ),
    #     generate_prompts=["What is the meaning of life?"],
    #     max_new_tokens=30,
    # )

    return encoder, edge_model


class MobileBertDecoder(nn.Module):
    def __init__(self, config: cfg.ModelConfig) -> None:
        super().__init__()
        self.config = config

        hidden_size = config.embedding_dim
        self.embedding = nn.Embedding(config.vocab_size, hidden_size)
        self.decoder_layers = nn.ModuleList(
            [
                LightweightDecoderLayer(hidden_size, config.embedding_dim)
                for _ in range(1)  # two layers
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.output_projection = nn.Linear(hidden_size, config.vocab_size)

    def forward(self, encoder_outputs, decoder_input_ids):
        text_embeds = self.embedding(decoder_input_ids)
        for layer in self.decoder_layers:
            text_embeds = layer(text_embeds, encoder_outputs)

        text_embeds = self.norm(text_embeds)

        logits = self.output_projection(text_embeds)
        return logits


class LightweightDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            hidden_size, num_heads=1, batch_first=True
        )
        self.self_attention_norm = nn.LayerNorm(hidden_size)

        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads=1, batch_first=True
        )
        self.cross_attention_norm = nn.LayerNorm(hidden_size)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Linear(intermediate_size, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, encoder_hidden_states):
        residual = hidden_states
        hidden_states = self.self_attention_norm(hidden_states)

        hidden_states, _ = self.self_attention(
            hidden_states, hidden_states, hidden_states
        )
        hidden_states = residual + hidden_states

        # cross-attention
        residual = hidden_states
        hidden_states = self.cross_attention_norm(hidden_states)
        hidden_states, _ = self.cross_attention(
            hidden_states, encoder_hidden_states, hidden_states
        )
        hidden_states = residual + hidden_states

        # feed forward
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


def _reauthor_decoder(original_bin: Path, decoder_path: Path):
    config, state_dict = _prepare_config(original_bin, is_encoder=False)
    decoder = MobileBertDecoder(config)

    print("Loading weights onto the encoder")
    tensor_names = loading_utils.ModelLoader.TensorNames(
        ff_up_proj="decoder.bert.encoder.layer.{}.intermediate.dense",
        ff_down_proj="decoder.bert.encoder.layer.{}.output.dense",
        attn_query_proj="decoder.bert.encoder.layer.{}.attention.self.query",
        attn_key_proj="decoder.bert.encoder.layer.{}.attention.self.key",
        attn_value_proj="decoder.bert.encoder.layer.{}.attention.self.value",
        attn_output_proj="decoder.bert.encoder.layer.{}.attention.output.dense",
        pre_attn_norm="decoder.bert.encoder.layer.{}.attention.output.LayerNorm",
        post_attn_norm="decoder.bert.encoder.layer.{}.output.LayerNorm",
        embedding="decoder.bert.embeddings.word_embeddings",
        final_norm="decoder.bert.embeddings.LayerNorm",
    )
    loader = LoadedModelLoader(original_bin, state_dict, names=tensor_names)
    loader.load(decoder, strict=False)

    encoded_tensor = torch.randn((1, 1, 768), dtype=torch.float32)
    tokens_tensor = torch.tensor([[2]], dtype=torch.int)

    quant_config = quant_recipes.full_int8_dynamic_recipe()
    edge_model = ai_edge_torch.convert(
        decoder.eval(), (encoded_tensor, tokens_tensor), quant_config=quant_config
    )
    edge_model.export(str(decoder_path))
    print(decoder_path, "@", decoder_path.stat().st_size)

    return decoder, edge_model


def _prepare_config(original_bin: Path, is_encoder: bool):
    state_dict = torch.load(
        str(original_bin),
        map_location=torch.device("cpu") if not torch.cuda.is_available() else None,
    )
    print("state_dict=", state_dict.keys())

    attn_config = cfg.AttentionConfig(
        num_heads=12,
        head_dim=64,
        num_query_groups=12,
    )
    ff_config = cfg.FeedForwardConfig(
        type=cfg.FeedForwardType.SEQUENTIAL,
        activation=cfg.ActivationConfig(cfg.ActivationType.RELU),
        intermediate_size=3072,
    )
    block_config = cfg.TransformerBlockConfig(
        attn_config=attn_config, relative_attention=True, ff_config=ff_config
    )

    encoder_layers = 8  # NOTE: Originally 12
    decoder_layers = 2
    vocab = dataset.load_vocab()
    config = cfg.ModelConfig(
        vocab_size=len(vocab),
        num_layers=encoder_layers if is_encoder else decoder_layers,
        max_seq_len=300,
        embedding_dim=768,
        block_configs=block_config,
    )
    return config, state_dict


def reauthor(force: bool = False, evaluate: bool = False):
    original_bin = download.hf(MANGA_OCR_BASE, "pytorch_model.bin")
    assert original_bin, "Failed to fetch manga-ocr-base model"

    encoder_path = OUTPUTS / "manga-ocr.encoder.tflite"
    decoder_path = OUTPUTS / "manga-ocr.decoder.tflite"
    if force or not encoder_path.exists():
        _reauthor_encoder(original_bin, encoder_path)
    else:
        print("Already converted:", encoder_path)
    if force or not decoder_path.exists():
        _reauthor_decoder(original_bin, decoder_path)
    else:
        print("Already converted:", decoder_path)

    if evaluate:
        encoder = ai_edge_torch.load(str(encoder_path))
        decoder = ai_edge_torch.load(str(decoder_path))

        # vocab = dataset.load_vocab()

        dataset_path = dataset.download_manga109s()
        sample = dataset.onnx_calibration_reader(dataset_path, sample=False).get_next()
        image_data = torch.tensor(sample["image"])
        print("image_data=", image_data)

        encoded = cast(torch.Tensor, encoder(image_data))
        print("encoded=", encoded.shape)
        print("pass to", decoder)
        decoded = decoder(encoded, torch.tensor([[2]], dtype=torch.int))
        print(decoded)
