import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

SEQ_LEN = 128
config = AutoConfig.from_pretrained("./embeddinggemma_weights")
config._attn_implementation = "eager"
config.attention_bias = False
model = AutoModel.from_pretrained(
    "./embeddinggemma_weights",
    config=config,
    dtype=torch.float32,
    attn_implementation="eager",
).eval()

class Wrapper(nn.Module):
    def __init__(self, model, seq_len):
        super().__init__()
        self.model = model
        self.register_buffer("position_ids", torch.arange(seq_len, dtype=torch.int64).unsqueeze(0))

    def forward(self, inputs_embeds: torch.Tensor, attention_bias: torch.Tensor) -> torch.Tensor:
        masks = {"full_attention": attention_bias, "sliding_attention": attention_bias}
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            position_ids=self.position_ids,
            attention_mask=masks,
            use_cache=False,
            return_dict=False,
        )
        return outputs[0]

wrapped = Wrapper(model, SEQ_LEN).eval()
sample_embeds = torch.randn(1, SEQ_LEN, 768, dtype=torch.float32)
sample_bias = torch.zeros(1, 1, 1, SEQ_LEN, dtype=torch.float32)
with torch.no_grad():
    out = wrapped(sample_embeds, sample_bias)
    print("Shape:", tuple(out.shape), "finite:", bool(torch.isfinite(out).all()))
onnx_program = torch.onnx.export(wrapped, (sample_embeds, sample_bias), dynamo=True)
onnx_program.save("embeddinggemma_seq128_bias_hidden.onnx")
print("Saved embeddinggemma_seq128_bias_hidden.onnx")
