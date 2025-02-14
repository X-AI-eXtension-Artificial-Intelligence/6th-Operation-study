### Transformer Architecture

**(1) Encoder**

Encoder block
- [Add&Norm]
- [FFN]
- [Add&Norm]
- [Multi-Head Attn]
---
- [POS Encoding]
- [Input Embedding]



**(2) Decoder**
- Decoder Block
- [Add&Norm]
- [FFN]
- [Add&Norm]
- [Multi-Head Attn]
- [Add&Norm]
- [Masked Multi-Head Attn]
---
- [POS Encoding]
- [Output Embedding] -> shifted right
