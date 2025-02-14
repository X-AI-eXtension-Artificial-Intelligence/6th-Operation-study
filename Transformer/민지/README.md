## Transformer Architecture

### **(1) Encoder**

- Encoder block
- [Add&Norm]
- [FFN]
- [Add&Norm]
- [Multi-Head Attn]
---
- [POS Encoding]
- [Input Embedding]



### **(2) Decoder**
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

![image](https://github.com/user-attachments/assets/2dcd0e32-8d1e-4168-be2a-7ad4dd993230)
![image](https://github.com/user-attachments/assets/94495d3a-cb74-486e-bd21-6c5cf2094a4e)
