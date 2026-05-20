# YOLO26 Model Analysis

- model_path: `/home/paipaiqi01/workspace/rknnquantizationLearning/spacer_640.pt`
- torch: `2.12.0+cu130`
- ultralytics: `8.4.51`
- cuda_available: `True`
- total_modules: `454`
- target_layers: `242`

## Forward Output

```json
[
  {
    "type": "Tensor",
    "shape": [
      1,
      300,
      6
    ],
    "dtype": "torch.float32",
    "min": -46.80003356933594,
    "max": 753.7049560546875
  },
  {
    "one2many": {
      "boxes": {
        "type": "Tensor",
        "shape": [
          1,
          4,
          8400
        ],
        "dtype": "torch.float32",
        "min": 0.4577754735946655,
        "max": 10.768367767333984
      },
      "scores": {
        "type": "Tensor",
        "shape": [
          1,
          1,
          8400
        ],
        "dtype": "torch.float32",
        "min": -19.641719818115234,
        "max": -3.090226173400879
      },
      "feats": [
        {
          "type": "Tensor",
          "shape": [
            1,
            64,
            80,
            80
          ],
          "dtype": "torch.float32",
          "min": -0.27846455574035645,
          "max": 5.566335201263428
        },
        {
          "type": "Tensor",
          "shape": [
            1,
            128,
            40,
            40
          ],
          "dtype": "torch.float32",
          "min": -0.27846455574035645,
          "max": 5.753626823425293
        },
        {
          "type": "Tensor",
          "shape": [
            1,
            256,
            20,
            20
          ],
          "dtype": "torch.float32",
          "min": -0.27846455574035645,
          "max": 11.906808853149414
        }
      ]
    },
    "one2one": {
      "boxes": {
        "type": "Tensor",
        "shape": [
          1,
          4,
          8400
        ],
        "dtype": "torch.float32",
        "min": 0.2047930806875229,
        "max": 10.6143159866333
      },
      "scores": {
        "type": "Tensor",
        "shape": [
          1,
          1,
          8400
        ],
        "dtype": "torch.float32",
        "min": -20.24595832824707,
        "max": -5.579802513122559
      },
      "feats": [
        {
          "type": "Tensor",
          "shape": [
            1,
            64,
            80,
            80
          ],
          "dtype": "torch.float32",
          "min": -0.27846455574035645,
          "max": 5.566335201263428
        },
        {
          "type": "Tensor",
          "shape": [
            1,
            128,
            40,
            40
          ],
          "dtype": "torch.float32",
          "min": -0.27846455574035645,
          "max": 5.753626823425293
        },
        {
          "type": "Tensor",
          "shape": [
            1,
            256,
            20,
            20
          ],
          "dtype": "torch.float32",
          "min": -0.27846455574035645,
          "max": 11.906808853149414
        }
      ]
    }
  }
]
```

## Target Hook Layers

| idx | name | type | weight_shape | params |
|---:|---|---|---|---:|
| 0 | `model.0` | `Conv` | `None` | 0 |
| 1 | `model.0.conv` | `Conv2d` | `[16, 3, 3, 3]` | 432 |
| 2 | `model.1` | `Conv` | `None` | 0 |
| 3 | `model.1.conv` | `Conv2d` | `[32, 16, 3, 3]` | 4608 |
| 4 | `model.2.cv1` | `Conv` | `None` | 0 |
| 5 | `model.2.cv1.conv` | `Conv2d` | `[32, 32, 1, 1]` | 1024 |
| 6 | `model.2.cv2` | `Conv` | `None` | 0 |
| 7 | `model.2.cv2.conv` | `Conv2d` | `[64, 48, 1, 1]` | 3072 |
| 8 | `model.2.m.0.cv1` | `Conv` | `None` | 0 |
| 9 | `model.2.m.0.cv1.conv` | `Conv2d` | `[8, 16, 3, 3]` | 1152 |
| 10 | `model.2.m.0.cv2` | `Conv` | `None` | 0 |
| 11 | `model.2.m.0.cv2.conv` | `Conv2d` | `[16, 8, 3, 3]` | 1152 |
| 12 | `model.3` | `Conv` | `None` | 0 |
| 13 | `model.3.conv` | `Conv2d` | `[64, 64, 3, 3]` | 36864 |
| 14 | `model.4.cv1` | `Conv` | `None` | 0 |
| 15 | `model.4.cv1.conv` | `Conv2d` | `[64, 64, 1, 1]` | 4096 |
| 16 | `model.4.cv2` | `Conv` | `None` | 0 |
| 17 | `model.4.cv2.conv` | `Conv2d` | `[128, 96, 1, 1]` | 12288 |
| 18 | `model.4.m.0.cv1` | `Conv` | `None` | 0 |
| 19 | `model.4.m.0.cv1.conv` | `Conv2d` | `[16, 32, 3, 3]` | 4608 |
| 20 | `model.4.m.0.cv2` | `Conv` | `None` | 0 |
| 21 | `model.4.m.0.cv2.conv` | `Conv2d` | `[32, 16, 3, 3]` | 4608 |
| 22 | `model.5` | `Conv` | `None` | 0 |
| 23 | `model.5.conv` | `Conv2d` | `[128, 128, 3, 3]` | 147456 |
| 24 | `model.6.cv1` | `Conv` | `None` | 0 |
| 25 | `model.6.cv1.conv` | `Conv2d` | `[128, 128, 1, 1]` | 16384 |
| 26 | `model.6.cv2` | `Conv` | `None` | 0 |
| 27 | `model.6.cv2.conv` | `Conv2d` | `[128, 192, 1, 1]` | 24576 |
| 28 | `model.6.m.0.cv1` | `Conv` | `None` | 0 |
| 29 | `model.6.m.0.cv1.conv` | `Conv2d` | `[32, 64, 1, 1]` | 2048 |
| 30 | `model.6.m.0.cv2` | `Conv` | `None` | 0 |
| 31 | `model.6.m.0.cv2.conv` | `Conv2d` | `[32, 64, 1, 1]` | 2048 |
| 32 | `model.6.m.0.cv3` | `Conv` | `None` | 0 |
| 33 | `model.6.m.0.cv3.conv` | `Conv2d` | `[64, 64, 1, 1]` | 4096 |
| 34 | `model.6.m.0.m.0.cv1` | `Conv` | `None` | 0 |
| 35 | `model.6.m.0.m.0.cv1.conv` | `Conv2d` | `[32, 32, 3, 3]` | 9216 |
| 36 | `model.6.m.0.m.0.cv2` | `Conv` | `None` | 0 |
| 37 | `model.6.m.0.m.0.cv2.conv` | `Conv2d` | `[32, 32, 3, 3]` | 9216 |
| 38 | `model.6.m.0.m.1.cv1` | `Conv` | `None` | 0 |
| 39 | `model.6.m.0.m.1.cv1.conv` | `Conv2d` | `[32, 32, 3, 3]` | 9216 |
| 40 | `model.6.m.0.m.1.cv2` | `Conv` | `None` | 0 |
| 41 | `model.6.m.0.m.1.cv2.conv` | `Conv2d` | `[32, 32, 3, 3]` | 9216 |
| 42 | `model.7` | `Conv` | `None` | 0 |
| 43 | `model.7.conv` | `Conv2d` | `[256, 128, 3, 3]` | 294912 |
| 44 | `model.8.cv1` | `Conv` | `None` | 0 |
| 45 | `model.8.cv1.conv` | `Conv2d` | `[256, 256, 1, 1]` | 65536 |
| 46 | `model.8.cv2` | `Conv` | `None` | 0 |
| 47 | `model.8.cv2.conv` | `Conv2d` | `[256, 384, 1, 1]` | 98304 |
| 48 | `model.8.m.0.cv1` | `Conv` | `None` | 0 |
| 49 | `model.8.m.0.cv1.conv` | `Conv2d` | `[64, 128, 1, 1]` | 8192 |
| 50 | `model.8.m.0.cv2` | `Conv` | `None` | 0 |
| 51 | `model.8.m.0.cv2.conv` | `Conv2d` | `[64, 128, 1, 1]` | 8192 |
| 52 | `model.8.m.0.cv3` | `Conv` | `None` | 0 |
| 53 | `model.8.m.0.cv3.conv` | `Conv2d` | `[128, 128, 1, 1]` | 16384 |
| 54 | `model.8.m.0.m.0.cv1` | `Conv` | `None` | 0 |
| 55 | `model.8.m.0.m.0.cv1.conv` | `Conv2d` | `[64, 64, 3, 3]` | 36864 |
| 56 | `model.8.m.0.m.0.cv2` | `Conv` | `None` | 0 |
| 57 | `model.8.m.0.m.0.cv2.conv` | `Conv2d` | `[64, 64, 3, 3]` | 36864 |
| 58 | `model.8.m.0.m.1.cv1` | `Conv` | `None` | 0 |
| 59 | `model.8.m.0.m.1.cv1.conv` | `Conv2d` | `[64, 64, 3, 3]` | 36864 |
| 60 | `model.8.m.0.m.1.cv2` | `Conv` | `None` | 0 |
| 61 | `model.8.m.0.m.1.cv2.conv` | `Conv2d` | `[64, 64, 3, 3]` | 36864 |
| 62 | `model.9` | `SPPF` | `None` | 0 |
| 63 | `model.9.cv1` | `Conv` | `None` | 0 |
| 64 | `model.9.cv1.conv` | `Conv2d` | `[128, 256, 1, 1]` | 32768 |
| 65 | `model.9.cv2` | `Conv` | `None` | 0 |
| 66 | `model.9.cv2.conv` | `Conv2d` | `[256, 512, 1, 1]` | 131072 |
| 67 | `model.10.cv1` | `Conv` | `None` | 0 |
| 68 | `model.10.cv1.conv` | `Conv2d` | `[256, 256, 1, 1]` | 65536 |
| 69 | `model.10.cv2` | `Conv` | `None` | 0 |
| 70 | `model.10.cv2.conv` | `Conv2d` | `[256, 256, 1, 1]` | 65536 |
| 71 | `model.10.m.0.attn.qkv` | `Conv` | `None` | 0 |
| 72 | `model.10.m.0.attn.qkv.conv` | `Conv2d` | `[256, 128, 1, 1]` | 32768 |
| 73 | `model.10.m.0.attn.proj` | `Conv` | `None` | 0 |
| 74 | `model.10.m.0.attn.proj.conv` | `Conv2d` | `[128, 128, 1, 1]` | 16384 |
| 75 | `model.10.m.0.attn.pe` | `Conv` | `None` | 0 |
| 76 | `model.10.m.0.attn.pe.conv` | `Conv2d` | `[128, 1, 3, 3]` | 1152 |
| 77 | `model.10.m.0.ffn.0` | `Conv` | `None` | 0 |
| 78 | `model.10.m.0.ffn.0.conv` | `Conv2d` | `[256, 128, 1, 1]` | 32768 |
| 79 | `model.10.m.0.ffn.1` | `Conv` | `None` | 0 |
| 80 | `model.10.m.0.ffn.1.conv` | `Conv2d` | `[128, 256, 1, 1]` | 32768 |
| 81 | `model.13.cv1` | `Conv` | `None` | 0 |
| 82 | `model.13.cv1.conv` | `Conv2d` | `[128, 384, 1, 1]` | 49152 |
| 83 | `model.13.cv2` | `Conv` | `None` | 0 |
| 84 | `model.13.cv2.conv` | `Conv2d` | `[128, 192, 1, 1]` | 24576 |
| 85 | `model.13.m.0.cv1` | `Conv` | `None` | 0 |
| 86 | `model.13.m.0.cv1.conv` | `Conv2d` | `[32, 64, 1, 1]` | 2048 |
| 87 | `model.13.m.0.cv2` | `Conv` | `None` | 0 |
| 88 | `model.13.m.0.cv2.conv` | `Conv2d` | `[32, 64, 1, 1]` | 2048 |
| 89 | `model.13.m.0.cv3` | `Conv` | `None` | 0 |
| 90 | `model.13.m.0.cv3.conv` | `Conv2d` | `[64, 64, 1, 1]` | 4096 |
| 91 | `model.13.m.0.m.0.cv1` | `Conv` | `None` | 0 |
| 92 | `model.13.m.0.m.0.cv1.conv` | `Conv2d` | `[32, 32, 3, 3]` | 9216 |
| 93 | `model.13.m.0.m.0.cv2` | `Conv` | `None` | 0 |
| 94 | `model.13.m.0.m.0.cv2.conv` | `Conv2d` | `[32, 32, 3, 3]` | 9216 |
| 95 | `model.13.m.0.m.1.cv1` | `Conv` | `None` | 0 |
| 96 | `model.13.m.0.m.1.cv1.conv` | `Conv2d` | `[32, 32, 3, 3]` | 9216 |
| 97 | `model.13.m.0.m.1.cv2` | `Conv` | `None` | 0 |
| 98 | `model.13.m.0.m.1.cv2.conv` | `Conv2d` | `[32, 32, 3, 3]` | 9216 |
| 99 | `model.16.cv1` | `Conv` | `None` | 0 |
| 100 | `model.16.cv1.conv` | `Conv2d` | `[64, 256, 1, 1]` | 16384 |
| 101 | `model.16.cv2` | `Conv` | `None` | 0 |
| 102 | `model.16.cv2.conv` | `Conv2d` | `[64, 96, 1, 1]` | 6144 |
| 103 | `model.16.m.0.cv1` | `Conv` | `None` | 0 |
| 104 | `model.16.m.0.cv1.conv` | `Conv2d` | `[16, 32, 1, 1]` | 512 |
| 105 | `model.16.m.0.cv2` | `Conv` | `None` | 0 |
| 106 | `model.16.m.0.cv2.conv` | `Conv2d` | `[16, 32, 1, 1]` | 512 |
| 107 | `model.16.m.0.cv3` | `Conv` | `None` | 0 |
| 108 | `model.16.m.0.cv3.conv` | `Conv2d` | `[32, 32, 1, 1]` | 1024 |
| 109 | `model.16.m.0.m.0.cv1` | `Conv` | `None` | 0 |
| 110 | `model.16.m.0.m.0.cv1.conv` | `Conv2d` | `[16, 16, 3, 3]` | 2304 |
| 111 | `model.16.m.0.m.0.cv2` | `Conv` | `None` | 0 |
| 112 | `model.16.m.0.m.0.cv2.conv` | `Conv2d` | `[16, 16, 3, 3]` | 2304 |
| 113 | `model.16.m.0.m.1.cv1` | `Conv` | `None` | 0 |
| 114 | `model.16.m.0.m.1.cv1.conv` | `Conv2d` | `[16, 16, 3, 3]` | 2304 |
| 115 | `model.16.m.0.m.1.cv2` | `Conv` | `None` | 0 |
| 116 | `model.16.m.0.m.1.cv2.conv` | `Conv2d` | `[16, 16, 3, 3]` | 2304 |
| 117 | `model.17` | `Conv` | `None` | 0 |
| 118 | `model.17.conv` | `Conv2d` | `[64, 64, 3, 3]` | 36864 |
| 119 | `model.19.cv1` | `Conv` | `None` | 0 |
| 120 | `model.19.cv1.conv` | `Conv2d` | `[128, 192, 1, 1]` | 24576 |
| 121 | `model.19.cv2` | `Conv` | `None` | 0 |
| 122 | `model.19.cv2.conv` | `Conv2d` | `[128, 192, 1, 1]` | 24576 |
| 123 | `model.19.m.0.cv1` | `Conv` | `None` | 0 |
| 124 | `model.19.m.0.cv1.conv` | `Conv2d` | `[32, 64, 1, 1]` | 2048 |
| 125 | `model.19.m.0.cv2` | `Conv` | `None` | 0 |
| 126 | `model.19.m.0.cv2.conv` | `Conv2d` | `[32, 64, 1, 1]` | 2048 |
| 127 | `model.19.m.0.cv3` | `Conv` | `None` | 0 |
| 128 | `model.19.m.0.cv3.conv` | `Conv2d` | `[64, 64, 1, 1]` | 4096 |
| 129 | `model.19.m.0.m.0.cv1` | `Conv` | `None` | 0 |
| 130 | `model.19.m.0.m.0.cv1.conv` | `Conv2d` | `[32, 32, 3, 3]` | 9216 |
| 131 | `model.19.m.0.m.0.cv2` | `Conv` | `None` | 0 |
| 132 | `model.19.m.0.m.0.cv2.conv` | `Conv2d` | `[32, 32, 3, 3]` | 9216 |
| 133 | `model.19.m.0.m.1.cv1` | `Conv` | `None` | 0 |
| 134 | `model.19.m.0.m.1.cv1.conv` | `Conv2d` | `[32, 32, 3, 3]` | 9216 |
| 135 | `model.19.m.0.m.1.cv2` | `Conv` | `None` | 0 |
| 136 | `model.19.m.0.m.1.cv2.conv` | `Conv2d` | `[32, 32, 3, 3]` | 9216 |
| 137 | `model.20` | `Conv` | `None` | 0 |
| 138 | `model.20.conv` | `Conv2d` | `[128, 128, 3, 3]` | 147456 |
| 139 | `model.22.cv1` | `Conv` | `None` | 0 |
| 140 | `model.22.cv1.conv` | `Conv2d` | `[256, 384, 1, 1]` | 98304 |
| 141 | `model.22.cv2` | `Conv` | `None` | 0 |
| 142 | `model.22.cv2.conv` | `Conv2d` | `[256, 384, 1, 1]` | 98304 |
| 143 | `model.22.m.0.0.cv1` | `Conv` | `None` | 0 |
| 144 | `model.22.m.0.0.cv1.conv` | `Conv2d` | `[64, 128, 3, 3]` | 73728 |
| 145 | `model.22.m.0.0.cv2` | `Conv` | `None` | 0 |
| 146 | `model.22.m.0.0.cv2.conv` | `Conv2d` | `[128, 64, 3, 3]` | 73728 |
| 147 | `model.22.m.0.1.attn.qkv` | `Conv` | `None` | 0 |
| 148 | `model.22.m.0.1.attn.qkv.conv` | `Conv2d` | `[256, 128, 1, 1]` | 32768 |
| 149 | `model.22.m.0.1.attn.proj` | `Conv` | `None` | 0 |
| 150 | `model.22.m.0.1.attn.proj.conv` | `Conv2d` | `[128, 128, 1, 1]` | 16384 |
| 151 | `model.22.m.0.1.attn.pe` | `Conv` | `None` | 0 |
| 152 | `model.22.m.0.1.attn.pe.conv` | `Conv2d` | `[128, 1, 3, 3]` | 1152 |
| 153 | `model.22.m.0.1.ffn.0` | `Conv` | `None` | 0 |
| 154 | `model.22.m.0.1.ffn.0.conv` | `Conv2d` | `[256, 128, 1, 1]` | 32768 |
| 155 | `model.22.m.0.1.ffn.1` | `Conv` | `None` | 0 |
| 156 | `model.22.m.0.1.ffn.1.conv` | `Conv2d` | `[128, 256, 1, 1]` | 32768 |
| 157 | `model.23` | `Detect` | `None` | 0 |
| 158 | `model.23.cv2.0.0` | `Conv` | `None` | 0 |
| 159 | `model.23.cv2.0.0.conv` | `Conv2d` | `[16, 64, 3, 3]` | 9216 |
| 160 | `model.23.cv2.0.1` | `Conv` | `None` | 0 |
| 161 | `model.23.cv2.0.1.conv` | `Conv2d` | `[16, 16, 3, 3]` | 2304 |
| 162 | `model.23.cv2.0.2` | `Conv2d` | `[4, 16, 1, 1]` | 68 |
| 163 | `model.23.cv2.1.0` | `Conv` | `None` | 0 |
| 164 | `model.23.cv2.1.0.conv` | `Conv2d` | `[16, 128, 3, 3]` | 18432 |
| 165 | `model.23.cv2.1.1` | `Conv` | `None` | 0 |
| 166 | `model.23.cv2.1.1.conv` | `Conv2d` | `[16, 16, 3, 3]` | 2304 |
| 167 | `model.23.cv2.1.2` | `Conv2d` | `[4, 16, 1, 1]` | 68 |
| 168 | `model.23.cv2.2.0` | `Conv` | `None` | 0 |
| 169 | `model.23.cv2.2.0.conv` | `Conv2d` | `[16, 256, 3, 3]` | 36864 |
| 170 | `model.23.cv2.2.1` | `Conv` | `None` | 0 |
| 171 | `model.23.cv2.2.1.conv` | `Conv2d` | `[16, 16, 3, 3]` | 2304 |
| 172 | `model.23.cv2.2.2` | `Conv2d` | `[4, 16, 1, 1]` | 68 |
| 173 | `model.23.cv3.0.0.0` | `DWConv` | `None` | 0 |
| 174 | `model.23.cv3.0.0.0.conv` | `Conv2d` | `[64, 1, 3, 3]` | 576 |
| 175 | `model.23.cv3.0.0.1` | `Conv` | `None` | 0 |
| 176 | `model.23.cv3.0.0.1.conv` | `Conv2d` | `[64, 64, 1, 1]` | 4096 |
| 177 | `model.23.cv3.0.1.0` | `DWConv` | `None` | 0 |
| 178 | `model.23.cv3.0.1.0.conv` | `Conv2d` | `[64, 1, 3, 3]` | 576 |
| 179 | `model.23.cv3.0.1.1` | `Conv` | `None` | 0 |
| 180 | `model.23.cv3.0.1.1.conv` | `Conv2d` | `[64, 64, 1, 1]` | 4096 |
| 181 | `model.23.cv3.0.2` | `Conv2d` | `[1, 64, 1, 1]` | 65 |
| 182 | `model.23.cv3.1.0.0` | `DWConv` | `None` | 0 |
| 183 | `model.23.cv3.1.0.0.conv` | `Conv2d` | `[128, 1, 3, 3]` | 1152 |
| 184 | `model.23.cv3.1.0.1` | `Conv` | `None` | 0 |
| 185 | `model.23.cv3.1.0.1.conv` | `Conv2d` | `[64, 128, 1, 1]` | 8192 |
| 186 | `model.23.cv3.1.1.0` | `DWConv` | `None` | 0 |
| 187 | `model.23.cv3.1.1.0.conv` | `Conv2d` | `[64, 1, 3, 3]` | 576 |
| 188 | `model.23.cv3.1.1.1` | `Conv` | `None` | 0 |
| 189 | `model.23.cv3.1.1.1.conv` | `Conv2d` | `[64, 64, 1, 1]` | 4096 |
| 190 | `model.23.cv3.1.2` | `Conv2d` | `[1, 64, 1, 1]` | 65 |
| 191 | `model.23.cv3.2.0.0` | `DWConv` | `None` | 0 |
| 192 | `model.23.cv3.2.0.0.conv` | `Conv2d` | `[256, 1, 3, 3]` | 2304 |
| 193 | `model.23.cv3.2.0.1` | `Conv` | `None` | 0 |
| 194 | `model.23.cv3.2.0.1.conv` | `Conv2d` | `[64, 256, 1, 1]` | 16384 |
| 195 | `model.23.cv3.2.1.0` | `DWConv` | `None` | 0 |
| 196 | `model.23.cv3.2.1.0.conv` | `Conv2d` | `[64, 1, 3, 3]` | 576 |
| 197 | `model.23.cv3.2.1.1` | `Conv` | `None` | 0 |
| 198 | `model.23.cv3.2.1.1.conv` | `Conv2d` | `[64, 64, 1, 1]` | 4096 |
| 199 | `model.23.cv3.2.2` | `Conv2d` | `[1, 64, 1, 1]` | 65 |
| 200 | `model.23.one2one_cv2.0.0` | `Conv` | `None` | 0 |
| 201 | `model.23.one2one_cv2.0.0.conv` | `Conv2d` | `[16, 64, 3, 3]` | 9216 |
| 202 | `model.23.one2one_cv2.0.1` | `Conv` | `None` | 0 |
| 203 | `model.23.one2one_cv2.0.1.conv` | `Conv2d` | `[16, 16, 3, 3]` | 2304 |
| 204 | `model.23.one2one_cv2.0.2` | `Conv2d` | `[4, 16, 1, 1]` | 68 |
| 205 | `model.23.one2one_cv2.1.0` | `Conv` | `None` | 0 |
| 206 | `model.23.one2one_cv2.1.0.conv` | `Conv2d` | `[16, 128, 3, 3]` | 18432 |
| 207 | `model.23.one2one_cv2.1.1` | `Conv` | `None` | 0 |
| 208 | `model.23.one2one_cv2.1.1.conv` | `Conv2d` | `[16, 16, 3, 3]` | 2304 |
| 209 | `model.23.one2one_cv2.1.2` | `Conv2d` | `[4, 16, 1, 1]` | 68 |
| 210 | `model.23.one2one_cv2.2.0` | `Conv` | `None` | 0 |
| 211 | `model.23.one2one_cv2.2.0.conv` | `Conv2d` | `[16, 256, 3, 3]` | 36864 |
| 212 | `model.23.one2one_cv2.2.1` | `Conv` | `None` | 0 |
| 213 | `model.23.one2one_cv2.2.1.conv` | `Conv2d` | `[16, 16, 3, 3]` | 2304 |
| 214 | `model.23.one2one_cv2.2.2` | `Conv2d` | `[4, 16, 1, 1]` | 68 |
| 215 | `model.23.one2one_cv3.0.0.0` | `DWConv` | `None` | 0 |
| 216 | `model.23.one2one_cv3.0.0.0.conv` | `Conv2d` | `[64, 1, 3, 3]` | 576 |
| 217 | `model.23.one2one_cv3.0.0.1` | `Conv` | `None` | 0 |
| 218 | `model.23.one2one_cv3.0.0.1.conv` | `Conv2d` | `[64, 64, 1, 1]` | 4096 |
| 219 | `model.23.one2one_cv3.0.1.0` | `DWConv` | `None` | 0 |
| 220 | `model.23.one2one_cv3.0.1.0.conv` | `Conv2d` | `[64, 1, 3, 3]` | 576 |
| 221 | `model.23.one2one_cv3.0.1.1` | `Conv` | `None` | 0 |
| 222 | `model.23.one2one_cv3.0.1.1.conv` | `Conv2d` | `[64, 64, 1, 1]` | 4096 |
| 223 | `model.23.one2one_cv3.0.2` | `Conv2d` | `[1, 64, 1, 1]` | 65 |
| 224 | `model.23.one2one_cv3.1.0.0` | `DWConv` | `None` | 0 |
| 225 | `model.23.one2one_cv3.1.0.0.conv` | `Conv2d` | `[128, 1, 3, 3]` | 1152 |
| 226 | `model.23.one2one_cv3.1.0.1` | `Conv` | `None` | 0 |
| 227 | `model.23.one2one_cv3.1.0.1.conv` | `Conv2d` | `[64, 128, 1, 1]` | 8192 |
| 228 | `model.23.one2one_cv3.1.1.0` | `DWConv` | `None` | 0 |
| 229 | `model.23.one2one_cv3.1.1.0.conv` | `Conv2d` | `[64, 1, 3, 3]` | 576 |
| 230 | `model.23.one2one_cv3.1.1.1` | `Conv` | `None` | 0 |
| 231 | `model.23.one2one_cv3.1.1.1.conv` | `Conv2d` | `[64, 64, 1, 1]` | 4096 |
| 232 | `model.23.one2one_cv3.1.2` | `Conv2d` | `[1, 64, 1, 1]` | 65 |
| 233 | `model.23.one2one_cv3.2.0.0` | `DWConv` | `None` | 0 |
| 234 | `model.23.one2one_cv3.2.0.0.conv` | `Conv2d` | `[256, 1, 3, 3]` | 2304 |
| 235 | `model.23.one2one_cv3.2.0.1` | `Conv` | `None` | 0 |
| 236 | `model.23.one2one_cv3.2.0.1.conv` | `Conv2d` | `[64, 256, 1, 1]` | 16384 |
| 237 | `model.23.one2one_cv3.2.1.0` | `DWConv` | `None` | 0 |
| 238 | `model.23.one2one_cv3.2.1.0.conv` | `Conv2d` | `[64, 1, 3, 3]` | 576 |
| 239 | `model.23.one2one_cv3.2.1.1` | `Conv` | `None` | 0 |
| 240 | `model.23.one2one_cv3.2.1.1.conv` | `Conv2d` | `[64, 64, 1, 1]` | 4096 |
| 241 | `model.23.one2one_cv3.2.2` | `Conv2d` | `[1, 64, 1, 1]` | 65 |

## All Modules

| name | type | params |
|---|---|---:|
| `<root>` | `DetectionModel` | 0 |
| `model` | `Sequential` | 0 |
| `model.0` | `Conv` | 0 |
| `model.0.conv` | `Conv2d` | 432 |
| `model.0.bn` | `BatchNorm2d` | 32 |
| `model.0.act` | `SiLU` | 0 |
| `model.1` | `Conv` | 0 |
| `model.1.conv` | `Conv2d` | 4608 |
| `model.1.bn` | `BatchNorm2d` | 64 |
| `model.2` | `C3k2` | 0 |
| `model.2.cv1` | `Conv` | 0 |
| `model.2.cv1.conv` | `Conv2d` | 1024 |
| `model.2.cv1.bn` | `BatchNorm2d` | 64 |
| `model.2.cv2` | `Conv` | 0 |
| `model.2.cv2.conv` | `Conv2d` | 3072 |
| `model.2.cv2.bn` | `BatchNorm2d` | 128 |
| `model.2.m` | `ModuleList` | 0 |
| `model.2.m.0` | `Bottleneck` | 0 |
| `model.2.m.0.cv1` | `Conv` | 0 |
| `model.2.m.0.cv1.conv` | `Conv2d` | 1152 |
| `model.2.m.0.cv1.bn` | `BatchNorm2d` | 16 |
| `model.2.m.0.cv2` | `Conv` | 0 |
| `model.2.m.0.cv2.conv` | `Conv2d` | 1152 |
| `model.2.m.0.cv2.bn` | `BatchNorm2d` | 32 |
| `model.3` | `Conv` | 0 |
| `model.3.conv` | `Conv2d` | 36864 |
| `model.3.bn` | `BatchNorm2d` | 128 |
| `model.4` | `C3k2` | 0 |
| `model.4.cv1` | `Conv` | 0 |
| `model.4.cv1.conv` | `Conv2d` | 4096 |
| `model.4.cv1.bn` | `BatchNorm2d` | 128 |
| `model.4.cv2` | `Conv` | 0 |
| `model.4.cv2.conv` | `Conv2d` | 12288 |
| `model.4.cv2.bn` | `BatchNorm2d` | 256 |
| `model.4.m` | `ModuleList` | 0 |
| `model.4.m.0` | `Bottleneck` | 0 |
| `model.4.m.0.cv1` | `Conv` | 0 |
| `model.4.m.0.cv1.conv` | `Conv2d` | 4608 |
| `model.4.m.0.cv1.bn` | `BatchNorm2d` | 32 |
| `model.4.m.0.cv2` | `Conv` | 0 |
| `model.4.m.0.cv2.conv` | `Conv2d` | 4608 |
| `model.4.m.0.cv2.bn` | `BatchNorm2d` | 64 |
| `model.5` | `Conv` | 0 |
| `model.5.conv` | `Conv2d` | 147456 |
| `model.5.bn` | `BatchNorm2d` | 256 |
| `model.6` | `C3k2` | 0 |
| `model.6.cv1` | `Conv` | 0 |
| `model.6.cv1.conv` | `Conv2d` | 16384 |
| `model.6.cv1.bn` | `BatchNorm2d` | 256 |
| `model.6.cv2` | `Conv` | 0 |
| `model.6.cv2.conv` | `Conv2d` | 24576 |
| `model.6.cv2.bn` | `BatchNorm2d` | 256 |
| `model.6.m` | `ModuleList` | 0 |
| `model.6.m.0` | `C3k` | 0 |
| `model.6.m.0.cv1` | `Conv` | 0 |
| `model.6.m.0.cv1.conv` | `Conv2d` | 2048 |
| `model.6.m.0.cv1.bn` | `BatchNorm2d` | 64 |
| `model.6.m.0.cv2` | `Conv` | 0 |
| `model.6.m.0.cv2.conv` | `Conv2d` | 2048 |
| `model.6.m.0.cv2.bn` | `BatchNorm2d` | 64 |
| `model.6.m.0.cv3` | `Conv` | 0 |
| `model.6.m.0.cv3.conv` | `Conv2d` | 4096 |
| `model.6.m.0.cv3.bn` | `BatchNorm2d` | 128 |
| `model.6.m.0.m` | `Sequential` | 0 |
| `model.6.m.0.m.0` | `Bottleneck` | 0 |
| `model.6.m.0.m.0.cv1` | `Conv` | 0 |
| `model.6.m.0.m.0.cv1.conv` | `Conv2d` | 9216 |
| `model.6.m.0.m.0.cv1.bn` | `BatchNorm2d` | 64 |
| `model.6.m.0.m.0.cv2` | `Conv` | 0 |
| `model.6.m.0.m.0.cv2.conv` | `Conv2d` | 9216 |
| `model.6.m.0.m.0.cv2.bn` | `BatchNorm2d` | 64 |
| `model.6.m.0.m.1` | `Bottleneck` | 0 |
| `model.6.m.0.m.1.cv1` | `Conv` | 0 |
| `model.6.m.0.m.1.cv1.conv` | `Conv2d` | 9216 |
| `model.6.m.0.m.1.cv1.bn` | `BatchNorm2d` | 64 |
| `model.6.m.0.m.1.cv2` | `Conv` | 0 |
| `model.6.m.0.m.1.cv2.conv` | `Conv2d` | 9216 |
| `model.6.m.0.m.1.cv2.bn` | `BatchNorm2d` | 64 |
| `model.7` | `Conv` | 0 |
| `model.7.conv` | `Conv2d` | 294912 |
| `model.7.bn` | `BatchNorm2d` | 512 |
| `model.8` | `C3k2` | 0 |
| `model.8.cv1` | `Conv` | 0 |
| `model.8.cv1.conv` | `Conv2d` | 65536 |
| `model.8.cv1.bn` | `BatchNorm2d` | 512 |
| `model.8.cv2` | `Conv` | 0 |
| `model.8.cv2.conv` | `Conv2d` | 98304 |
| `model.8.cv2.bn` | `BatchNorm2d` | 512 |
| `model.8.m` | `ModuleList` | 0 |
| `model.8.m.0` | `C3k` | 0 |
| `model.8.m.0.cv1` | `Conv` | 0 |
| `model.8.m.0.cv1.conv` | `Conv2d` | 8192 |
| `model.8.m.0.cv1.bn` | `BatchNorm2d` | 128 |
| `model.8.m.0.cv2` | `Conv` | 0 |
| `model.8.m.0.cv2.conv` | `Conv2d` | 8192 |
| `model.8.m.0.cv2.bn` | `BatchNorm2d` | 128 |
| `model.8.m.0.cv3` | `Conv` | 0 |
| `model.8.m.0.cv3.conv` | `Conv2d` | 16384 |
| `model.8.m.0.cv3.bn` | `BatchNorm2d` | 256 |
| `model.8.m.0.m` | `Sequential` | 0 |
| `model.8.m.0.m.0` | `Bottleneck` | 0 |
| `model.8.m.0.m.0.cv1` | `Conv` | 0 |
| `model.8.m.0.m.0.cv1.conv` | `Conv2d` | 36864 |
| `model.8.m.0.m.0.cv1.bn` | `BatchNorm2d` | 128 |
| `model.8.m.0.m.0.cv2` | `Conv` | 0 |
| `model.8.m.0.m.0.cv2.conv` | `Conv2d` | 36864 |
| `model.8.m.0.m.0.cv2.bn` | `BatchNorm2d` | 128 |
| `model.8.m.0.m.1` | `Bottleneck` | 0 |
| `model.8.m.0.m.1.cv1` | `Conv` | 0 |
| `model.8.m.0.m.1.cv1.conv` | `Conv2d` | 36864 |
| `model.8.m.0.m.1.cv1.bn` | `BatchNorm2d` | 128 |
| `model.8.m.0.m.1.cv2` | `Conv` | 0 |
| `model.8.m.0.m.1.cv2.conv` | `Conv2d` | 36864 |
| `model.8.m.0.m.1.cv2.bn` | `BatchNorm2d` | 128 |
| `model.9` | `SPPF` | 0 |
| `model.9.cv1` | `Conv` | 0 |
| `model.9.cv1.conv` | `Conv2d` | 32768 |
| `model.9.cv1.bn` | `BatchNorm2d` | 256 |
| `model.9.cv1.act` | `Identity` | 0 |
| `model.9.cv2` | `Conv` | 0 |
| `model.9.cv2.conv` | `Conv2d` | 131072 |
| `model.9.cv2.bn` | `BatchNorm2d` | 512 |
| `model.9.m` | `MaxPool2d` | 0 |
| `model.10` | `C2PSA` | 0 |
| `model.10.cv1` | `Conv` | 0 |
| `model.10.cv1.conv` | `Conv2d` | 65536 |
| `model.10.cv1.bn` | `BatchNorm2d` | 512 |
| `model.10.cv2` | `Conv` | 0 |
| `model.10.cv2.conv` | `Conv2d` | 65536 |
| `model.10.cv2.bn` | `BatchNorm2d` | 512 |
| `model.10.m` | `Sequential` | 0 |
| `model.10.m.0` | `PSABlock` | 0 |
| `model.10.m.0.attn` | `Attention` | 0 |
| `model.10.m.0.attn.qkv` | `Conv` | 0 |
| `model.10.m.0.attn.qkv.conv` | `Conv2d` | 32768 |
| `model.10.m.0.attn.qkv.bn` | `BatchNorm2d` | 512 |
| `model.10.m.0.attn.qkv.act` | `Identity` | 0 |
| `model.10.m.0.attn.proj` | `Conv` | 0 |
| `model.10.m.0.attn.proj.conv` | `Conv2d` | 16384 |
| `model.10.m.0.attn.proj.bn` | `BatchNorm2d` | 256 |
| `model.10.m.0.attn.proj.act` | `Identity` | 0 |
| `model.10.m.0.attn.pe` | `Conv` | 0 |
| `model.10.m.0.attn.pe.conv` | `Conv2d` | 1152 |
| `model.10.m.0.attn.pe.bn` | `BatchNorm2d` | 256 |
| `model.10.m.0.attn.pe.act` | `Identity` | 0 |
| `model.10.m.0.ffn` | `Sequential` | 0 |
| `model.10.m.0.ffn.0` | `Conv` | 0 |
| `model.10.m.0.ffn.0.conv` | `Conv2d` | 32768 |
| `model.10.m.0.ffn.0.bn` | `BatchNorm2d` | 512 |
| `model.10.m.0.ffn.1` | `Conv` | 0 |
| `model.10.m.0.ffn.1.conv` | `Conv2d` | 32768 |
| `model.10.m.0.ffn.1.bn` | `BatchNorm2d` | 256 |
| `model.10.m.0.ffn.1.act` | `Identity` | 0 |
| `model.11` | `Upsample` | 0 |
| `model.12` | `Concat` | 0 |
| `model.13` | `C3k2` | 0 |
| `model.13.cv1` | `Conv` | 0 |
| `model.13.cv1.conv` | `Conv2d` | 49152 |
| `model.13.cv1.bn` | `BatchNorm2d` | 256 |
| `model.13.cv2` | `Conv` | 0 |
| `model.13.cv2.conv` | `Conv2d` | 24576 |
| `model.13.cv2.bn` | `BatchNorm2d` | 256 |
| `model.13.m` | `ModuleList` | 0 |
| `model.13.m.0` | `C3k` | 0 |
| `model.13.m.0.cv1` | `Conv` | 0 |
| `model.13.m.0.cv1.conv` | `Conv2d` | 2048 |
| `model.13.m.0.cv1.bn` | `BatchNorm2d` | 64 |
| `model.13.m.0.cv2` | `Conv` | 0 |
| `model.13.m.0.cv2.conv` | `Conv2d` | 2048 |
| `model.13.m.0.cv2.bn` | `BatchNorm2d` | 64 |
| `model.13.m.0.cv3` | `Conv` | 0 |
| `model.13.m.0.cv3.conv` | `Conv2d` | 4096 |
| `model.13.m.0.cv3.bn` | `BatchNorm2d` | 128 |
| `model.13.m.0.m` | `Sequential` | 0 |
| `model.13.m.0.m.0` | `Bottleneck` | 0 |
| `model.13.m.0.m.0.cv1` | `Conv` | 0 |
| `model.13.m.0.m.0.cv1.conv` | `Conv2d` | 9216 |
| `model.13.m.0.m.0.cv1.bn` | `BatchNorm2d` | 64 |
| `model.13.m.0.m.0.cv2` | `Conv` | 0 |
| `model.13.m.0.m.0.cv2.conv` | `Conv2d` | 9216 |
| `model.13.m.0.m.0.cv2.bn` | `BatchNorm2d` | 64 |
| `model.13.m.0.m.1` | `Bottleneck` | 0 |
| `model.13.m.0.m.1.cv1` | `Conv` | 0 |
| `model.13.m.0.m.1.cv1.conv` | `Conv2d` | 9216 |
| `model.13.m.0.m.1.cv1.bn` | `BatchNorm2d` | 64 |
| `model.13.m.0.m.1.cv2` | `Conv` | 0 |
| `model.13.m.0.m.1.cv2.conv` | `Conv2d` | 9216 |
| `model.13.m.0.m.1.cv2.bn` | `BatchNorm2d` | 64 |
| `model.14` | `Upsample` | 0 |
| `model.15` | `Concat` | 0 |
| `model.16` | `C3k2` | 0 |
| `model.16.cv1` | `Conv` | 0 |
| `model.16.cv1.conv` | `Conv2d` | 16384 |
| `model.16.cv1.bn` | `BatchNorm2d` | 128 |
| `model.16.cv2` | `Conv` | 0 |
| `model.16.cv2.conv` | `Conv2d` | 6144 |
| `model.16.cv2.bn` | `BatchNorm2d` | 128 |
| `model.16.m` | `ModuleList` | 0 |
| `model.16.m.0` | `C3k` | 0 |
| `model.16.m.0.cv1` | `Conv` | 0 |
| `model.16.m.0.cv1.conv` | `Conv2d` | 512 |
| `model.16.m.0.cv1.bn` | `BatchNorm2d` | 32 |
| `model.16.m.0.cv2` | `Conv` | 0 |
| `model.16.m.0.cv2.conv` | `Conv2d` | 512 |
| `model.16.m.0.cv2.bn` | `BatchNorm2d` | 32 |
| `model.16.m.0.cv3` | `Conv` | 0 |
| `model.16.m.0.cv3.conv` | `Conv2d` | 1024 |
| `model.16.m.0.cv3.bn` | `BatchNorm2d` | 64 |
| `model.16.m.0.m` | `Sequential` | 0 |
| `model.16.m.0.m.0` | `Bottleneck` | 0 |
| `model.16.m.0.m.0.cv1` | `Conv` | 0 |
| `model.16.m.0.m.0.cv1.conv` | `Conv2d` | 2304 |
| `model.16.m.0.m.0.cv1.bn` | `BatchNorm2d` | 32 |
| `model.16.m.0.m.0.cv2` | `Conv` | 0 |
| `model.16.m.0.m.0.cv2.conv` | `Conv2d` | 2304 |
| `model.16.m.0.m.0.cv2.bn` | `BatchNorm2d` | 32 |
| `model.16.m.0.m.1` | `Bottleneck` | 0 |
| `model.16.m.0.m.1.cv1` | `Conv` | 0 |
| `model.16.m.0.m.1.cv1.conv` | `Conv2d` | 2304 |
| `model.16.m.0.m.1.cv1.bn` | `BatchNorm2d` | 32 |
| `model.16.m.0.m.1.cv2` | `Conv` | 0 |
| `model.16.m.0.m.1.cv2.conv` | `Conv2d` | 2304 |
| `model.16.m.0.m.1.cv2.bn` | `BatchNorm2d` | 32 |
| `model.17` | `Conv` | 0 |
| `model.17.conv` | `Conv2d` | 36864 |
| `model.17.bn` | `BatchNorm2d` | 128 |
| `model.18` | `Concat` | 0 |
| `model.19` | `C3k2` | 0 |
| `model.19.cv1` | `Conv` | 0 |
| `model.19.cv1.conv` | `Conv2d` | 24576 |
| `model.19.cv1.bn` | `BatchNorm2d` | 256 |
| `model.19.cv2` | `Conv` | 0 |
| `model.19.cv2.conv` | `Conv2d` | 24576 |
| `model.19.cv2.bn` | `BatchNorm2d` | 256 |
| `model.19.m` | `ModuleList` | 0 |
| `model.19.m.0` | `C3k` | 0 |
| `model.19.m.0.cv1` | `Conv` | 0 |
| `model.19.m.0.cv1.conv` | `Conv2d` | 2048 |
| `model.19.m.0.cv1.bn` | `BatchNorm2d` | 64 |
| `model.19.m.0.cv2` | `Conv` | 0 |
| `model.19.m.0.cv2.conv` | `Conv2d` | 2048 |
| `model.19.m.0.cv2.bn` | `BatchNorm2d` | 64 |
| `model.19.m.0.cv3` | `Conv` | 0 |
| `model.19.m.0.cv3.conv` | `Conv2d` | 4096 |
| `model.19.m.0.cv3.bn` | `BatchNorm2d` | 128 |
| `model.19.m.0.m` | `Sequential` | 0 |
| `model.19.m.0.m.0` | `Bottleneck` | 0 |
| `model.19.m.0.m.0.cv1` | `Conv` | 0 |
| `model.19.m.0.m.0.cv1.conv` | `Conv2d` | 9216 |
| `model.19.m.0.m.0.cv1.bn` | `BatchNorm2d` | 64 |
| `model.19.m.0.m.0.cv2` | `Conv` | 0 |
| `model.19.m.0.m.0.cv2.conv` | `Conv2d` | 9216 |
| `model.19.m.0.m.0.cv2.bn` | `BatchNorm2d` | 64 |
| `model.19.m.0.m.1` | `Bottleneck` | 0 |
| `model.19.m.0.m.1.cv1` | `Conv` | 0 |
| `model.19.m.0.m.1.cv1.conv` | `Conv2d` | 9216 |
| `model.19.m.0.m.1.cv1.bn` | `BatchNorm2d` | 64 |
| `model.19.m.0.m.1.cv2` | `Conv` | 0 |
| `model.19.m.0.m.1.cv2.conv` | `Conv2d` | 9216 |
| `model.19.m.0.m.1.cv2.bn` | `BatchNorm2d` | 64 |
| `model.20` | `Conv` | 0 |
| `model.20.conv` | `Conv2d` | 147456 |
| `model.20.bn` | `BatchNorm2d` | 256 |
| `model.21` | `Concat` | 0 |
| `model.22` | `C3k2` | 0 |
| `model.22.cv1` | `Conv` | 0 |
| `model.22.cv1.conv` | `Conv2d` | 98304 |
| `model.22.cv1.bn` | `BatchNorm2d` | 512 |
| `model.22.cv2` | `Conv` | 0 |
| `model.22.cv2.conv` | `Conv2d` | 98304 |
| `model.22.cv2.bn` | `BatchNorm2d` | 512 |
| `model.22.m` | `ModuleList` | 0 |
| `model.22.m.0` | `Sequential` | 0 |
| `model.22.m.0.0` | `Bottleneck` | 0 |
| `model.22.m.0.0.cv1` | `Conv` | 0 |
| `model.22.m.0.0.cv1.conv` | `Conv2d` | 73728 |
| `model.22.m.0.0.cv1.bn` | `BatchNorm2d` | 128 |
| `model.22.m.0.0.cv2` | `Conv` | 0 |
| `model.22.m.0.0.cv2.conv` | `Conv2d` | 73728 |
| `model.22.m.0.0.cv2.bn` | `BatchNorm2d` | 256 |
| `model.22.m.0.1` | `PSABlock` | 0 |
| `model.22.m.0.1.attn` | `Attention` | 0 |
| `model.22.m.0.1.attn.qkv` | `Conv` | 0 |
| `model.22.m.0.1.attn.qkv.conv` | `Conv2d` | 32768 |
| `model.22.m.0.1.attn.qkv.bn` | `BatchNorm2d` | 512 |
| `model.22.m.0.1.attn.qkv.act` | `Identity` | 0 |
| `model.22.m.0.1.attn.proj` | `Conv` | 0 |
| `model.22.m.0.1.attn.proj.conv` | `Conv2d` | 16384 |
| `model.22.m.0.1.attn.proj.bn` | `BatchNorm2d` | 256 |
| `model.22.m.0.1.attn.proj.act` | `Identity` | 0 |
| `model.22.m.0.1.attn.pe` | `Conv` | 0 |
| `model.22.m.0.1.attn.pe.conv` | `Conv2d` | 1152 |
| `model.22.m.0.1.attn.pe.bn` | `BatchNorm2d` | 256 |
| `model.22.m.0.1.attn.pe.act` | `Identity` | 0 |
| `model.22.m.0.1.ffn` | `Sequential` | 0 |
| `model.22.m.0.1.ffn.0` | `Conv` | 0 |
| `model.22.m.0.1.ffn.0.conv` | `Conv2d` | 32768 |
| `model.22.m.0.1.ffn.0.bn` | `BatchNorm2d` | 512 |
| `model.22.m.0.1.ffn.1` | `Conv` | 0 |
| `model.22.m.0.1.ffn.1.conv` | `Conv2d` | 32768 |
| `model.22.m.0.1.ffn.1.bn` | `BatchNorm2d` | 256 |
| `model.22.m.0.1.ffn.1.act` | `Identity` | 0 |
| `model.23` | `Detect` | 0 |
| `model.23.cv2` | `ModuleList` | 0 |
| `model.23.cv2.0` | `Sequential` | 0 |
| `model.23.cv2.0.0` | `Conv` | 0 |
| `model.23.cv2.0.0.conv` | `Conv2d` | 9216 |
| `model.23.cv2.0.0.bn` | `BatchNorm2d` | 32 |
| `model.23.cv2.0.1` | `Conv` | 0 |
| `model.23.cv2.0.1.conv` | `Conv2d` | 2304 |
| `model.23.cv2.0.1.bn` | `BatchNorm2d` | 32 |
| `model.23.cv2.0.2` | `Conv2d` | 68 |
| `model.23.cv2.1` | `Sequential` | 0 |
| `model.23.cv2.1.0` | `Conv` | 0 |
| `model.23.cv2.1.0.conv` | `Conv2d` | 18432 |
| `model.23.cv2.1.0.bn` | `BatchNorm2d` | 32 |
| `model.23.cv2.1.1` | `Conv` | 0 |
| `model.23.cv2.1.1.conv` | `Conv2d` | 2304 |
| `model.23.cv2.1.1.bn` | `BatchNorm2d` | 32 |
| `model.23.cv2.1.2` | `Conv2d` | 68 |
| `model.23.cv2.2` | `Sequential` | 0 |
| `model.23.cv2.2.0` | `Conv` | 0 |
| `model.23.cv2.2.0.conv` | `Conv2d` | 36864 |
| `model.23.cv2.2.0.bn` | `BatchNorm2d` | 32 |
| `model.23.cv2.2.1` | `Conv` | 0 |
| `model.23.cv2.2.1.conv` | `Conv2d` | 2304 |
| `model.23.cv2.2.1.bn` | `BatchNorm2d` | 32 |
| `model.23.cv2.2.2` | `Conv2d` | 68 |
| `model.23.cv3` | `ModuleList` | 0 |
| `model.23.cv3.0` | `Sequential` | 0 |
| `model.23.cv3.0.0` | `Sequential` | 0 |
| `model.23.cv3.0.0.0` | `DWConv` | 0 |
| `model.23.cv3.0.0.0.conv` | `Conv2d` | 576 |
| `model.23.cv3.0.0.0.bn` | `BatchNorm2d` | 128 |
| `model.23.cv3.0.0.1` | `Conv` | 0 |
| `model.23.cv3.0.0.1.conv` | `Conv2d` | 4096 |
| `model.23.cv3.0.0.1.bn` | `BatchNorm2d` | 128 |
| `model.23.cv3.0.1` | `Sequential` | 0 |
| `model.23.cv3.0.1.0` | `DWConv` | 0 |
| `model.23.cv3.0.1.0.conv` | `Conv2d` | 576 |
| `model.23.cv3.0.1.0.bn` | `BatchNorm2d` | 128 |
| `model.23.cv3.0.1.1` | `Conv` | 0 |
| `model.23.cv3.0.1.1.conv` | `Conv2d` | 4096 |
| `model.23.cv3.0.1.1.bn` | `BatchNorm2d` | 128 |
| `model.23.cv3.0.2` | `Conv2d` | 65 |
| `model.23.cv3.1` | `Sequential` | 0 |
| `model.23.cv3.1.0` | `Sequential` | 0 |
| `model.23.cv3.1.0.0` | `DWConv` | 0 |
| `model.23.cv3.1.0.0.conv` | `Conv2d` | 1152 |
| `model.23.cv3.1.0.0.bn` | `BatchNorm2d` | 256 |
| `model.23.cv3.1.0.1` | `Conv` | 0 |
| `model.23.cv3.1.0.1.conv` | `Conv2d` | 8192 |
| `model.23.cv3.1.0.1.bn` | `BatchNorm2d` | 128 |
| `model.23.cv3.1.1` | `Sequential` | 0 |
| `model.23.cv3.1.1.0` | `DWConv` | 0 |
| `model.23.cv3.1.1.0.conv` | `Conv2d` | 576 |
| `model.23.cv3.1.1.0.bn` | `BatchNorm2d` | 128 |
| `model.23.cv3.1.1.1` | `Conv` | 0 |
| `model.23.cv3.1.1.1.conv` | `Conv2d` | 4096 |
| `model.23.cv3.1.1.1.bn` | `BatchNorm2d` | 128 |
| `model.23.cv3.1.2` | `Conv2d` | 65 |
| `model.23.cv3.2` | `Sequential` | 0 |
| `model.23.cv3.2.0` | `Sequential` | 0 |
| `model.23.cv3.2.0.0` | `DWConv` | 0 |
| `model.23.cv3.2.0.0.conv` | `Conv2d` | 2304 |
| `model.23.cv3.2.0.0.bn` | `BatchNorm2d` | 512 |
| `model.23.cv3.2.0.1` | `Conv` | 0 |
| `model.23.cv3.2.0.1.conv` | `Conv2d` | 16384 |
| `model.23.cv3.2.0.1.bn` | `BatchNorm2d` | 128 |
| `model.23.cv3.2.1` | `Sequential` | 0 |
| `model.23.cv3.2.1.0` | `DWConv` | 0 |
| `model.23.cv3.2.1.0.conv` | `Conv2d` | 576 |
| `model.23.cv3.2.1.0.bn` | `BatchNorm2d` | 128 |
| `model.23.cv3.2.1.1` | `Conv` | 0 |
| `model.23.cv3.2.1.1.conv` | `Conv2d` | 4096 |
| `model.23.cv3.2.1.1.bn` | `BatchNorm2d` | 128 |
| `model.23.cv3.2.2` | `Conv2d` | 65 |
| `model.23.dfl` | `Identity` | 0 |
| `model.23.one2one_cv2` | `ModuleList` | 0 |
| `model.23.one2one_cv2.0` | `Sequential` | 0 |
| `model.23.one2one_cv2.0.0` | `Conv` | 0 |
| `model.23.one2one_cv2.0.0.conv` | `Conv2d` | 9216 |
| `model.23.one2one_cv2.0.0.bn` | `BatchNorm2d` | 32 |
| `model.23.one2one_cv2.0.0.act` | `SiLU` | 0 |
| `model.23.one2one_cv2.0.1` | `Conv` | 0 |
| `model.23.one2one_cv2.0.1.conv` | `Conv2d` | 2304 |
| `model.23.one2one_cv2.0.1.bn` | `BatchNorm2d` | 32 |
| `model.23.one2one_cv2.0.2` | `Conv2d` | 68 |
| `model.23.one2one_cv2.1` | `Sequential` | 0 |
| `model.23.one2one_cv2.1.0` | `Conv` | 0 |
| `model.23.one2one_cv2.1.0.conv` | `Conv2d` | 18432 |
| `model.23.one2one_cv2.1.0.bn` | `BatchNorm2d` | 32 |
| `model.23.one2one_cv2.1.1` | `Conv` | 0 |
| `model.23.one2one_cv2.1.1.conv` | `Conv2d` | 2304 |
| `model.23.one2one_cv2.1.1.bn` | `BatchNorm2d` | 32 |
| `model.23.one2one_cv2.1.2` | `Conv2d` | 68 |
| `model.23.one2one_cv2.2` | `Sequential` | 0 |
| `model.23.one2one_cv2.2.0` | `Conv` | 0 |
| `model.23.one2one_cv2.2.0.conv` | `Conv2d` | 36864 |
| `model.23.one2one_cv2.2.0.bn` | `BatchNorm2d` | 32 |
| `model.23.one2one_cv2.2.1` | `Conv` | 0 |
| `model.23.one2one_cv2.2.1.conv` | `Conv2d` | 2304 |
| `model.23.one2one_cv2.2.1.bn` | `BatchNorm2d` | 32 |
| `model.23.one2one_cv2.2.2` | `Conv2d` | 68 |
| `model.23.one2one_cv3` | `ModuleList` | 0 |
| `model.23.one2one_cv3.0` | `Sequential` | 0 |
| `model.23.one2one_cv3.0.0` | `Sequential` | 0 |
| `model.23.one2one_cv3.0.0.0` | `DWConv` | 0 |
| `model.23.one2one_cv3.0.0.0.conv` | `Conv2d` | 576 |
| `model.23.one2one_cv3.0.0.0.bn` | `BatchNorm2d` | 128 |
| `model.23.one2one_cv3.0.0.0.act` | `SiLU` | 0 |
| `model.23.one2one_cv3.0.0.1` | `Conv` | 0 |
| `model.23.one2one_cv3.0.0.1.conv` | `Conv2d` | 4096 |
| `model.23.one2one_cv3.0.0.1.bn` | `BatchNorm2d` | 128 |
| `model.23.one2one_cv3.0.1` | `Sequential` | 0 |
| `model.23.one2one_cv3.0.1.0` | `DWConv` | 0 |
| `model.23.one2one_cv3.0.1.0.conv` | `Conv2d` | 576 |
| `model.23.one2one_cv3.0.1.0.bn` | `BatchNorm2d` | 128 |
| `model.23.one2one_cv3.0.1.1` | `Conv` | 0 |
| `model.23.one2one_cv3.0.1.1.conv` | `Conv2d` | 4096 |
| `model.23.one2one_cv3.0.1.1.bn` | `BatchNorm2d` | 128 |
| `model.23.one2one_cv3.0.2` | `Conv2d` | 65 |
| `model.23.one2one_cv3.1` | `Sequential` | 0 |
| `model.23.one2one_cv3.1.0` | `Sequential` | 0 |
| `model.23.one2one_cv3.1.0.0` | `DWConv` | 0 |
| `model.23.one2one_cv3.1.0.0.conv` | `Conv2d` | 1152 |
| `model.23.one2one_cv3.1.0.0.bn` | `BatchNorm2d` | 256 |
| `model.23.one2one_cv3.1.0.1` | `Conv` | 0 |
| `model.23.one2one_cv3.1.0.1.conv` | `Conv2d` | 8192 |
| `model.23.one2one_cv3.1.0.1.bn` | `BatchNorm2d` | 128 |
| `model.23.one2one_cv3.1.1` | `Sequential` | 0 |
| `model.23.one2one_cv3.1.1.0` | `DWConv` | 0 |
| `model.23.one2one_cv3.1.1.0.conv` | `Conv2d` | 576 |
| `model.23.one2one_cv3.1.1.0.bn` | `BatchNorm2d` | 128 |
| `model.23.one2one_cv3.1.1.1` | `Conv` | 0 |
| `model.23.one2one_cv3.1.1.1.conv` | `Conv2d` | 4096 |
| `model.23.one2one_cv3.1.1.1.bn` | `BatchNorm2d` | 128 |
| `model.23.one2one_cv3.1.2` | `Conv2d` | 65 |
| `model.23.one2one_cv3.2` | `Sequential` | 0 |
| `model.23.one2one_cv3.2.0` | `Sequential` | 0 |
| `model.23.one2one_cv3.2.0.0` | `DWConv` | 0 |
| `model.23.one2one_cv3.2.0.0.conv` | `Conv2d` | 2304 |
| `model.23.one2one_cv3.2.0.0.bn` | `BatchNorm2d` | 512 |
| `model.23.one2one_cv3.2.0.1` | `Conv` | 0 |
| `model.23.one2one_cv3.2.0.1.conv` | `Conv2d` | 16384 |
| `model.23.one2one_cv3.2.0.1.bn` | `BatchNorm2d` | 128 |
| `model.23.one2one_cv3.2.1` | `Sequential` | 0 |
| `model.23.one2one_cv3.2.1.0` | `DWConv` | 0 |
| `model.23.one2one_cv3.2.1.0.conv` | `Conv2d` | 576 |
| `model.23.one2one_cv3.2.1.0.bn` | `BatchNorm2d` | 128 |
| `model.23.one2one_cv3.2.1.1` | `Conv` | 0 |
| `model.23.one2one_cv3.2.1.1.conv` | `Conv2d` | 4096 |
| `model.23.one2one_cv3.2.1.1.bn` | `BatchNorm2d` | 128 |
| `model.23.one2one_cv3.2.2` | `Conv2d` | 65 |
