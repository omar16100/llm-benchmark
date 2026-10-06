# Model regression evidence, 6 October 2026

These packages contain small reproduction scripts, token evidence, and test records.
They contain no model weights or private runtime configuration.

| Package | Result |
| --- | --- |
| [Gemma](gemma) | Stock MTP difference reproduced; cache diagnosis and limited exact reference |
| [Ling](ling) | Custom KDA adapter patch with synthetic CPU comparisons |
| [TensorFold](tensorfold) | Native read-error leak reproduced and corrected; all 54 regression cases pass |

The [receipt validator](../../tools/receipt_validator) checks a separate portable receipt schema.
These raw evidence packages do not claim conformance to that schema.
