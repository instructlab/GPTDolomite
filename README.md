# IBM Dolomite Engine

This repo contains code for IBM models, extracted from https://github.com/ibm-granite/dolomite-engine; all credit goes to the original authors:
- mayank31398

## Extracted Models

The main motivation for using these models is for padding-free. We perform a bare-bones extraction for code only needed for instructlab purposes. In particular we extracted the following:

**Supported Models**
- `hf_models/models/gpt_dolomite`: This is a padding-free transformer, that can be converted to/from `bigcode` and `llama` models.
