# Third-Party Notices

## NX-AI xLSTM

`src/models.py` contains a project-local, modified and simplified native
PyTorch mLSTM implementation derived from concepts and source code in the
official NX-AI xLSTM project:

- Source: https://github.com/NX-AI/xlstm
- Referenced version: 2.0.5
- Copyright: NXAI GmbH and its affiliates 2024
- License: Apache License 2.0

The project-local version removes sLSTM, custom CUDA compilation,
language-model components and generation APIs. It retains a stabilized parallel
mLSTM memory equation and adapts the block stack for seven-class sentence
classification.

The Apache License 2.0 is available at:

https://www.apache.org/licenses/LICENSE-2.0

