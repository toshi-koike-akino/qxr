# Quantum & Mixed Reality (qXR)

This project is an attempt to integrate **quantum machine learning (QML)** and **mixed reality (XR)**, for the hackathon event [QHack 2023](https://github.com/XanaduAI/QHack).

![qxr](./images/qxr.png)
<!--
![qusic](./images/qxr.png)
-->

# Prerequisite

We may use the package manager [pip](https://pip.pypa.io/en/stable/) for python=3.9.
We use [Pennylane](https://pennylane.ai/) for QML.

```bash
pip install pennylane
pip install argparse
pip install plotly 
pip install bpy # Blender API
pip install openxr # OpenXR
```

# Procedural 3D Modeling

## Generative QML

Generative artificial intelligence (AI) technologies such as VAE, GAN, DDPM, and ChatGPT have shown impressive performance recently for artistic work, linguistic work, etc. (e.g., refer [awesome list](https://github.com/yzy1996/Awesome-Generative-Model)) We use generative QML framework for procedural 3D modeling in XR scenarios. TBD...

## QML Ansatz

We use QML ansatz based on [SimplifiedTwoDesign](https://docs.pennylane.ai/en/stable/code/api/pennylane.SimplifiedTwoDesign.html) to train. TBD...

## QML Visualization

We also provide a visualization tool chain of QML model. We use plotly for interactive 3D visualization of QML model. TBD...

## Quantum in Mixed Reality

We use Blender for XR experience of QML. TBD...

# GPU/QPU Devices

## NVIDIA GPU

It is straightforward to use a (or multiple) **graphic processing unit (GPU)** for accelerating QML training/testing. [Pennylane lightening](https://pennylane.ai/blog/2022/07/lightning-fast-simulations-with-pennylane-and-the-nvidia-cuquantum-sdk/) can speed-up quantum simulations through the use of [NVIDIA cuQuantum](https://developer.nvidia.com/cuquantum-sdk). You just need to change the device as follows:

```bash
python main.py --dev lightening.gpu
```

Below is a benchmark comparison with NVIDIA A100: TBD...

## IBM QPU

It is straightforward to use a real **quantum processing unit (QPU)** for testing our *qusic*.
For example, we may use [IBM Q Experience](https://pennylaneqiskit.readthedocs.io/en/latest/devices/ibmq.html).
You may specify the account token via [Pennylane configulation file](https://pennylane.readthedocs.io/en/latest/introduction/configuration.html), and a scpecific backend of real QPU, such as 'ibmq_london'.
To run our [qusic.py](./qusic.py) on a real quantum computer, you just need to change the device as follows:

```bash
pip install pennylane-qiskit # qiskit plugin
python main.py --dev qiskit.ibmq
```

## AWS Braket

We can use different backends such as [Amazon braket](https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/).
For example, we may run as

```bash
pip install amazon-braket-pennylane-plugin # AWS plugin
python main.py --dev braket.aws.qubit
```

Further examples and tutorials are found at [amazon-braket-example](https://github.com/aws/amazon-braket-examples) and [README](https://github.com/aws/amazon-braket-sdk-python/blob/main/README.md).

# License

[MIT](https://choosealicense.com/licenses/mit/).
Copyright (c) 2023 Toshi Koike-Akino. This project is provided 'as-is', without warranty of any kind. In no event shall the authors be liable for any claims, damages, or other such variants.
