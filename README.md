# Quantum & Mixed Reality (qXR)

This project is an attempt to integrate **quantum machine learning (QML)** and **mixed reality (XR)**, for the hackathon event [QHack 2023](https://github.com/XanaduAI/QHack).

![qxr](./images/qxr.png)
<!--
![qusic](./images/qxr.png)
-->

# Prerequisite

We may use the package manager [pip](https://pip.pypa.io/en/stable/) for python=3.10.
We use [Pennylane](https://pennylane.ai/) for QML.

```bash
pip install pennylane
pip install argparse
pip install plotly 
pip install bpy # Blender API (python=3.10 required)
pip install openxr # OpenXR
```

The above packages can be installed by [requirements.txt](requirements.txt)

```bash
pip install -r requirements.txt
```

# Procedural 3D Modeling

## Generative QML

Generative artificial intelligence (AI) technologies such as VAE, GAN, DDPM, and ChatGPT have shown impressive performance lately for artistic work, linguistic work, etc. (e.g., refer [awesome list](https://github.com/yzy1996/Awesome-Generative-Model)). We use generative QML framework for procedural 3D modeling in XR scenarios, based on [patched quantum WGAN-GP](https://arxiv.org/pdf/2212.11614.pdf) proposed in 2023 Jan. TBD...

## QML Ansatz

We use QML ansatz based on  to train. TBD...

## QML Visualization

We also provide a new visualization tool chain of QML model. We use [plotly](https://plotly.com/python/3d-charts/) for interactive 3D visualization of QML model.

### Built-in QML Visualization

Suppose we use a QML model having [AngleEmbedding](https://docs.pennylane.ai/en/stable/code/api/pennylane.AngleEmbedding.html?highlight=qml.AngleEmbedding) to encode features and [SimplifiedTwoDesign](https://docs.pennylane.ai/en/stable/code/api/pennylane.SimplifiedTwoDesign.html) to entangle the states via trainable params like below:

```python
dev = qml.device('default.qubit', 5)
# sample QNN model
@qml.qnode(dev)
def qnode(features, params):
    qml.AngleEmbedding(features, wires=dev.wires)
    qml.SimplifiedTwoDesign(params[0], params[1], wires=dev.wires)
    return [qml.expval(qml.PauliZ(k)) for k in dev.wires]
```

With a built-in [draw](https://docs.pennylane.ai/en/stable/code/api/pennylane.drawer.draw.html), we can draw it in 'gradient' or 'device' expansion strategy as follows:

```python
print(qml.draw(qnode, expansion_strategy="gradient")(features, params))
0: ─╭AngleEmbedding(M0)─╭SimplifiedTwoDesign─┤  <Z>
1: ─├AngleEmbedding(M0)─├SimplifiedTwoDesign─┤  <Z>
2: ─├AngleEmbedding(M0)─├SimplifiedTwoDesign─┤  <Z>
3: ─├AngleEmbedding(M0)─├SimplifiedTwoDesign─┤  <Z>
4: ─╰AngleEmbedding(M0)─╰SimplifiedTwoDesign─┤  <Z>

print(qml.draw(qnode, expansion_strategy="device")(features, params))
0: ──RX(0.86)──RY(0.82)─╭●──RY(0.09)──────────────╭●──RY(0.56)──────────────┤  <Z>
1: ──RX(0.37)──RY(0.10)─╰Z──RY(0.35)─╭●──RY(0.55)─╰Z──RY(0.77)─╭●──RY(0.90)─┤  <Z>
2: ──RX(0.56)──RY(0.93)─╭●──RY(0.66)─╰Z──RY(0.70)─╭●──RY(0.91)─╰Z──RY(0.46)─┤  <Z>
3: ──RX(0.96)──RY(0.61)─╰Z──RY(0.44)─╭●──RY(0.59)─╰Z──RY(0.09)─╭●──RY(0.45)─┤  <Z>
4: ──RX(0.74)──RY(0.60)──────────────╰Z──RY(0.05)──────────────╰Z──RY(1.00)─┤  <Z>
```

The built-in [draw_mpl](https://docs.pennylane.ai/en/stable/code/api/pennylane.drawer.draw_mpl.html) provides the corresponding plots like:

```python
fig, _ = qml.draw_mpl(qnode, style="sketch", expansion_strategy="device")(features, params)
fig.show()
```

![draw_device](./images/draw_device.png)

### New QML Visualization

The above drawers need specific 'features' and 'params' to visualize, but they are not used to show how those variables behave in QML model. We visualize the evolution of quantum states across the QML circuit by measureming [state](https://docs.pennylane.ai/en/stable/code/api/pennylane.state.html?highlight=qml.state) at intermediate circuits decomposed in a quantum [tape](https://docs.pennylane.ai/en/stable/code/api/pennylane.tape.QuantumTape.html).

Our visualization tool is written in [plot_state.py](plot_state.py). For example, draw_states() plots 3D interactive state vector evolutions:

```python
from plot_state import draw_states
fig = draw_states(qnode)(features, params)
```

![plot_state_hover](images/plot_state_hover.png)
As shown above, we can check probability and phase of each computatinal basis along each quantum operation at the hover text in plotly.
Please experience an example 3D interactive html [here](https://toshi-koike-akino.github.io/qxr/).

```python
animate_fig(fig) # create animation gif moving camera
```

![plot_state.gif](images/plot_state.gif)

## QML in Mixed Reality

We use Blender for XR experience of QML. TBD...

# GPU/QPU Devices

## NVIDIA GPU

It is straightforward to use a (or multiple) **graphic processing unit (GPU)** for accelerating QML training/testing. [Pennylane lightening](https://pennylane.ai/blog/2022/07/lightning-fast-simulations-with-pennylane-and-the-nvidia-cuquantum-sdk/) can speed-up quantum simulations through the use of [NVIDIA cuQuantum](https://developer.nvidia.com/cuquantum-sdk). You just need to change the device as follows:

```bash
pip install pennylane-lightning[gpu]
python main.py --dev lightening.gpu
```

For more information, please refer to the [PennyLane Lightning GPU plugin](https://docs.pennylane.ai/projects/lightning-gpu/en/latest/) documentation.

Below is a benchmark comparison with NVIDIA A100: TBD...

## IBM QPU

It is straightforward to use a real **quantum processing unit (QPU)** for testing our *qusic*.
For example, we may use [IBM Quantum plugin](https://pennylaneqiskit.readthedocs.io/en/latest/devices/ibmq.html).
You may specify the account token via [Pennylane configulation file](https://pennylane.readthedocs.io/en/latest/introduction/configuration.html), and a scpecific backend of real QPU, such as 'ibmq_london'.
To run our on a real QPU, you just need to change the device as follows:

```bash
pip install pennylane-qiskit # qiskit plugin
python main.py --dev qiskit.ibmq
```

## AWS Braket

We can use different backends such as [Amazon braket plugin](https://amazon-braket-pennylane-plugin-python.readthedocs.io/en/latest/).
For example, we may run as

```bash
pip install amazon-braket-pennylane-plugin # AWS plugin
python main.py --dev braket.aws.qubit
```

Further examples and tutorials are found at [amazon-braket-example](https://github.com/aws/amazon-braket-examples) and [README](https://github.com/aws/amazon-braket-sdk-python/blob/main/README.md).

# License

[MIT](https://choosealicense.com/licenses/mit/).
Copyright (c) 2023 Toshi Koike-Akino. This project is provided 'as-is', without warranty of any kind. In no event shall the authors be liable for any claims, damages, or other such variants.
