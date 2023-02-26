# plot_state.py
import pennylane as qml
import pennylane.numpy as np
import plotly
import argparse
import plotly.graph_objects as go
from functools import wraps
import PIL
import io


# sample args
def get_args():
    parser = argparse.ArgumentParser(__name__)
    parser.add_argument("--seed", default=32, type=int)
    parser.add_argument("--dev", default="default.qubit", type=str)
    parser.add_argument("--qubit", default=5, type=int, help="number of qubits")
    parser.add_argument("--layer", default=2, type=int)
    parser.add_argument("--ansatz", default="SimplifiedTwoDesign", type=str)
    parser.add_argument("--embed", default="AngleEmbedding", type=str)
    return parser.parse_args()


# sample QNN model
def get_qnode(args):
    dev = qml.device(args.dev, args.qubit)

    def circuit(features, params):
        getattr(qml, args.embed)(features, wires=dev.wires)
        getattr(qml, args.ansatz)(params[0], params[1], wires=dev.wires)
        return [qml.expval(qml.PauliZ(k)) for k in dev.wires]

    qnode = qml.QNode(circuit, dev)
    return qnode


# example use case
def main():
    # args
    args = get_args()
    print(args)

    # reseed if positive
    if args.seed > 0:
        np.random.seed(args.seed)

    # test qnode
    qnode = get_qnode(args)
    print(qnode)

    # test features/params
    features = np.random.random(size=args.qubit)  # assuming default embed
    shapes = getattr(qml, args.ansatz).shape(args.layer, args.qubit)
    params = [np.random.random(size=shape) for shape in shapes]

    # circuit draw
    print(qml.draw(qnode, expansion_strategy="gradient")(features, params))
    print(qml.draw(qnode, expansion_strategy="device")(features, params))

    # circuit draw_mpl
    fig, _ = qml.draw_mpl(qnode, style="sketch", expansion_strategy="gradient")(
        features, params
    )
    fig.savefig("draw_gradient.png")

    fig, _ = qml.draw_mpl(qnode, style="sketch", expansion_strategy="device")(
        features, params
    )
    fig.savefig("draw_device.png")

    # new draw tool to visualize intermediate states
    fig = draw_states(qnode)(features, params)
    fig.write_html("plot_state.html")  # save html

    # create animation gif
    animate_fig(fig)


def animate_fig(fig, nframe=30, dx=0.3, dy=0.3):
    def plot(k):
        # camera position
        theta = k / nframe * np.pi * 2  # angle
        pos = [1.25 + dx * np.cos(theta), 1.25 + dy * np.sin(theta), 1.25]
        print("frame", k, pos)

        # update figure layout
        eye = dict(x=pos[0], y=pos[1], z=pos[2])
        fig.update_layout(scene_camera_eye=eye)
        # fig.show()

        # save frame
        return PIL.Image.open(io.BytesIO(fig.to_image(format="png")))

    # motion loops
    frames = []
    for k in range(nframe):
        frame = plot(k)
        frames.append(frame)

    # save animation gif
    frames[0].save(
        "plot_state.gif",
        save_all=True,
        append_images=frames[1:],
        optimize=True,
        duration=50,
        loop=0,
    )


def draw_states(qnode, decimals=2, expansion_strategy="device", logscale=True):
    @wraps(qnode)
    def wrapper(*args, **kwargs):
        # tape construct
        original_expansion_strategy = getattr(qnode, "expansion_strategy", None)

        try:
            qnode.expansion_strategy = expansion_strategy or original_expansion_strategy
            tapes = qnode.construct(args, kwargs)
        finally:
            qnode.expansion_strategy = original_expansion_strategy

        # generate states over operation
        states, ops = state_evolve(qnode, decimals=decimals)

        # plot states
        fig = plot_states(states, ops, logscale=logscale)

        return fig

    return wrapper


def plot_states(
    states,
    ops,
    html=None,
    image=None,
    logscale=True,
    contour=False,
    colorscale="edge",
):
    epsilon = 1e-8
    probs = np.abs(states) ** 2
    probs = np.log(probs + epsilon) if logscale else probs
    ops = [ops] * states.shape[1]  # operation
    phase = np.angle(states)  # phase

    # surface trace
    trace = go.Surface(z=probs, customdata=ops, surfacecolor=phase)
    trace["colorscale"] = colorscale
    trace["hovertemplate"] = (
        "<b>Basis</b>: %{x} |%{x:0b}><br>"
        + "<b>Operation</b>: %{y} %{customdata}<br>"
        + "<b>"
        + ("Log " if logscale else "")
        + "Probability</b>: %{z:.6f}<br>"
        + "<b>Phase</b>: %{surfacecolor:.3f}"
        + "<extra></extra>"
    )
    trace.colorbar.title = "Phase"
    # https://plotly.com/python/builtin-colorscales/
    # cyclical: "hsv", "twilight", 'edge', 'icefire', 'phase', 'mrybm', 'mygbm'

    fig = go.Figure(data=[trace])
    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        scene=dict(
            xaxis_title="Basis",
            yaxis_title="Operations",
            zaxis_title="Log Probability" if logscale else "Probability",
        ),
    )

    if contour:
        fig.update_traces(
            contours_z=dict(
                show=True, usecolormap=True, highlightcolor="limegreen", project_z=True
            )
        )
    fig.show()

    # save to files
    if html:
        fig.write_html(html)
    if image:
        fig.write_image(image)
    return fig


# evolve state vector per operation in qtape
def state_evolve(qnode, progressive=False, decimals=2, verb=False):
    dev = qnode.device
    wires = qnode.device.wires
    tape = qnode.qtape.operations  # no obsearvables

    # size
    nwire = len(wires)  # number of wires
    nstate = 2**nwire  # size of state vector
    ntape = len(tape)  # number of operations in tape
    if verb:
        print("state_evolve", dev, wires, nstate, ntape)

    # loop over operations starting zero state |0>
    states = []
    names = []
    state = np.eye(1, nstate, requires_grad=False)
    for i, op in enumerate(tape):
        if verb:
            print(i, op, op.name, op.parameters, op.wires.tolist())

        if progressive:  # progressive state evolution
            prep = [qml.MottonenStatePreparation(np.ravel(state), wires=wires)]
            tape2 = qml.tape.QuantumTape([op], [qml.state()], prep=prep)
        else:  # resume from zero state
            tape2 = qml.tape.QuantumTape(tape[: i + 1], [qml.state()])

        # print(tape2.circuit)
        [state] = qml.execute([tape2], dev)

        # record state evolution
        op_wires = ",".join(map(str, op.wires.tolist()))
        op_params = ",".join(map(str, np.round(op.parameters, decimals).tolist()))
        names = names + [
            f"{op.name}[{op_wires}]" + (f"({op_params})" if len(op_params) else "")
        ]
        states = states + [np.ravel(state)]

    states = np.stack(states)

    # measurements
    if False:
        prep = [qml.MottonenStatePreparation(np.ravel(state), wires=wires)]
        # print(qnode.qtape.observables)
        tape2 = qml.tape.QuantumTape([], qnode.qtape.observables, prep=prep)
        # print(tape2.circuit)
        [out] = qml.execute([tape2], dev)
        # print(out)

    if verb:
        print(names)
        print(states.shape)
        print(states)

    # return states, names, out
    return states, names


if __name__ == "__main__":
    # test case
    main()
