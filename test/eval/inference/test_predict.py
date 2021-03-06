import torch
from mp.eval.inference.predict import arg_max, softmax
from mp.data.pytorch.transformation import per_label_channel

def test_argmax_pred():
    output = torch.tensor([[[[0.3, 0.7], [0.2, 0.1]], [[8., .0], [.03, 0.4]], 
        [[5.7, .1], [.55, 0.45]]], [[[0.3, 0.7], [0.2, 0.1]], [[8., .0], 
        [.03, 0.4]], [[5.7, .1], [.55, 0.45]]]])
    target = torch.tensor([[[1, 0], [2, 2]], [[1, 0], [2, 2]]])
    assert output.numpy().shape == (2,3,2,2)
    pred = arg_max(output).numpy()
    assert pred.shape == (2,2,2)
    assert (pred == target.numpy()).all()

def test_argmax_another_channel_output():
    output = torch.tensor([[[0.3, 0.7], [0.2, 0.1]], [[8., .0], [.03, 0.4]], 
        [[5.7, .1], [.55, 0.45]]])
    target = torch.tensor([[1, 0], [2, 2]])
    assert output.numpy().shape == (3,2,2)
    pred = arg_max(output, channel_dim=0).numpy()
    assert pred.shape == (2,2)
    assert (pred == target.numpy()).all()

def test_softmax():
    output = torch.tensor([[[[3., 1.], [0.2, 0.05]], [[4., .0], [0.8, 0.4]], 
        [[3., 9.], [0., 0.45]]],[[[3., 1.], [0.2, 0.05]], [[4., .0], [0.8, 0.4]], 
        [[3., 9.], [0., 0.45]]]])
    softmaxed_output = softmax(output).numpy()
    assert softmaxed_output.shape == (2,3,2,2)
    for k in [0,1]:
        for i, j in [(0,0),(0,1),(1,0),(1,1)]:
            assert abs(1 - sum(x[i][j] for x in softmaxed_output[k])) < 0.0001

def test_softmax_another_channel_output():
    output = torch.tensor([[[3., 1.], [0.2, 0.05]], [[4., .0], [0.8, 0.4]], 
        [[3., 9.], [0., 0.45]]])
    softmaxed_output = softmax(output, channel_dim=0).numpy()
    assert softmaxed_output.shape == (3,2,2)
    for i, j in [(0,0),(0,1),(1,0),(1,1)]:
        assert abs(1 - sum(x[i][j] for x in softmaxed_output)) < 0.0001

def test_per_label_channel_to_pred():
    A_1=[[0,0,0,0,0,0,0],
        [0,1,3,3,0,1,0],
        [0,0,3,1,1,2,2],
        [0,0,0,1,1,2,2]]
    A_2=[[0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,3,3,3],
        [2,2,1,1,1,1,0]]
    A_3=[[0,0,0,0,0,0,0],
        [0,0,0,1,0,0,0],
        [2,0,1,1,0,0,0],
        [2,0,0,0,0,0,0]]
    A_4=[[1,1,1,0,0,0,0],
        [0,0,0,0,2,2,0],
        [0,2,0,0,2,2,0],
        [0,2,0,0,3,3,3]]
    a = torch.tensor([A_1, A_2, A_3, A_4])
    a = a.unsqueeze(0)
    per_label_channel_a = per_label_channel(a, nr_labels=4, channel_dim=0)
    a_pred = arg_max(per_label_channel_a, channel_dim=0).numpy()
    assert (a.numpy() == a_pred).all()