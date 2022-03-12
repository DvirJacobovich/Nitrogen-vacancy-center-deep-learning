import torch


def norm_data(data):
    """
    No need to keep the mean and std since measure is the input.
    :param data:
    """

    # data = torch.squeeze(measure_tensor)
    mean_ = torch.mean(data)
    std_ = torch.std(data)
    meas_scaled = data - mean_
    x1_scaled_normed = meas_scaled / std_
    return x1_scaled_normed

def norm_targets(target):
    # data = torch.squeeze(measure_tensor)
    mean_ = torch.mean(target)
    std_ = torch.std(target)
    meas_scaled = target - mean_
    x1_scaled_normed = meas_scaled / std_
    return x1_scaled_normed, mean_, std_


    # mean_target = torch.mean(target)
    # std_target = torch.std(target)
    # return ((target - torch.mean(target)) / torch.std(target)) / 4.5, mean_target, std_target

