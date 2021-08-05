import torch


def m_pearsonr(output, target):
    x = output
    y = target

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    pr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))

    return pr


def accloss(output, target):
    pr = m_pearsonr(output, target)
    return pr


if __name__ == '__main__':
    output = torch.rand(1, 5)
    target = torch.rand(1, 5)
    pytt = accloss(output, target)
