from sodeep_master.sodeep import load_sorter, SpearmanLoss


def Spear(sorter_checkpoint_path, device="cuda"):
    criterion = SpearmanLoss(*load_sorter(sorter_checkpoint_path))
    criterion.to(device)
    return criterion
