import torch


def compute_rotations(images, image_size, task_dict, targets, task_id):
    images_rot = torch.stack([torch.rot90(images, k, (2, 3)) for k in range(1, 4)], 1)
    images_rot = images_rot.view(-1, 3, image_size, image_size)
    targets_rot = torch.stack([(targets * 3 + k) + len(task_dict[task_id]) - 1 for k in range(1, 4)], 1).view(-1)
    return images_rot, targets_rot


@torch.no_grad()
def get_old_new_features(model, old_model, trn_loader, device):
    model.eval()
    old_model.eval()
    features_list, old_features_list = [], []
    for images, _ in trn_loader:
        images = images.to(device)
        _, features = model(images)
        _, old_features = old_model(images)
        features_list.append(features)
        old_features_list.append(old_features)
    return torch.cat(features_list), torch.cat(old_features_list)
