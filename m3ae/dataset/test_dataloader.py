from brats import get_datasets23_train_rf_forpretrain
import torch.utils.data
import random
from pathlib import Path

#train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=2, shuffle=True,
     #                                      num_workers=8, pin_memory=True, drop_last=True)


# Assuming the Brats class and other necessary imports are already available.

def test_brats_getitem():
    full_train_dataset, _, _, _ = get_datasets23_train_rf_forpretrain(10)

    print(f"Dataset loaded with {len(full_train_dataset)} samples.")

    # random index
    test_idx = random.randint(0, len(full_train_dataset) - 1)
    print(f"Testing `__getitem__` for index: {test_idx}")

    batch = full_train_dataset[test_idx]

    print(f"Patient ID: {batch['patient_id']}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Label shape: {batch['label'].shape}")
    print(f"Crop indexes: {batch['crop_indexes']}")
    print(f"ET present: {batch['et_present']}")


test_brats_getitem()