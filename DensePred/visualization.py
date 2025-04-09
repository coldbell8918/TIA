import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from models.get_model import get_model
import argparse

# =======================
# Argument parsing
# =======================
parser = argparse.ArgumentParser(description='Multi-task Prediction Visualization Saver')
parser.add_argument('--backbone', default='resnet50', type=str, help='shared backbone')
parser.add_argument('--method', default='vanilla', type=str, help='vanilla or mtan')
parser.add_argument('--tasks', default=['semantic', 'depth', 'normal'], nargs='+', help='Task(s) to be trained')
parser.add_argument('--head', default='deeplab_head', type=str, help='task-specific decoder')
parser.add_argument('--num_images', default=654, type=int, help='Number of images to visualize')
parser.add_argument('--output_dir', default='./visualization_results', type=str, help='Directory to save visualization images')
opt = parser.parse_args()

# =======================
# Model Setup
# =======================
tasks_outputs = {
    'semantic': 13,
    'depth': 1,
    'normal': 3,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_model(opt, tasks_outputs=tasks_outputs).to(device)
pretrained_model_path = '/workspace/UniversalRepresentations/DensePred/results/mtl-pjc4/resnet50_deeplab_head_url_vanilla_uniform_alr_0.01_model_best.pth.tar'
model.load_state_dict(torch.load(pretrained_model_path)['state_dict'])
model.eval()

# =======================
# Output directory
# =======================
os.makedirs(opt.output_dir, exist_ok=True)

# =======================
# Visualization Function
# =======================
def save_visualization(idx):
    # Load input and ground truth
    base_path = f'/workspace/UniversalRepresentations/DensePred/data/nyuv2/val'
    image_data = np.load(f'{base_path}/image/{idx}.npy').astype(np.float32)
    img_array_label = np.load(f'{base_path}/label/{idx}.npy')
    img_array_depth = np.load(f'{base_path}/depth/{idx}.npy')
    img_array_normal = np.load(f'{base_path}/normal/{idx}.npy')

    image_data = np.transpose(image_data, (2, 0, 1))
    image_tensor = torch.from_numpy(image_data).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)

    semantic_pred_tensor = outputs['semantic']
    depth_pred_tensor = outputs['depth']
    normal_pred_tensor = outputs['normal']

    semantic_pred = semantic_pred_tensor.argmax(1).squeeze().cpu().numpy()
    depth_pred = depth_pred_tensor.squeeze().cpu().numpy()
    normal_pred = normal_pred_tensor.squeeze().cpu().numpy()
    normal_pred = np.transpose(normal_pred, (1, 2, 0))

    # Normalize normal vector
    epsilon = 1e-8
    normal_viz = (normal_pred - normal_pred.min()) / (normal_pred.max() - normal_pred.min() + epsilon)

    # Plotting
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(np.transpose(image_data, (1, 2, 0)))
    axes[0, 0].set(title='Image', xticks=[], yticks=[])

    axes[0, 1].imshow(semantic_pred)
    axes[0, 1].set(title='Semantic Segmentation', xticks=[], yticks=[])

    axes[0, 2].imshow(depth_pred)
    axes[0, 2].set(title='Depth Prediction', xticks=[], yticks=[])

    axes[0, 3].imshow(normal_viz, vmin=0, vmax=1)
    axes[0, 3].set(title='Normal Estimation', xticks=[], yticks=[])

    axes[1, 0].imshow(np.transpose(image_data, (1, 2, 0)))
    axes[1, 0].set(title='Image', xticks=[], yticks=[])

    axes[1, 1].imshow(img_array_label)
    axes[1, 1].set(title='Label', xticks=[], yticks=[])

    axes[1, 2].imshow(img_array_depth)
    axes[1, 2].set(title='Depth', xticks=[], yticks=[])

    axes[1, 3].imshow(img_array_normal)
    axes[1, 3].set(title='Normal', xticks=[], yticks=[])

    # Save figure
    save_path = os.path.join(opt.output_dir, f'{idx}.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)

# =======================
# Batch Visualization Save
# =======================
if __name__ == '__main__':
    for i in range(opt.num_images):
        try:
            save_visualization(i)
            print(f"[✓] Saved: {i}.png")
        except Exception as e:
            print(f"[✗] Failed on index {i}: {e}")
