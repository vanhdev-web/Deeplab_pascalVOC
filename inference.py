import cv2
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import torch
import numpy as np
from torchvision.transforms import Normalize
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path","-i", type=str, default="demo/deeplab1.png")
    parser.add_argument("--checkpoint","-c", type=str, default="trained_models/best.pt")
    args = parser.parse_args()
    return args

def inference(args):
    device = "cuda" if torch.cuda.is_available else "cpu"
    filename = args.image_path
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.transpose(image, (2, 0, 1)) / 255.
    image = torch.from_numpy(image).float()
    image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
    image = image[None, :, :, :]
    image = image.to(device)


    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu"
    )

    model = deeplabv3_mobilenet_v3_large(weights =None , aux_loss=True)
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)['out'][0]
    output_predictions = output.argmax(0)


    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    mask = output_predictions.cpu().numpy()  # shape HxW
    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for class_id in np.unique(mask):
        mask_color[mask == class_id] = colors[class_id]

    # nếu muốn overlay lên ảnh gốc
    image_rgb = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_rgb, 0.5, mask_color, 0.5, 0)

    # cv2.imshow("Segmentation Mask", mask_color)
    # cv2.imshow("Overlay", overlay)
    path = args.image_path
    path_prediction = path.replace(".","_prediction.")
    path_overlay = path.replace(".", "_overlay.")
    cv2.imwrite(path_prediction,mask_color)
    cv2.imwrite(path_overlay, overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = get_args()
    inference(args)



