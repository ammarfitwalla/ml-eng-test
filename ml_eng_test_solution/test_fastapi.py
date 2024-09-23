from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import io
import torch
from fastapi import File, UploadFile
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F
import base64
import numpy as np
from fastapi.responses import StreamingResponse, JSONResponse
from floortrans.models import get_model
import matplotlib.cm as cm
from floortrans.loaders import RotateNTurns
from floortrans.post_prosessing import split_prediction, get_polygons
from floortrans.plotting import polygons_to_image, discrete_cmap
from custom_wall_detection import *
import cv2, os


discrete_cmap()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the model (once on startup)
room_classes = ["Background", "Outdoor", "Wall", "Kitchen", "Living Room" ,"Bed Room", "Bath", "Entry", "Railing", "Storage", "Garage", "Undefined"]
icon_classes = ["No Icon", "Window", "Door", "Closet", "Electrical Appliance" ,"Toilet", "Sink", "Sauna Bench", "Fire Place", "Bathtub", "Chimney"]

rot = RotateNTurns()
model = get_model('hg_furukawa_original', 51)
n_classes = 44
split = [21, 12, 11]
model.conv4_ = torch.nn.Conv2d(256, n_classes, bias=True, kernel_size=1)
model.upsample = torch.nn.ConvTranspose2d(n_classes, n_classes, kernel_size=4, stride=4)
checkpoint = torch.load(f'CubiCasa{os.sep}model_best_val_loss_var.pkl', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state'])
model.eval()

# Preprocessing function
def preprocess_image(image: Image.Image):
    preprocess = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return preprocess(image).unsqueeze(0)

def apply_colormap(segmentation, cmap_name='viridis'):
    cmap = cm.get_cmap(cmap_name)
    rgba_img = cmap(segmentation / np.max(segmentation))  # Apply color map
    rgb_img = np.delete(rgba_img, 3, 2)  # Remove alpha channel
    return (rgb_img * 255).astype(np.uint8)

@app.post("/predict/room")
def predict(file: UploadFile = File(...)):
    # Load the image
    print("START")
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")

    # Preprocess the image
    image = preprocess_image(image)
    label = torch.zeros(1, 2, 512, 512)
    junctions = {}

    label_np = label.data.numpy()[0]

    # Get the prediction from the model
    print("PREDICTION")
    with torch.no_grad():
        height = label_np.shape[1]
        width = label_np.shape[2]
        img_size = (height, width)

        rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
        pred_count = len(rotations)
        prediction = torch.zeros([pred_count, n_classes, height, width])
        for i, r in enumerate(rotations):
            forward, back = r
            # We rotate first the image
            rot_image = rot(image, 'tensor', forward)
            pred = model(rot_image)
            # We rotate prediction back
            pred = rot(pred, 'tensor', back)
            # We fix heatmaps
            pred = rot(pred, 'points', back)
            # We make sure the size is correct
            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
            # We add the prediction to output
            prediction[i] = pred[0]

    prediction = torch.mean(prediction, 0, True)
    rooms_label = label_np[0]
    icons_label = label_np[1]
    print("PREDICT")

    # rooms_pred = F.softmax(prediction[0, 21:21 + 12], 0).cpu().data.numpy()
    # rooms_pred = np.argmax(rooms_pred, axis=0)

    # icons_pred = F.softmax(prediction[0, 21 + 12:], 0).cpu().data.numpy()
    # icons_pred = np.argmax(icons_pred, axis=0)

    heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
    polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2])

    # Return the images as responses
    pol_room_seg, pol_icon_seg = polygons_to_image(polygons, types, room_polygons, room_types, height, width)

    colored_room_seg = apply_colormap(pol_room_seg, cmap_name='rooms')
    # colored_icon_seg = apply_colormap(pol_icon_seg, cmap_name='icons')

    room_buf = io.BytesIO()
    room_image = Image.fromarray(colored_room_seg)
    room_image.save(room_buf, format="PNG")
    room_buf.seek(0)

    # Return the image as a StreamingResponse
    return StreamingResponse(room_buf, media_type="image/png")

    # room_buf = io.BytesIO()
    # room_image = Image.fromarray(colored_room_seg)
    # room_image.save(room_buf, format="PNG")
    # room_buf.seek(0)
    # room_image_base64 = base64.b64encode(room_buf.getvalue()).decode('utf-8')
    #
    # icon_buf = io.BytesIO()
    # icon_image = Image.fromarray(colored_icon_seg)
    # icon_image.save(icon_buf, format="PNG")
    # icon_buf.seek(0)
    # icon_image_base64 = base64.b64encode(icon_buf.getvalue()).decode('utf-8')

    # Return the images as base64 strings
    # return JSONResponse({
    #     "room_segmentation": room_image_base64,
    #     "icon_segmentation": icon_image_base64
    # })

@app.post("/predict/detect_wall")
def predict_wall_without_cubicasa(file: UploadFile = File(...)):
    # Load the image
    print("START")
    image = Image.open(io.BytesIO(file.file.read())).convert("L")
    # image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess the image
    print("Determine how many parts to split the image into based on its size")
    rows, cols = determine_split(image)

    print("Split the image into sub-images if necessary")
    if rows > 1 or cols > 1:
        sub_images, sub_height, sub_width = split_image(image, rows, cols)
        print("Process each sub-image individually")
        processed_sub_images = [process_sub_image(sub_image) for sub_image in sub_images]
        print("Stitch the processed sub-images back together")
        stitched_image = stitch_images(processed_sub_images, rows, cols, sub_height, sub_width)
    else:
        # Process the whole image if no splitting is necessary
        stitched_image = process_sub_image(image)


    room_buf = io.BytesIO()
    room_image = Image.fromarray(stitched_image)
    room_image.save(room_buf, format="PNG")
    room_buf.seek(0)

    # Return the image as a StreamingResponse
    return StreamingResponse(room_buf, media_type="image/png")


@app.post("/predict/icon")
def predict(file: UploadFile = File(...)):
    # Load the image
    print("START")
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")

    # Preprocess the image
    image = preprocess_image(image)
    label = torch.zeros(1, 2, 512, 512)
    junctions = {}

    label_np = label.data.numpy()[0]

    # Get the prediction from the model
    print("PREDICTION")
    with torch.no_grad():
        height = label_np.shape[1]
        width = label_np.shape[2]
        img_size = (height, width)

        rotations = [(0, 0), (1, -1), (2, 2), (-1, 1)]
        pred_count = len(rotations)
        prediction = torch.zeros([pred_count, n_classes, height, width])
        for i, r in enumerate(rotations):
            forward, back = r
            # We rotate first the image
            rot_image = rot(image, 'tensor', forward)
            pred = model(rot_image)
            # We rotate prediction back
            pred = rot(pred, 'tensor', back)
            # We fix heatmaps
            pred = rot(pred, 'points', back)
            # We make sure the size is correct
            pred = F.interpolate(pred, size=(height, width), mode='bilinear', align_corners=True)
            # We add the prediction to output
            prediction[i] = pred[0]

    prediction = torch.mean(prediction, 0, True)
    rooms_label = label_np[0]
    icons_label = label_np[1]
    print("PREDICT")

    # rooms_pred = F.softmax(prediction[0, 21:21 + 12], 0).cpu().data.numpy()
    # rooms_pred = np.argmax(rooms_pred, axis=0)

    # icons_pred = F.softmax(prediction[0, 21 + 12:], 0).cpu().data.numpy()
    # icons_pred = np.argmax(icons_pred, axis=0)

    heatmaps, rooms, icons = split_prediction(prediction, img_size, split)
    polygons, types, room_polygons, room_types = get_polygons((heatmaps, rooms, icons), 0.2, [1, 2])

    # Return the images as responses
    pol_room_seg, pol_icon_seg = polygons_to_image(polygons, types, room_polygons, room_types, height, width)

    # colored_room_seg = apply_colormap(pol_room_seg, cmap_name='rooms')
    colored_icon_seg = apply_colormap(pol_icon_seg, cmap_name='icons')

    icon_buf = io.BytesIO()
    icon_image = Image.fromarray(colored_icon_seg)
    icon_image.save(icon_buf, format="PNG")
    icon_buf.seek(0)

    # Return the image as a StreamingResponse
    return StreamingResponse(icon_buf, media_type="image/png")
