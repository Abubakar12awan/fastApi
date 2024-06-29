# # Commands to install required packages (execute these in the terminal):
# # pip install fastapi uvicorn opencv-python mediapipe rembg pillow numpy

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import logging
import uvicorn
import cv2
import mediapipe as mp
from rembg import remove
from PIL import Image
import numpy as np

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# # Initialize FastAPI application
# app = FastAPI()

# # Function to remove the background of an image
# def remove_bg(image_np):
#     # Convert the NumPy array to a PIL image
#     pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    
#     # Remove the background using rembg
#     output_pil_image = remove(pil_image)
    
#     # Convert the PIL image back to a NumPy array and convert color format to BGR
#     output_np = cv2.cvtColor(np.array(output_pil_image), cv2.COLOR_RGB2BGR)
#     return output_np

# # Endpoint to process images
# @app.post("/process-images")
# async def process_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
#     logging.info("Received request for /process-images")

#     # Read image data from UploadFile objects
#     image1_data = await image1.read()
#     image2_data = await image2.read()






#     import cv2
#     import numpy as np
#     import torch
#     from torchvision import models, transforms
#     from PIL import Image

#     # Function to apply segmentation
#     def segment_body(image_path, output_path):
#         # Load the pre-trained model with specified weights
#         model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
#         model.eval()

#         # Define the preprocessing transformations for the input image
#         preprocess = transforms.Compose([
#             transforms.Resize((520, 520)),  # Resize to the expected input size
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])

#         # Load and preprocess the image
#         image_pil = Image.open(image_path).convert('RGB')
#         input_tensor = preprocess(image_pil)
#         input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
#         print("3")

#         # Apply the model to the input tensor
#         with torch.no_grad():
#             output = model(input_tensor)
        
#         # Extract the predicted segmentation mask
#         mask = output['out'][0].argmax(0).numpy()

#         # Load the original image using OpenCV
#         original_image = cv2.imread(image_path)

#         # Resize the mask to the original image dimensions
#         mask_resized = cv2.resize(mask.astype(np.uint8), (original_image.shape[1], original_image.shape[0]))

#         # Create an RGBA image (four channels) with the same shape as the original image
#         segmented_image = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)
#         print("1")
#         count=0
#         peer=0

#         # Copy the t-shirt pixels from the original image and set alpha based on the mask
#         for y in range(original_image.shape[0]):
#             for x in range(original_image.shape[1]):
#                 # print("2")

#                 if mask_resized[y, x] == 15:  # If the pixel belongs to the t-shirt class
#                     segmented_image[y, x, :3] = original_image[y, x]  # Copy RGB channels
#                     segmented_image[y, x, 3] = 255  # Set alpha to 255 (fully opaque)
#                     if count==0:

#                         print("here")
#                         count+=1

#                 else:
#                     segmented_image[y, x, 3] = 0  # Set alpha to 0 (fully transparent)
#                     # print("ere")
#                     if peer==0:
#                         print("nnn")
#                         peer+=1
#         # Save the segmented image as a PNG file to preserve transparency
#         print("output")
#         print("segmented_image",segmented_image)
#         print("shape",segmented_image.shape)
#         success=cv2.imwrite(output_path, segmented_image)
#         print("Image saved:", success)
#         return segmented_image


#     # Example usage
#     # input_image_path = "/content/blank-white-t-shirt.jpg"
#     # output_image_path = "/content/output_image.png"

#     # segment_body(input_image_path, output_image_path)







#     # Convert image data to NumPy arrays using OpenCV's imdecode
#     image_np1 = cv2.imdecode(np.frombuffer(image1_data, np.uint8), cv2.IMREAD_COLOR)
#     image_np2 = cv2.imdecode(np.frombuffer(image2_data, np.uint8), cv2.IMREAD_COLOR)

#     pathi="nr.jpg"

#     cv2.imwrite(pathi,image_np2)

#     # Remove backgrounds from the images
#     image_rgb = await remove_bg(image_np1)
#     image_rgb_s = await segment_body(pathi)
#     cv2.imwrite("image1test.jpg", image_rgb)
#     cv2.imwrite("image2test.jpg", image_rgb_s)


#     # Process the images using MediaPipe Pose
#     mp_pose = mp.solutions.pose
#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         results = pose.process(image_rgb)

#         # Check if pose landmarks are detected
#         if results.pose_landmarks:
            
            
#             results.pose_landmarks.landmark[11].x=results.pose_landmarks.landmark[11].x+0.03
#             results.pose_landmarks.landmark[11].y=results.pose_landmarks.landmark[11].y-0.03
            
#             results.pose_landmarks.landmark[12].x=results.pose_landmarks.landmark[12].x-0.03
#             results.pose_landmarks.landmark[12].y=results.pose_landmarks.landmark[12].y-0.03
            
            
#             results.pose_landmarks.landmark[23].x=results.pose_landmarks.landmark[23].x+0.03
#             results.pose_landmarks.landmark[23].y=results.pose_landmarks.landmark[23].y+0.01
            
#             results.pose_landmarks.landmark[24].x=results.pose_landmarks.landmark[24].x-0.03
#             results.pose_landmarks.landmark[24].y=results.pose_landmarks.landmark[24].y+0.01
            
            
#             # Get landmarks for shoulders and hips
#             left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
#             right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#             left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
#             right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

#             # Convert landmarks to image coordinates
#             left_shoulder_x = int(left_shoulder.x * image_rgb.shape[1])
#             left_shoulder_y = int(left_shoulder.y * image_rgb.shape[0])
#             right_shoulder_x = int(right_shoulder.x * image_rgb.shape[1])
#             right_shoulder_y = int(right_shoulder.y * image_rgb.shape[0])
#             left_hip_x = int(left_hip.x * image_rgb.shape[1])
#             left_hip_y = int(left_hip.y * image_rgb.shape[0])
#             right_hip_x = int(right_hip.x * image_rgb.shape[1])
#             right_hip_y = int(right_hip.y * image_rgb.shape[0])

#             # Calculate the bounding box coordinates
#             x_min = max(0, min(left_shoulder_x, right_shoulder_x))
#             x_max = min(image_rgb.shape[1], max(left_shoulder_x, right_shoulder_x))
#             y_min = max(0, min(left_shoulder_y, right_shoulder_y))
#             y_max = min(image_rgb.shape[0], max(left_hip_y, right_hip_y))

#             # Crop the image to the region of interest (shoulders to hips)
#             cropped_region = image_rgb[y_min:y_max, x_min:x_max]
#             print("Cropped Image Shoulder to Hips", cropped_region.shape)

#             print(f"Cropped region shape: {cropped_region.shape}")

#             # Resize the second image to the size of the cropped region
#             resized_shirt_image = cv2.resize(image_rgb_s, (cropped_region.shape[1], cropped_region.shape[0]))
#             print(f"Resized second image shape: {resized_shirt_image.shape}")

#             # Replace the cropped region with the resized second image
#             image_rgb[y_min:y_max, x_min:x_max] = resized_shirt_image

#             # Save the modified image
#             cv2.imwrite("new_imag1.jpg", image_rgb)
#             print(f"Modified image saved ")

#         # Display the cropped image
# #         cv2.imshow('Cropped Image (Shoulder to Hips)', cropped_region)

#     # Wait for a key press to close the window
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()
#     # with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

#     #     logging.info("1")

#     #     # Process the first image
#     #     results1 = pose.process(cv2.cvtColor(image_np1, cv2.COLOR_BGR2RGB))
#     #     logging.info("2")
#     #     print("results1",results1.pose_landmarks)


#     #     # Check if pose landmarks are detected in the first image
#     #     if results1.pose_landmarks:
#     #         logging.info("4")

#     #         landmarks1 = results1.pose_landmarks.landmark

#     #         # Process the second image
#     #         results2 = pose.process(cv2.cvtColor(image_np2, cv2.COLOR_BGR2RGB))
#     #         print("results2",results2.pose_landmarks)

#     #         # Check if pose landmarks are detected in the second image
#     #         # if results2.pose_landmarks:
#     #         logging.info("5")
#     #         # landmarks2 = results2.pose_landmarks.landmark

#     #         # Get landmarks for shoulders and hips from the first image
#     #         left_shoulder1 = landmarks1[mp_pose.PoseLandmark.LEFT_SHOULDER]
#     #         right_shoulder1 = landmarks1[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#     #         left_hip1 = landmarks1[mp_pose.PoseLandmark.LEFT_HIP]
#     #         right_hip1 = landmarks1[mp_pose.PoseLandmark.RIGHT_HIP]

#     #         # Convert landmarks to image coordinates
#     #         def get_coords(landmark, img_shape):
#     #             x = int(landmark.x * img_shape[1])
#     #             y = int(landmark.y * img_shape[0])
#     #             return x, y

#     #         left_shoulder_x1, left_shoulder_y1 = get_coords(left_shoulder1, image_np1.shape)
#     #         right_shoulder_x1, right_shoulder_y1 = get_coords(right_shoulder1, image_np1.shape)
#     #         left_hip_x1, left_hip_y1 = get_coords(left_hip1, image_np1.shape)
#     #         right_hip_x1, right_hip_y1 = get_coords(right_hip1, image_np1.shape)

#     #         # Calculate the bounding box coordinates
#     #         x_min1 = max(0, min(left_shoulder_x1, right_shoulder_x1))
#     #         x_max1 = min(image_np1.shape[1], max(left_shoulder_x1, right_shoulder_x1))
#     #         y_min1 = max(0, min(left_shoulder_y1, right_shoulder_y1))
#     #         y_max1 = min(image_np1.shape[0], max(left_hip_y1, right_hip_y1))

#     #         # Crop the region from shoulders to hips from the first image
#     #         cropped_region1 = image_np1[y_min1:y_max1, x_min1:x_max1]
#     #         logging.info("3")

#     #         print("cropped_region1",cropped_region1)
#     #         cv2.imwrite("cropped_region1.jpg", cropped_region1)
            
#     #         # Resize the second image to the size of the cropped region from the first image
#     #         resized_image2 = cv2.resize(image_np2, (cropped_region1.shape[1], cropped_region1.shape[0]))

#     #         # Replace the cropped region of the first image with the resized second image
#     #         image_np1[y_min1:y_max1, x_min1:x_max1] = resized_image2

#     # Save the modified image
#     cv2.imwrite("modified_image.jpg", image_np1)
#     logging.info(f"Modified image saved as 'modified_image.jpg'")

#     # Response indicating success
#     return JSONResponse(content={"message": "Images processed successfully"})

# if __name__ == "__main__":
#     # Start the FastAPI application with Uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)  # Enable auto-reload




# Commands to install required packages (execute these in the terminal):
# pip install fastapi uvicorn opencv-python mediapipe rembg pillow numpy torch torchvision


# # Configure logging
# logging.basicConfig(level=logging.INFO)

# # Initialize FastAPI application
# app = FastAPI()

# # Function to remove the background of an image
# def remove_bg(image_np):
#     # Convert the NumPy array to a PIL image
#     pil_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    
#     # Remove the background using rembg
#     output_pil_image = remove(pil_image)
    
#     # Convert the PIL image back to a NumPy array and convert color format to BGR
#     output_np = cv2.cvtColor(np.array(output_pil_image), cv2.COLOR_RGB2BGR)
#     return output_np

# # Function to apply segmentation
# def segment_body(image_path):
#     # Load the pre-trained model with specified weights
#     model = models.segmentation.deeplabv3_resnet101(
#         weights=models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1
#     )
#     model.eval()

#     # Define the preprocessing transformations for the input image
#     preprocess = transforms.Compose([
#         transforms.Resize((520, 520)),  # Resize to the expected input size
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     # Load and preprocess the image
#     image_pil = Image.open(image_path).convert('RGB')
#     input_tensor = preprocess(image_pil)
#     input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

#     # Apply the model to the input tensor
#     with torch.no_grad():
#         output = model(input_tensor)
    
#     # Extract the predicted segmentation mask
#     mask = output['out'][0].argmax(0).numpy()

#     # Load the original image using OpenCV
#     original_image = cv2.imread(image_path)

#     # Resize the mask to the original image dimensions
#     mask_resized = cv2.resize(mask.astype(np.uint8), (original_image.shape[1], original_image.shape[0]))

#     # Create an RGBA image (four channels) with the same shape as the original image
#     segmented_image = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)

#     # Copy the t-shirt pixels from the original image and set alpha based on the mask
#     for y in range(original_image.shape[0]):
#         for x in range(original_image.shape[1]):
#             if mask_resized[y, x] == 15:  # If the pixel belongs to the t-shirt class
#                 segmented_image[y, x, :3] = original_image[y, x]  # Copy RGB channels
#                 segmented_image[y, x, 3] = 255  # Set alpha to 255 (fully opaque)
#             else:
#                 segmented_image[y, x, 3] = 0  # Set alpha to 0 (fully transparent)

#     # Return the segmented image
#     return segmented_image

# # Endpoint to process images
# @app.post("/process-images")
# async def process_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
#     logging.info("Received request for /process-images")

#     # Read image data from UploadFile objects
#     image1_data = await image1.read()
#     image2_data = await image2.read()

#     # Convert image data to NumPy arrays using OpenCV's imdecode
#     image_np1 = cv2.imdecode(np.frombuffer(image1_data, np.uint8), cv2.IMREAD_COLOR)
#     image_np2 = cv2.imdecode(np.frombuffer(image2_data, np.uint8), cv2.IMREAD_COLOR)

#     # Remove backgrounds from the images
#     image_rgb = remove_bg(image_np1)
#     image_rgb_s = segment_body("image2.jpg")

#     # Save images for debugging
#     cv2.imwrite("image1test.jpg", image_rgb)
#     cv2.imwrite("image2test.jpg", image_rgb_s)

#     # Process the images using MediaPipe Pose
#     mp_pose = mp.solutions.pose
#     with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#         results = pose.process(image_rgb)

#         # Check if pose landmarks are detected
#         if results.pose_landmarks:
#             # Adjust pose landmarks
#             results.pose_landmarks.landmark[11].x += 0.03
#             results.pose_landmarks.landmark[11].y -= 0.03
#             results.pose_landmarks.landmark[12].x -= 0.03
#             results.pose_landmarks.landmark[12].y -= 0.03
#             results.pose_landmarks.landmark[23].x += 0.03
#             results.pose_landmarks.landmark[23].y += 0.01
#             results.pose_landmarks.landmark[24].x -= 0.03
#             results.pose_landmarks.landmark[24].y += 0.01

#             # Get landmarks for shoulders and hips
#             left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
#             right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#             left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
#             right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

#             # Convert landmarks to image coordinates
#             left_shoulder_x = int(left_shoulder.x * image_rgb.shape[1])
#             left_shoulder_y = int(left_shoulder.y * image_rgb.shape[0])
#             right_shoulder_x = int(right_shoulder.x * image_rgb.shape[1])
#             right_shoulder_y = int(right_shoulder.y * image_rgb.shape[0])
#             left_hip_x = int(left_hip.x * image_rgb.shape[1])
#             left_hip_y = int(left_hip.y * image_rgb.shape[0])
#             right_hip_x = int(right_hip.x * image_rgb.shape[1])
#             right_hip_y = int(right_hip.y * image_rgb.shape[0])

#             # Calculate the bounding box coordinates
#             x_min = max(0, min(left_shoulder_x, right_shoulder_x))
#             x_max = min(image_rgb.shape[1], max(left_shoulder_x, right_shoulder_x))
#             y_min = max(0, min(left_shoulder_y, right_shoulder_y))
#             y_max = min(image_rgb.shape[0], max(left_hip_y, right_hip_y))

#             # Crop the image to the region of interest (shoulders to hips)
#             cropped_region = image_rgb[y_min:y_max, x_min:x_max]

#             # Resize the second image to the size of the cropped region
#             resized_shirt_image = cv2.resize(image_rgb_s, (cropped_region.shape[1], cropped_region.shape[0]))

#             image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2RGBA)


#             # Replace the cropped region with the resized second image
#             image_rgb[y_min:y_max, x_min:x_max] = resized_shirt_image

#             # Save the modified image
#             cv2.imwrite("new_imag1.jpg", image_rgb)
#             print(f"Modified image saved.")

#         # Display the cropped image (uncomment if needed)
#         # cv2.imshow('Cropped Image (Shoulder to Hips)', cropped_region)
#         # cv2.waitKey(0)
#         # cv2.destroyAllWindows()

#     # Save the modified image
#     cv2.imwrite("modified_image.jpg", image_np1)
#     logging.info(f"Modified image saved as 'modified_image.jpg'")

#     # Response indicating success
#     return JSONResponse(content={"message": "Images processed successfully"})



######       Code for save images
# from fastapi import FastAPI, File, UploadFile
# import logging
# import cv2
# import numpy as np
# import uvicorn

# app = FastAPI()

# @app.post("/process-images")
# async def process_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
#     logging.info("Received request for /process-images")

#     # Read image data from UploadFile objects
#     image1_data = await image1.read()
#     image2_data = await image2.read()

#     with open("image1.jpg", "wb") as f:
#         f.write(image1_data)
#     with open("image2.jpg", "wb") as f:
#         f.write(image2_data)


#     print("Done")

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)



# from fastapi import FastAPI, File, UploadFile
# import logging
# import cv2
# import numpy as np
# import uvicorn

# app = FastAPI()

# @app.post("/process-images")
# async def process_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
#     print("hit")
#     print("1",image1)
#     print("2",image2)
#     logging.info("Received request for /process-images")

#     # Read image data from UploadFile objects
#     image1_data = await image1.read()
#     image2_data = await image2.read()

#     with open("image1.jpg", "wb") as f:
#         f.write(image1_data)
#     with open("image2.jpg", "wb") as f:
#         f.write(image2_data)


#     mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose   


# def remove_bg(img_path):
#     # Open the image file
# #     with open(img_path, 'rb') as f:
# #         input_image = f.read()
#     pil_image = Image.fromarray(img_path)


#     # Remove the background using rembg
#     output_image = remove(pil_image)
#     print("1",output_image)
#     print("2",type(output_image))
    
#     output_image=np.array(output_image)
#     output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

#     print("2",type(output_image))
#     print(output_image.shape)

#     # Convert the output to a PIL image
# #     result_image = Image.open(io.BytesIO(output_image))

#     # Display the image (if using a Jupyter Notebook)
# #     result_image.show()

#     # Save the resulting image if desired
# #     output_image_path = '1_removed_background.png'
# #     result_image.save(output_image_path)

#     # Return the PIL image (result_image) for further usage
#     return output_image



# def segment_body(image_path):
#     # Load the pre-trained model with specified weights
#     model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
#     model.eval()

#     # Define the preprocessing transformations for the input image
#     preprocess = transforms.Compose([
#         transforms.Resize((520, 520)),  # Resize to the expected input size
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

#     # Load and preprocess the image
#     image_pil = Image.open(image_path).convert('RGB')
#     input_tensor = preprocess(image_pil)
#     input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
#     print("3")

#     # Apply the model to the input tensor
#     with torch.no_grad():
#         output = model(input_tensor)
    
#     # Extract the predicted segmentation mask
#     mask = output['out'][0].argmax(0).numpy()

#     # Load the original image using OpenCV
#     original_image = cv2.imread(image_path)

#     # Resize the mask to the original image dimensions
#     mask_resized = cv2.resize(mask.astype(np.uint8), (original_image.shape[1], original_image.shape[0]))

#     # Create an RGBA image (four channels) with the same shape as the original image
#     segmented_image = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)
#     print("1")
#     count=0
#     peer=0

#     # Copy the t-shirt pixels from the original image and set alpha based on the mask
#     for y in range(original_image.shape[0]):
#         for x in range(original_image.shape[1]):
#             # print("2")

#             if mask_resized[y, x] == 15:  # If the pixel belongs to the t-shirt class
#                 segmented_image[y, x, :3] = original_image[y, x]  # Copy RGB channels
#                 segmented_image[y, x, 3] = 100  # Set alpha to 255 (fully opaque)
#                 if count==0:

#                   print("here")
#                   count+=1

#             else:
#                 segmented_image[y, x, 3] = 0  # Set alpha to 0 (fully transparent)
#                 # print("ere")
#                 if peer==0:
#                   print("nnn")
#                   peer+=1
#     # Save the segmented image as a PNG file to preserve transparency
#     print("output")
#     print("segmented_image",segmented_image)
#     print("shape",segmented_image.shape)
#     return segmented_image




# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils

# # Read the image from the file
# image = cv2.imread("214.jpg")
# # image_s = cv2.imread("blank-white-t-shirt.jpg")



# # Convert the image to RGB (MediaPipe expects RGB format)
# image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# print("image_shape",image_rgb.shape)
# # image_rgb_s = cv2.cvtColor(image_s, cv2.COLOR_BGR2RGB)
# # print("image_shape1",image_rgb_s.shape)


# image_rgb=remove_bg(image_rgb)

# cv2.imwrite("main_image withoutbackgroung.jpg",image_rgb)



# image_rgb_s=segment_body("212.webp")
# print("4",image_rgb_s.shape)

# # image_rgb_s = cv2.cvtColor(image_rgb_s, cv2.COLOR_BGR2RGB)



# cv2.imwrite("segmented_mag.png",image_rgb_s)

# image_rgb_s_3 = cv2.cvtColor(image_rgb_s, cv2.COLOR_BGRA2BGR)

# print("3",image_rgb_s_3.shape)
# # Process the image using MediaPipe Pose
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     results = pose.process(image_rgb)

#     # Check if pose landmarks are detected
#     if results.pose_landmarks:
        
        
#         results.pose_landmarks.landmark[11].x=results.pose_landmarks.landmark[11].x+0.15
#         results.pose_landmarks.landmark[11].y=results.pose_landmarks.landmark[11].y-0.07
        
#         results.pose_landmarks.landmark[12].x=results.pose_landmarks.landmark[12].x-0.14
#         results.pose_landmarks.landmark[12].y=results.pose_landmarks.landmark[12].y-0.07
        
        
#         results.pose_landmarks.landmark[23].x=results.pose_landmarks.landmark[23].x+0.14
#         results.pose_landmarks.landmark[23].y=results.pose_landmarks.landmark[23].y+0.04
        
#         results.pose_landmarks.landmark[24].x=results.pose_landmarks.landmark[24].x-0.16
#         results.pose_landmarks.landmark[24].y=results.pose_landmarks.landmark[24].y+0.04
        
        
#         # Get landmarks for shoulders and hips
#         left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
#         right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
#         left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
#         right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

#         # Convert landmarks to image coordinates
#         left_shoulder_x = int(left_shoulder.x * image_rgb.shape[1])
#         left_shoulder_y = int(left_shoulder.y * image_rgb.shape[0])
#         right_shoulder_x = int(right_shoulder.x * image_rgb.shape[1])
#         right_shoulder_y = int(right_shoulder.y * image_rgb.shape[0])
#         left_hip_x = int(left_hip.x * image_rgb.shape[1])
#         left_hip_y = int(left_hip.y * image_rgb.shape[0])
#         right_hip_x = int(right_hip.x * image_rgb.shape[1])
#         right_hip_y = int(right_hip.y * image_rgb.shape[0])

#         # Calculate the bounding box coordinates
#         x_min = max(0, min(left_shoulder_x, right_shoulder_x))
#         x_max = min(image_rgb.shape[1], max(left_shoulder_x, right_shoulder_x))
#         y_min = max(0, min(left_shoulder_y, right_shoulder_y))
#         y_max = min(image_rgb.shape[0], max(left_hip_y, right_hip_y))
        
#         cv2.imwrite("image_rgg.jpg",image_rgb[y_min:y_max, x_min:x_max])
        
        
        
        
        
        
        
        
        
        
        
#         alpha_channel = np.ones((image_rgb[y_min:y_max, x_min:x_max].shape[0], image_rgb[y_min:y_max, x_min:x_max].shape[1], 1), dtype=np.uint8) * 255


# # # Concatenate the RGB array with the alpha channel to form an RGBA array
#         image_rgbb = np.concatenate((image_rgb[y_min:y_max, x_min:x_max], alpha_channel), axis=-1)
    
#         print("after_rgba_shape",image_rgbb.shape)
        

        # resized_shirt_image = cv2.resize(image_rgb_s_3, (image_rgb[y_min:y_max, x_min:x_max].shape[1], image_rgb[y_min:y_max, x_min:x_max].shape[0]))

        # image_rgb[y_min:y_max, x_min:x_max]=resized_shirt_image

        # cv2.imwrite("full.png", image_rgb[y_min:y_max, x_min:x_max])


        # image_rgbb=image_rgb[y_min:y_max, x_min:x_max]


        # blended_image = cv2.addWeighted(image_rgbb, 0.1, resized_shirt_image, 0.9, 0)

        
        # print("blended_image",blended_image.shape)

        # cv2.imwrite("blended_image.png",blended_image)

        # image_rgb[y_min:y_max, x_min:x_max] = blended_image
        # cv2.imwrite("fullblended.png", image_rgb)





        







        
     


       

        
        
        
        
        
#         gray_overlay = cv2.cvtColor(resized_shirt_image, cv2.COLOR_BGR2GRAY)
#         print("resized_gray",gray_overlay.shape)

# #         # Create a binary mask of the overlay
#         _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)

# #         # Create an inverse mask
#         inverse_mask = cv2.bitwise_not(mask)

# #         # Extract the region from the base image where the overlay will be placed
#         base_bg = cv2.bitwise_and(image_rgbb, image_rgbb, mask=inverse_mask)

# #         # Extract the region from the overlay image
#         overlay_fg = cv2.bitwise_and(resized_shirt_image, resized_shirt_image, mask=mask)

# #         # Combine the base image background and the overlay foreground
#         result_img = cv2.add(base_bg, overlay_fg)

#         print("result_img",result_img.shape)

#         cv2.imwrite("result_img.png",result_img)


#         # rgb_image_final = cv2.cvtColor(result_img, cv2.COLOR_BGRA2BGR)
#         image_rgb[y_min:y_max, x_min:x_max] = result_img
#         cv2.imwrite("full.png", image_rgb)
#         print("Done")




#     print("Done")

# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)










from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import logging
import uvicorn
import cv2
import mediapipe as mp
from rembg import remove
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms
import io


import torch
import base64
from typing import Union
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()



origins = [
    "http://localhost:3000",  # Add the origin of your frontend application here
    # Add other allowed origins if necessary
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)






@app.post("/process-images")
async def process_images(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    print("image1",image1)
    print("hit")
    logging.info("Received request for /process-images")

    # Read image data from UploadFile objects
    image1_data = await image1.read()
    image2_data = await image2.read()

    with open("image1.jpg", "wb") as f:
        f.write(image1_data)
    with open("image2.jpg", "wb") as f:
        f.write(image2_data)

    # mp_drawing = mp.solutions.drawing_utils
    # mp_pose = mp.solutions.pose   

    def remove_bg(img_path):
        pil_image = Image.fromarray(img_path)

        # Remove the background using rembg
        output_image = remove(pil_image)
        print("1", output_image)
        print("2", type(output_image))

        output_image = np.array(output_image)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

        print("2", type(output_image))
        print(output_image.shape)

        # Return the output image
        return output_image

    def segment_body(image_path):
        # Load the pre-trained model with specified weights
        model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1)
        model.eval()

        # Define the preprocessing transformations for the input image
        preprocess = transforms.Compose([
            transforms.Resize((520, 520)),  # Resize to the expected input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Load and preprocess the image
        image_pil = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image_pil)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension
        print("3")

        # Apply the model to the input tensor
        with torch.no_grad():
            output = model(input_tensor)

        # Extract the predicted segmentation mask
        mask = output['out'][0].argmax(0).numpy()

        # Load the original image using OpenCV
        original_image = cv2.imread(image_path)

        # Resize the mask to the original image dimensions
        mask_resized = cv2.resize(mask.astype(np.uint8), (original_image.shape[1], original_image.shape[0]))

        # Create an RGBA image (four channels) with the same shape as the original image
        segmented_image = np.zeros((original_image.shape[0], original_image.shape[1], 4), dtype=np.uint8)
        print("1")

        # Copy the t-shirt pixels from the original image and set alpha based on the mask
        for y in range(original_image.shape[0]):
            for x in range(original_image.shape[1]):
                if mask_resized[y, x] == 15:  # If the pixel belongs to the t-shirt class
                    segmented_image[y, x, :3] = original_image[y, x]  # Copy RGB channels
                    segmented_image[y, x, 3] = 255  # Set alpha to 255 (fully opaque)
                else:
                    segmented_image[y, x, 3] = 0  # Set alpha to 0 (fully transparent)

        # Return the segmented image
        return segmented_image

    # Further processing or function calls can go here

    # Cleanup any created files, etc.

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

# Read the image from the file
    image = cv2.imread("image1.jpg")
    # image_s = cv2.imread("blank-white-t-shirt.jpg")



    # Convert the image to RGB (MediaPipe expects RGB format)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print("image_shape",image_rgb.shape)
    # image_rgb_s = cv2.cvtColor(image_s, cv2.COLOR_BGR2RGB)
    # print("image_shape1",image_rgb_s.shape)


    image_rgb=remove_bg(image_rgb)

    cv2.imwrite("main_image withoutbackgroung.jpg",image_rgb)



    image_rgb_s=segment_body("image2.jpg")
    print("4",image_rgb_s.shape)

    # image_rgb_s = cv2.cvtColor(image_rgb_s, cv2.COLOR_BGR2RGB)



    cv2.imwrite("segmented_mag.png",image_rgb_s)

    image_rgb_s_3 = cv2.cvtColor(image_rgb_s, cv2.COLOR_BGRA2BGR)

    print("3",image_rgb_s_3.shape)
    # Process the image using MediaPipe Pose
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        results = pose.process(image_rgb)

        # Check if pose landmarks are detected
        if results.pose_landmarks:
            
            
            # results.pose_landmarks.landmark[11].x=results.pose_landmarks.landmark[11].x+0.16
            # results.pose_landmarks.landmark[11].y=results.pose_landmarks.landmark[11].y-0.1
            
            # results.pose_landmarks.landmark[12].x=results.pose_landmarks.landmark[12].x-0.14
            # results.pose_landmarks.landmark[12].y=results.pose_landmarks.landmark[12].y-0.1
            

            
            # results.pose_landmarks.landmark[23].x=results.pose_landmarks.landmark[23].x+0.16
            # results.pose_landmarks.landmark[23].y=results.pose_landmarks.landmark[23].y+0.04
            
            # results.pose_landmarks.landmark[24].x=results.pose_landmarks.landmark[24].x-0.16
            # results.pose_landmarks.landmark[24].y=results.pose_landmarks.landmark[24].y+0.04




            results.pose_landmarks.landmark[11].x=results.pose_landmarks.landmark[11].x+0.16
            results.pose_landmarks.landmark[11].y=results.pose_landmarks.landmark[11].y-0.09
            
            results.pose_landmarks.landmark[12].x=results.pose_landmarks.landmark[12].x-0.15
            results.pose_landmarks.landmark[12].y=results.pose_landmarks.landmark[12].y-0.09
            
            
            results.pose_landmarks.landmark[23].x=results.pose_landmarks.landmark[23].x+0.16
            results.pose_landmarks.landmark[23].y=results.pose_landmarks.landmark[23].y+0.04
            
            results.pose_landmarks.landmark[24].x=results.pose_landmarks.landmark[24].x-0.16

            results.pose_landmarks.landmark[24].y=results.pose_landmarks.landmark[24].y+0.04

            
            
            # Get landmarks for shoulders and hips
            left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]

            # Convert landmarks to image coordinates
            left_shoulder_x = int(left_shoulder.x * image_rgb.shape[1])
            left_shoulder_y = int(left_shoulder.y * image_rgb.shape[0])
            right_shoulder_x = int(right_shoulder.x * image_rgb.shape[1])
            right_shoulder_y = int(right_shoulder.y * image_rgb.shape[0])
            left_hip_x = int(left_hip.x * image_rgb.shape[1])
            left_hip_y = int(left_hip.y * image_rgb.shape[0])
            right_hip_x = int(right_hip.x * image_rgb.shape[1])
            right_hip_y = int(right_hip.y * image_rgb.shape[0])





            # Calculate the bounding box coordinates
            x_min = max(0, min(left_shoulder_x, right_shoulder_x))
            x_max = min(image_rgb.shape[1], max(left_shoulder_x, right_shoulder_x))
            y_min = max(0, min(left_shoulder_y, right_shoulder_y))
            y_max = min(image_rgb.shape[0], max(left_hip_y, right_hip_y))
            
            cv2.imwrite("image_rgg.jpg",image_rgb[y_min:y_max, x_min:x_max])
            
        
        
        
        
        
        
        
        
        
        
#         alpha_channel = np.ones((image_rgb[y_min:y_max, x_min:x_max].shape[0], image_rgb[y_min:y_max, x_min:x_max].shape[1], 1), dtype=np.uint8) * 255


# # # Concatenate the RGB array with the alpha channel to form an RGBA array
#         image_rgbb = np.concatenate((image_rgb[y_min:y_max, x_min:x_max], alpha_channel), axis=-1)
    
#         print("after_rgba_shape",image_rgbb.shape)
        

            resized_shirt_image = cv2.resize(image_rgb_s_3, (image_rgb[y_min:y_max, x_min:x_max].shape[1], image_rgb[y_min:y_max, x_min:x_max].shape[0]))

        # image_rgb[y_min:y_max, x_min:x_max]=resized_shirt_image

        # cv2.imwrite("full.png", image_rgb[y_min:y_max, x_min:x_max])


            image_rgbb=image_rgb[y_min:y_max, x_min:x_max]


            # blended_image = cv2.addWeighted(image_rgbb, 0.1, resized_shirt_image, 0.9, 0)

            
            # print("blended_image",blended_image.shape)

            # cv2.imwrite("blended_image.png",blended_image)

            # image_rgb[y_min:y_max, x_min:x_max] = blended_image
            # cv2.imwrite("fullblended.png", image_rgb)


            gray_overlay = cv2.cvtColor(resized_shirt_image, cv2.COLOR_BGR2GRAY)
            print("resized_gray",gray_overlay.shape)

    #         # Create a binary mask of the overlay
            _, mask = cv2.threshold(gray_overlay, 1, 255, cv2.THRESH_BINARY)

    #         # Create an inverse mask
            inverse_mask = cv2.bitwise_not(mask)

    #         # Extract the region from the base image where the overlay will be placed
            base_bg = cv2.bitwise_and(image_rgbb, image_rgbb, mask=inverse_mask)

    #         # Extract the region from the overlay image
            overlay_fg = cv2.bitwise_and(resized_shirt_image, resized_shirt_image, mask=mask)

    #         # Combine the base image background and the overlay foreground
            result_img = cv2.add(base_bg, overlay_fg)

            print("result_img",result_img.shape)

            cv2.imwrite("result_img.png",result_img)


            # rgb_image_final = cv2.cvtColor(result_img, cv2.COLOR_BGRA2BGR)
            image_rgb[y_min:y_max, x_min:x_max] = result_img
            _, buffer = cv2.imencode('.jpg', image_rgb)
            base64_image = base64.b64encode(buffer).decode('utf-8')
            cv2.imwrite("full.png", image_rgb)
            print("Done")




    print("Done")

    # base64_image = f"data:image/jpeg;base64,{base64_image}"









    return {
        "message": "Images processed successfully.",
        "image_rgb": base64_image
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)













