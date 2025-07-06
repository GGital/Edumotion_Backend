from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch

def InitializeObjectRecognitionModel(model_path='google/owlvit-base-patch32'):
    """
    Initialize the object recognition model.
    
    Args:
        model_path (str): Path to the pre-trained model.
        
    Returns:
        model: The initialized object recognition model (on CUDA if available).
        processor: The processor for the model.
    """
    processor = OwlViTProcessor.from_pretrained(model_path)
    model = OwlViTForObjectDetection.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    return model, processor

def recognize_objects_in_image(image_path, object_name, model, processor):
    """
    Recognize objects in an image using the OwlViT model.
    
    Args:
        image_path (str): Path to the input image.
        object_name (str): Name of the object to recognize.
        model: The initialized object recognition model.
        processor: The processor for the model.
        
    Returns:
        results: The recognition results containing bounding boxes, scores, and labels (on CPU).
    """
    image = Image.open(image_path)
    texts = [[f"a photo of a {object_name}"]]
    inputs = processor(text=texts, images=image, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    outputs = model(**inputs)

    # Target image sizes (height, width) to rescale box predictions [batch_size, 2]
    target_sizes = torch.Tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(outputs=outputs, threshold=0.1, target_sizes=target_sizes)

    # Move all result tensors to CPU to free GPU memory
    for res in results:
        for k, v in res.items():
            if hasattr(v, 'cpu'):
                res[k] = v.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return results

def display_recognition_results(results, object_name):
    """
    Display the recognition results.
    
    Args:
        results: The recognition results containing bounding boxes, scores, and labels.
        object_name (str): Name of the object to recognize.
        
    Returns:
        None
    """
    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    text = [f"a photo of a {object_name}"]
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Print detected objects and rescaled box coordinates
    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(f"Detected {text[label]} with confidence {round(score.item(), 3)} at location {box}")

def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Args:
        boxA (list or tensor): [x1, y1, x2, y2] for the first box.
        boxB (list or tensor): [x1, y1, x2, y2] for the second box.
    Returns:
        float: IoU value.
    """
    # Ensure boxes are in list format
    boxA = [float(x) for x in boxA]
    boxB = [float(x) for x in boxB]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def compare_boxes_iou(results1, results2, object_name):
    """
    Compare the best-matching bounding boxes for the same object in two images using IoU.
    For each box in image 1, find the box in image 2 with the highest IoU, and return the maximum IoU found.
    Args:
        results1: Recognition results for image 1.
        results2: Recognition results for image 2.
        object_name (str): Name of the object to compare.
    Returns:
        float: Maximum IoU value between any detected boxes for the object.
    """
    i = 0
    boxes1, labels1 = results1[i]["boxes"], results1[i]["labels"]
    boxes2, labels2 = results2[i]["boxes"], results2[i]["labels"]
    # Get all boxes for the object (label 0, since only one text prompt is used)
    idxs1 = (labels1 == 0).nonzero(as_tuple=True)[0]
    idxs2 = (labels2 == 0).nonzero(as_tuple=True)[0]
    if len(idxs1) == 0 or len(idxs2) == 0:
        return None  # Object not detected in one or both images
    max_iou = 0.0
    for idx1 in idxs1:
        box1 = boxes1[idx1].tolist()
        for idx2 in idxs2:
            box2 = boxes2[idx2].tolist()
            iou = calculate_iou(box1, box2)
            if iou > max_iou:
                max_iou = iou
    return max_iou

def compare_boxes_iou_all(results1, results2, object_name):
    """
    For each bounding box of the object in image 1, find the closest (highest IoU) box in image 2.
    Returns a list of IoU scores (one for each box in image 1, or empty if no matches).
    Args:
        results1: Recognition results for image 1.
        results2: Recognition results for image 2.
        object_name (str): Name of the object to compare.
    Returns:
        list: List of IoU values (one for each box in image 1, or empty if no matches).
    """
    i = 0
    boxes1, labels1 = results1[i]["boxes"], results1[i]["labels"]
    boxes2, labels2 = results2[i]["boxes"], results2[i]["labels"]
    idxs1 = (labels1 == 0).nonzero(as_tuple=True)[0]
    idxs2 = (labels2 == 0).nonzero(as_tuple=True)[0]
    if len(idxs1) == 0 or len(idxs2) == 0:
        return []  # Object not detected in one or both images
    iou_list = []
    for idx1 in idxs1:
        box1 = boxes1[idx1].tolist()
        best_iou = 0.0
        for idx2 in idxs2:
            box2 = boxes2[idx2].tolist()
            iou = calculate_iou(box1, box2)
            if iou > best_iou:
                best_iou = iou
        iou_list.append(best_iou)
    return iou_list

def is_iou_above_threshold(results1, results2, object_name, threshold=0.5):
    """
    Returns True if all IoU scores between each box in image 1 and its closest in image 2 are above the threshold,
    and the number of bounding boxes for the object matches in both images.
    Args:
        results1: Recognition results for image 1.
        results2: Recognition results for image 2.
        object_name (str): Name of the object to compare.
        threshold (float): IoU threshold.
    Returns:
        bool: True if all IoUs > threshold and number of boxes matches, False otherwise.
    """
    i = 0
    boxes1, labels1 = results1[i]["boxes"], results1[i]["labels"]
    boxes2, labels2 = results2[i]["boxes"], results2[i]["labels"]
    idxs1 = (labels1 == 0).nonzero(as_tuple=True)[0]
    idxs2 = (labels2 == 0).nonzero(as_tuple=True)[0]
    if len(idxs1) == 0 or len(idxs2) == 0:
        return False
    if len(idxs1) != len(idxs2):
        return False
    iou_list = compare_boxes_iou_all(results1, results2, object_name)
    if not iou_list:
        return False
    return all(iou > threshold for iou in iou_list)

def compare_images_iou(image_path1, image_path2, object_name, model, processor, threshold=0.5):
    """
    Compare two images for the same object using IoU threshold.
    Args:
        image_path1 (str): Path to the first image.
        image_path2 (str): Path to the second image.
        object_name (str): Name of the object to compare.
        model: The initialized object recognition model.
        processor: The processor for the model.
        threshold (float): IoU threshold.
    Returns:
        bool: True if all IoUs > threshold and number of boxes matches, False otherwise.
    """
    results1 = recognize_objects_in_image(image_path1, object_name, model, processor)
    results2 = recognize_objects_in_image(image_path2, object_name, model, processor)
    return is_iou_above_threshold(results1, results2, object_name, threshold)

