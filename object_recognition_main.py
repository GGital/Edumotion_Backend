from model_utils.object_recognition import InitializeObjectRecognitionModel, recognize_objects_in_image, display_recognition_results,is_iou_above_threshold,compare_images_iou

model, processor = InitializeObjectRecognitionModel()

results = recognize_objects_in_image('Test_medias/000000039769.jpg', 'cat', model, processor)

display_recognition_results(results, 'cat')

compare_images_iou('Test_medias/000000039769.jpg' , 'Test_medias/000000039769.jpg'  , 'cat' , model , processor)