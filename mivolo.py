import cv2
import torch
from mivolo.predictor import Predictor

class Args:
    def __init__(self,
                 detector_weights:str="/workspace/weights/yolov8x_person_face.pt",
                 checkpoint:str="/workspace/weights/model_imdb_cross_person_4.22_99.46.pth.tar",
                 device:int=0):
        self.detector_weights = detector_weights
        self.checkpoint = checkpoint
        self.device = f"cuda:{device}"
        self.with_persons = True
        self.disable_faces = False
        self.draw = True


class MiVOLO:
    def __init__(self):
        self.args = Args()
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        self.predictor = Predictor(self.args, verbose=True)

    def inference(self, image_path):
        image = cv2.imread(image_path)
        detected_objects, out_image = self.predictor.recognize(image)

        person_results = []
        for face_index, person_index in detected_objects.face_to_person_map.items():
            age = detected_objects.ages[face_index]
            gender = detected_objects.genders[face_index]
            gender_score = detected_objects.gender_scores[face_index]
            person_result ={"face": {}, "person": {}, "age": age, "gender": {"class": gender, "score": gender_score}}
            x1, y1, x2, y2 = detected_objects.get_bbox_by_ind(face_index).cpu().numpy()
            person_result["face"]["x"] = x1
            person_result["face"]["y"] = y1
            person_result["face"]["w"] = x2 - x1
            person_result["face"]["h"] = y2 - y1
            x1, y1, x2, y2 = detected_objects.get_bbox_by_ind(person_index).cpu().numpy()
            person_result["person"]["x"] = x1
            person_result["person"]["y"] = y1
            person_result["person"]["w"] = x2 - x1
            person_result["person"]["h"] = y2 - y1
            person_results.append(person_result)
        return person_results, out_image