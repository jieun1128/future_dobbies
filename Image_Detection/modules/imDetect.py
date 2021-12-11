import io, sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

# Detectron 모듈 : 설치 방법 - https://detectron2.readthedocs.io/en/latest/tutorials/install.html
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import cv2
import numpy as np

# 프로젝트가 로드될 때 모델이 미리 메모리에 로드되게 함
cfg = get_cfg()
cfg.merge_from_file(os.path.dirname(__file__) + "/DetectronFiles/config.yaml")
cfg.MODEL.WEIGHTS = os.path.dirname(__file__) + "/DetectronFiles/model_final.pth"  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)


def cropImg(img: bytes) -> list[bytes]:
    """
    bytes 형태로 인코딩된 이미지를 받아서 이미지에 해당되는 bytes 이미지를 반환합니다.

    [입력 파라미터]
    img: 바이트 단위로 인코딩된 이미지 파일.

    [출력]
    result: 바이트 단위로 인코딩된 이미지 파일들.(이미지 부분만 잘려서 다시 인코딩됨) list형태로 반환됩니다.
    """

    # 이미지 디코딩 실시
    dec_img = np.frombuffer(img, dtype=np.uint8)
    dec_img = cv2.imdecode(dec_img, flags=1)

    # 모델 예측
    output = predictor(dec_img)

    # 예측된 결과로 이미지만 잘라서 저장
    crop_imgs = []
    for box in range(len(output['instances'])):
        if output['instances'][box].get_fields()['pred_classes'].item() == 4: # 이미지 Label에 해당하는 부분만 사용함
            x1, y1, x2, y2 = output['instances'][box].get_fields()['pred_boxes'].tensor[0].to('cpu') # 예측 결과 언패킹

            crop = dec_img[int(y1):int(y2), int(x1):int(x2), :]
            crop_imgs.append(crop)
            cv2.imshow(f"{box}", crop)


    # 이미지를 바이트로 다시 인코딩
    result = []
    for crop in crop_imgs:
        result.append(crop.tobytes())

    return result
    

if __name__ == '__main__':

    """
    테스트를 위해 따로 실행됐을 경우, 임의의 샘플 이미지를 불러와서 실행하도록 진행합니다.
    """
    print(os.path.dirname(__file__))
    file_name = './Notebook/sampleImages/NewsWithImagesSmall.png'

    # bytes 형태로 불러와 진 것을 가정합니다.
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
    
    result = cropImg(content)

    cv2.waitKey()
