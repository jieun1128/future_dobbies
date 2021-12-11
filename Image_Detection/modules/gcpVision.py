import os, io

from google.cloud import vision # GCP 비전 라이브러리

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "api_key.json" # API 키가 저장된 파일 (.gitignore에 등록됨. 확인 필수)

def get_vision_ocr(img: bytes) -> 'list[str]':
    """
    Google Cloud Vision API를 사용해서 이미지에서 글자를 인식합니다.
    
    [입력 파라미터]
    img: 바이트 단위로 인코딩된 이미지 파일.

    [출력]
    result: 인식된 글자가 문;장 단위로 정렬되어 list형태로 반환됩니다.
    """

    # vision으로 detect
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)
    texts = response.text_annotations

    #1. texts[0]에 전체 텍스트 저장. 1번 인덱스부터는 부분 텍스트.
    #2. base64로 인코딩되어 있으므로, 이를 해결하기 위해 string 형태로 디코딩함.
    encoded_texts = f"{texts[0].description}"
    
    result = [t for t in encoded_texts.split('\n')] # 개행문자 단위로 스플릿해서 반환

    return result

def get_image_search(imgs: list[bytes]):
    """
    Google Cloud Vision API를 사용해서 유사한 이미지가 있는 사이트의 URL을 확인합니다.
    
    [입력 파라미터]
    img: 바이트 단위로 인코딩된 이미지 파일의 list. 각 원소는 이미지 하나씩을 포함합니다.

    [출력]
    result: 검색된 URL주소들의 리스트. 각 원소는 동일 인덱스의 이미지와 같은 배열을 입력받습니다.
    """

    for img in imgs:
        print(img)
        
if __name__ == '__main__':

    """
    테스트를 위해 따로 실행됐을 경우, 임의의 샘플 이미지를 불러와서 실행하도록 진행합니다.
    """

    file_name = './Notebook/sampleImages/NewsWithImages.png'

    # bytes 형태로 불러와 진 것을 가정합니다.
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
    
    result = get_vision_ocr(content)

    print(result)