import os, io, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from google.cloud import vision # GCP 비전 라이브러리

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.dirname(__file__) + "/api_key.json" # API 키가 저장된 파일 (.gitignore에 등록됨. 확인 필수)

# GCP비전과 연결하는 클라이언트
client = vision.ImageAnnotatorClient()

def get_vision_ocr(img: bytes) -> 'list[str]':
    """
    Google Cloud Vision API를 사용해서 이미지에서 글자를 인식합니다.
    
    [입력 파라미터]
    img: 바이트 단위로 인코딩된 이미지 파일.

    [출력]
    result: 인식된 글자가 문장 단위로 정렬되어 list형태로 반환됩니다.
    """

    # vision으로 detect
    image = vision.Image(content=img)
    response = client.document_text_detection(image=image)
    texts = response.text_annotations

    #1. texts[0]에 전체 텍스트 저장. 1번 인덱스부터는 부분 텍스트.
    #2. base64로 인코딩되어 있으므로, 이를 해결하기 위해 string 형태로 디코딩함.
    encoded_texts = f"{texts[0].description}"
    
    result = [t for t in encoded_texts.split('\n')] # 개행문자 단위로 스플릿해서 반환

    return result

def get_image_search(imgs: 'list[bytes]')-> 'list[list[str]]':
    """
    Google Cloud Vision API를 사용해서 유사한 이미지가 있는 사이트의 URL을 확인합니다.
    
    [입력 파라미터]
    img: 바이트 단위로 인코딩된 이미지 파일의 list. 각 원소는 이미지 하나씩을 포함합니다.

    [출력]
    result: 검색된 URL주소들의 리스트.
    각 원소는 동일 인덱스의 이미지와 같은 배열을 입력받습니다.
    각 배열의 원소는 최대 길이가 3인 list로 구성됩니다.
    해당하는 이미지가 없는 경우 해당 인덱스는 빈 배열이 입력됩니다.
    """

    result = []
    for img in imgs:
        urls = []

        # vision으로 detect
        image = vision.Image(content=img)

        response = client.web_detection(image=image)
        annotations = response.web_detection
        
        if annotations.pages_with_matching_images:
            pages = min(len(annotations.pages_with_matching_images), 3) # 최대 3개까지 반복, 없으면 있는만큼만 반복
            for p in range(0,pages):
                page = annotations.pages_with_matching_images[p]
                urls.append(page.url)
        else:
            urls.append([])
            print(f"None of Image Detected")

        result.append(urls)

    return result

if __name__ == '__main__':

    """
    테스트를 위해 따로 실행됐을 경우, 임의의 샘플 이미지를 불러와서 실행하도록 진행합니다.
    """

    # OCR 글자 인식 테스트
    file_name = './Notebook/SampleImages/sample2.jpg'

    # bytes 형태로 불러와 진 것을 가정합니다.
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
    

    result = get_vision_ocr(content)
    print("OCR-----------------------------")
    print(result)

    print("IMG-----------------------------")
    
    # 이미지 검색 테스트
    file_name = './Notebook/SampleImages/sample2.jpg'

    # bytes 형태로 불러와 진 것을 가정합니다.
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()

    result = get_image_search([content])

    print(result)