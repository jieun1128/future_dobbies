# 미래의 도비들 - OCR과 이미지 인식을 통한 검색 엔진 성능 향상 모듈

본 프로젝트는 2021년 동국대학교 팜 경진대회 출품작 `OCR과 이미지 인식을 통한 검색 엔진 성능 향상 모듈` 의 코드 레포지토리입니다.

사용자가 인터넷 게시글을 캡쳐한 이미지를 업로드하면 글자 인식과 광고 제거, 이미지 인식을 통해 가장 출처로 가능성이 높은 주소를 반환하는 검색 엔진과, 이를 사용할 수 있는 Django 웹 어플리케이션으로 구성되어 있습니다.

# 프로젝트 구성
## Django
사진을 업로드하고 검색 결과를 확인할 수 있는 Django 웹 어플리케이션입니다.

## ELK
OCR 결과와 이미지를 입력 받아 출처를 검색하는 ElasticSearch 검색 엔진입니다.

## Image_Detection
업로드한 사진에서 이미지 부분을 검출해 낼 수 있는 Image Detection 모듈입니다.

## TextClassification
OCR인식 결과에서 검색에 불필요한 광고, UI부분 텍스트를 제거하기 위한 모듈입니다.