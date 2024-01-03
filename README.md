# SRCNN

목표 : 입력 영상을 고해상도 영상으로 출력




![image](https://github.com/Suzi2n/SRCNN/assets/102611647/f7c93593-0ad4-4f2f-a74d-a07a4c5a9d6c)


자료
super_resolution_code.zip
  train.py: 학습 관련 코드
  test.py: 테스트 관련 코드
  • models.py: 모델 관련 코드
  • datasets.py: 데이터 로더 관련 코드
(데이터 전처리및데이터불러오는과정포함)
• DB.zip
  • 학습데이터셋: 6,168장의 저해상도/고해상도 영상
  • 테스트데이터셋: 685장의 저해상도/고해상도영상
  • 저해상도 영상과 고해상도 영상의 쌍이 주어짐(지도학습)


![image](https://github.com/Suzi2n/SRCNN/assets/102611647/6bc10a01-4ce1-470a-b0f7-bea7b6626148)


model.py 를 수정하여 SRCNN 구현하는 것이 과제의 목표


SRCNN 구현 후:


1. 코드의 논리구조
train.py

![image](https://github.com/Suzi2n/SRCNN/assets/102611647/f9cba074-2fcb-4fd5-8aaa-7ee3f55cecff)
