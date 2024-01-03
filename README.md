# SRCNN

목표 : 입력 영상을 고해상도 영상으로 출력




![image](https://github.com/Suzi2n/SRCNN/assets/102611647/f7c93593-0ad4-4f2f-a74d-a07a4c5a9d6c)



![image](https://github.com/Suzi2n/SRCNN/assets/102611647/6bc10a01-4ce1-470a-b0f7-bea7b6626148)


model.py 를 수정하여 SRCNN 구현하는 것이 과제의 목표


SRCNN 구현 후:


1. 코드의 논리구조


train.py

![image](https://github.com/Suzi2n/SRCNN/assets/102611647/f9cba074-2fcb-4fd5-8aaa-7ee3f55cecff)


model = SRCNN().to(device)
train.py 의 model = SRCNN().to(device) 코드는 모델을 생성하고 장치에 할당한다.
여기서 수정한 model.py의 코드는 다음과 같다.



2. PSNR 수치
코드 그대로 수행시 25.66dB
SRCNN 구현 시 26.94 dB
![image](https://github.com/Suzi2n/SRCNN/assets/102611647/5466404c-067b-4acb-b25b-635065614176)
![image](https://github.com/Suzi2n/SRCNN/assets/102611647/24e1d4a0-738f-4698-a537-e254fcf3de99)



3. 입력 / 예측 영상 비교

![image](https://github.com/Suzi2n/SRCNN/assets/102611647/10ba84c2-5a2f-41f4-aa45-539da0289cd4)
입력 영상

![image](https://github.com/Suzi2n/SRCNN/assets/102611647/3bc42525-157d-46b4-8ec3-6a8234bb157a)
예측 영상
