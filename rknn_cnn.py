#!/usr/bin/env python
# coding: utf-8


import numpy as np
import tensorflow as tf
from rknn.api import RKNN

# MFCC 작업을 하기위해 필요합니다
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

sample_rate = 16000             # Wav 파일의 Hz 수치
desired_samples = 16000         # Wav 파일의 Hz 수치
window_size_ms = 30
window_size_samples = 480
window_stride_ms = 10           # 총 음성 파일 길이에서 10ms 마다 슬라이스 해서 분석
window_stride_samples = 160
length_minus_window = 15520
feature_bin_count = 40

# spectrogram_length = 98 입니다
spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
average_window_width = -1

# fingerprint_width = 98 입니다
fingerprint_width = feature_bin_count

# fingerprint_size = 3920 입니다
fingerprint_size = fingerprint_width * spectrogram_length


# 메인 함수가 시작됩니다
if __name__ == '__main__':

	rknn = RKNN()
	print('--> Loading model')
	
	#rknn.config(channel_mean_value='0 0 0 255', reorder_channel='0 1 2')


	# Load TensorFlow Model
	print('--> Loading model')
	rknn.load_tensorflow(tf_pb='./freeze_conv.pb',
                     inputs=['fingerprint_input'],
                     outputs=['ArgMax'],
                     input_size_list=[[3920]])
	print('done')

	# Build Model
	print('--> Building model')
	rknn.build(do_quantization=False)
	print('done')

	# Export RKNN Model
	rknn.export_rknn('./CNN_RKNN.rknn')
	
	# Direct Load RKNN Model
	rknn.load_rknn('./CNN_RKNN.rknn')

	# init runtime environment
	print('--> Init runtime environment')
	ret = rknn.init_runtime()
	if ret != 0:
		print('Init runtime environment failed')
	

	# Inference
	print('--> Running model')
    
	# 가지고있는 Wav 갯수만큼 수정 할 수 있습니다
	for i in range(1,11):
		
		# Wav 파일이 어디있는지 잘 지정해줘야 합니다
		filename = "./data/on/on" + str(i) + ".wav"

        	# 음성파일을 [?, 1]의 형태의 Int 배열로 읽어들입니다
        	# ?는 음성파일이 딱 1초라면 16000이지만 그보다 길고 짧은게 있으므로 그때그때 다릅니다
		wav_loader = tf.io.read_file(filename)
		
        	# [?, 1] 형태의 Int 배열을 -1 ~ 1 사이의 Float 형태, [16000, 1]의 형태로 정규화 해줍니다
		# ?가 16000보다 작으면 0으로 채워지고 넘으면 뒤에는 버려집니다
		wav_decoder = tf.audio.decode_wav(wav_loader, desired_channels=1, desired_samples=desired_samples)
		
		
		# 고속 푸리에 변환을 통해 일렬로 된 음성데이터 파일의 시간축을 주파수축으로 바꾸어서 3차원 배열을 만들어줍니다
		# 아래의 코드는 [1, 98, 257]로 변환됩니다
		# 첫번째 1은 Channel을 뜻하며 스테레오 음성은 2, 5.1 채널 음성데이터는 그에 맞는 숫자가 나올것입니다
		# 두번째 98은 시간축을 의미하며 1초의 음성데이터를 10ms 간격으로 자르고 맨앞과 맨뒤를 버리고 98개가 남은것입니다
		# 세번째 257은 주파수별 음압(dB)의 크기를 담고 있습니다 257종류의 주파수별 dB 크기를 담고있다고 보면 되겠습니다
		spectrogram = contrib_audio.audio_spectrogram(
         				wav_decoder.audio,
          				window_size=window_size_samples,
          				stride=window_stride_samples,
          				magnitude_squared=True)
        
        
       		# 고속 푸리에 변환을 통해 얻은 스펙트로그램을 MFCC 기법을 통해 머신러닝에 유리한 정보만을 추려냅니다
        	# 아래의 코드는 [1, 98, 257]의 배열을 받아서, [1, 98, 40]의 배열로 변합니다
        	# 사람은 고주파의 소리는 잘 못듣고 저주파의 소리엔 민감하다고 합니다 
        	# 이를 통해 좀더 효율적으로 소리의 특징을 추출하기 위해 연구한것이 MFCC라고 보면 됩니다
        	# 257종류의 주파수중에 40개만의 주파수만 쓰려고 골라냈다고 생각하시면 편합니다
		output_ = contrib_audio.mfcc(
           				spectrogram,
            			wav_decoder.sample_rate,
           				dct_coefficient_count=fingerprint_width)
        
        
        	# 신경망에 데이터를 1열로 넣어주기 위해서 길게 펴줍니다
        	# [1, 98, 40]의 데이터가 들어가서 [1, 3920] 형태의 데이터가 나옵니다
		out = tf.compat.v1.layers.Flatten()(output_)
		
		
        	# 이때까지 했던 연산은 모두 Tensor 형태로 연산하는것이였는데 RKNN은 텐서를 이해하지 못합니다
        	# 따라서 Tensor 형태를 Numpy 형태의 배열로 변환해줍니다
		numpy_data = tf.Session().run(out)


		print("----> data standarization complete")
        
        	# 아무말도 하지않는 정적의 음성데이터를 인공적으로 가공한겁니다
		silence = np.zeros([3920], dtype=np.float32)	
        
        	# [1, 3920] Numpy 형태로 변환 해줬던 데이터를 신경망에 넣어줍니다
		test_predict = rknn.inference(inputs=[numpy_data])
		test_predict = np.array(test_predict, dtype = np.float64)
		print('done')
		#print('inference result: ', test_predict)
		#print('result shape: ' , test_predict.shape)
	    
	    	# 결과가 0 이라면 정적입니다
		if(test_predict == 0):
			print("silence");
			
        	# 결과가 1 이라면 학습하지 않은 데이터 입니다
		if(test_predict == 1):
			print("unknown");
			
        	# 결과가 2 라면 Yes 라는 단어입니다
		if(test_predict == 2):
			print("yes");
			
		# 결과가 3 이라면 No 라는 단어입니다
		if(test_predict == 3):
			print("no");
			
        	# 결과가 4 라면 Up 이라는 단어입니다
		if(test_predict == 4):
			print("up");
			
        	# 결과가 5 라면 Down 이라는 단어입니다
		if(test_predict == 5):
			print("down");
			
        	# 결과가 6 이라면 Left 라는 단어입니다
		if(test_predict == 6):
			print("left");
			
        	# 결과가 7 이라면 Right 라는 단어입니다
		if(test_predict == 7):
			print("right");
			
        	# 결과가 8 이라면 On 이라는 단어입니다
		if(test_predict == 8):
			print("on");

        	# 결과가 9 라면 Off 라는 단어입니다
		if(test_predict == 9):
			print("off");
			
        	# 결과가 10 이라면 Stop 이라는 단어입니다
		if(test_predict == 10):
			print("stop");
			
        	# 결과가 11 이라면 Go 라는 단어입니다
		if(test_predict == 11):
			print("go");


	# Evaluate Perf on Simulator
	#rknn.eval_perf(inputs=[std_sample_data])

	# Release RKNN Context
	rknn.release()

