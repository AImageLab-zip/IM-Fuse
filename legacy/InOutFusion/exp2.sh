docker run -i -t 944a76565c19 /bin/bash

python train.py --phase train --model_name TF_RMBTS
python train.py --phase test --model_name TF_RMBTS


python train.py --trans ff --checkpoint_dir ./checkpoint/ 
python train.py --phase test --model_name TF_U_Hemis3D --trans ff --checkpoint_dir ./checkpoint/ff- --test_epoch 100

python train.py --phase train --model_name TF_U_Hemis3D --trans basic --checkpoint_dir ./checkpoint/basic-
python train.py --phase test --model_name TF_U_Hemis3D --trans basic --checkpoint_dir ./checkpoint/basic- --test_epoch 100


python train.py --phase test --model_name TF_U_Hemis3D --trans ff --checkpoint_dir ./checkpoint/ff


python evaluation.py --model_name ffTF_U_Hemis3D
python evaluation.py --model_name basic-TF_U_Hemis3D



python evaluation.py --model_name InOut_U_Hemis3D
python train.py --model_name InOut_U_Hemis3D
python train.py --phase test --model_name RsInOut_U_Hemis3D
python train.py --phase train --model_name TF_U_Hemis3D --trans ff --checkpoint_dir ./checkpoint/ff




python evaluation.py --model_name ffTF_U_Hemis3D
python evaluation.py --model_name RsInOut_U_Hemis3D