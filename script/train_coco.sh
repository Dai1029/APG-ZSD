#python trainer_COCO.py --manualSeed 806 \
#--cls_weight 0.001 --cls_weight_unseen 0.001 --lr 0.00005 --lr_cls 0.0001 --lz_ratio 0.01 \
#--val_every 1 --cuda --netG_name MLP_G --netD_name MLP_D --lr_step 30 \
#--ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
#--nclass_all 81 --syn_num 300 --nepoch 60 --nepoch_cls 20 \
#--dataset coco --batch_size 96 --nz 300 --attSize 300 --resSize 1024 --gan_epoch_budget 38000 \
#--pretrain_classifier faster-pth/MSCOCO2014_epoch_12.pth \
#--datafeature_root /media/dxm/D/DXM/Feature/E12/train_0.7_0.3_feats.npy \
#--datalabel_root /media/dxm/D/DXM/Feature/E12/train_0.7_0.3_labels.npy \
#--dataroot /media/dxm/D/DXM/dataset/COCO2014/ \
#--testsplit test_0.6_0.3 --trainsplit train_0.7_0.3 --classes_split 65_15 \
#--class_embedding MSCOCO/fasttext.npy \
#--pretrain_classifier_unseen MSCOCO/unseen_Classifier.pth \
#--outname Fasttext_Train/Base/r3/



a=0.75
while (($(bc <<< "$a >=0.65")))
do
  python trainer_COCO.py --manualSeed 806 \
  --cls_weight 0.001 --cls_weight_unseen 0.001 --lr 0.00005 --lr_cls 0.0001 --lz_ratio 0.01 \
  --val_every 1 --cuda --netG_name MLP_G --netD_name MLP_D --lr_step 30 \
  --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
  --nclass_all 81 --syn_num 300 --nepoch 60 --nepoch_cls 20 \
  --dataset coco --batch_size 96 --nz 300 --attSize 300 --resSize 1024 --gan_epoch_budget 38000 \
  --pretrain_classifier faster-pth/MSCOCO2014_epoch_12.pth \
  --datafeature_root /media/dxm/D/DXM/Feature/E12/train_0.7_0.3_feats.npy \
  --datalabel_root /media/dxm/D/DXM/Feature/E12/train_0.7_0.3_labels.npy \
  --dataroot /media/dxm/D/DXM/dataset/COCO2014/ \
  --testsplit test_0.6_0.3 --trainsplit train_0.7_0.3 --classes_split 65_15 \
  --class_embedding MSCOCO/fasttext.npy \
  --pretrain_classifier_unseen MSCOCO/unseen_Classifier.pth \
  --bl "$a" \
  --cfxs 3.0 --unseen_num 5 \
  --outname F_BUC_Train/U3.0C$a/r1/

  python trainer_COCO.py --manualSeed 806 \
  --cls_weight 0.001 --cls_weight_unseen 0.001 --lr 0.00005 --lr_cls 0.0001 --lz_ratio 0.01 \
  --val_every 1 --cuda --netG_name MLP_G --netD_name MLP_D --lr_step 30 \
  --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
  --nclass_all 81 --syn_num 300 --nepoch 60 --nepoch_cls 20 \
  --dataset coco --batch_size 96 --nz 300 --attSize 300 --resSize 1024 --gan_epoch_budget 38000 \
  --pretrain_classifier faster-pth/MSCOCO2014_epoch_12.pth \
  --datafeature_root /media/dxm/D/DXM/Feature/E12/train_0.7_0.3_feats.npy \
  --datalabel_root /media/dxm/D/DXM/Feature/E12/train_0.7_0.3_labels.npy \
  --dataroot /media/dxm/D/DXM/dataset/COCO2014/ \
  --testsplit test_0.6_0.3 --trainsplit train_0.7_0.3 --classes_split 65_15 \
  --class_embedding MSCOCO/fasttext.npy \
  --pretrain_classifier_unseen MSCOCO/unseen_Classifier.pth \
  --bl "$a" \
  --cfxs 3.0 --unseen_num 5 \
  --outname F_BUC_Train/U3.0C$a/r2/

  a=$(bc <<< "$a-0.05")
done

#a=2.6
#while (($(bc <<< "$a <=3.0")))
#do
#  python trainer_COCO.py --manualSeed 806 \
#  --cls_weight 0.001 --cls_weight_unseen 0.001 --lr 0.00005 --lr_cls 0.0001 --lz_ratio 0.01 \
#  --val_every 1 --cuda --netG_name MLP_G --netD_name MLP_D --lr_step 30 \
#  --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
#  --nclass_all 81 --syn_num 300 --nepoch 65 --nepoch_cls 20 \
#  --dataset coco --batch_size 96 --nz 300 --attSize 300 --resSize 1024 --gan_epoch_budget 38000 \
#  --pretrain_classifier faster-pth/MSCOCO2014_epoch_12.pth \
#  --datafeature_root /media/dxm/D/DXM/Feature/E12/train_0.7_0.3_feats.npy \
#  --datalabel_root /media/dxm/D/DXM/Feature/E12/train_0.7_0.3_labels.npy \
#  --dataroot /media/dxm/D/DXM/dataset/COCO2014/ \
#  --testsplit test_0.6_0.3 --trainsplit train_0.7_0.3 --classes_split 65_15 \
#  --class_embedding MSCOCO/fasttext.npy \
#  --pretrain_classifier_unseen MSCOCO/unseen_Classifier.pth \
#  --cfxs "$a" --unseen_num 5 \
#  --outname F_BU_Train/U5_$a/r1/
#
#  python trainer_COCO.py --manualSeed 806 \
#  --cls_weight 0.001 --cls_weight_unseen 0.001 --lr 0.00005 --lr_cls 0.0001 --lz_ratio 0.01 \
#  --val_every 1 --cuda --netG_name MLP_G --netD_name MLP_D --lr_step 30 \
#  --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
#  --nclass_all 81 --syn_num 300 --nepoch 65 --nepoch_cls 20 \
#  --dataset coco --batch_size 96 --nz 300 --attSize 300 --resSize 1024 --gan_epoch_budget 38000 \
#  --pretrain_classifier faster-pth/MSCOCO2014_epoch_12.pth \
#  --datafeature_root /media/dxm/D/DXM/Feature/E12/train_0.7_0.3_feats.npy \
#  --datalabel_root /media/dxm/D/DXM/Feature/E12/train_0.7_0.3_labels.npy \
#  --dataroot /media/dxm/D/DXM/dataset/COCO2014/ \
#  --testsplit test_0.6_0.3 --trainsplit train_0.7_0.3 --classes_split 65_15 \
#  --class_embedding MSCOCO/fasttext.npy \
#  --pretrain_classifier_unseen MSCOCO/unseen_Classifier.pth \
#  --cfxs "$a" --unseen_num 5 \
#  --outname F_BU_Train/U5_$a/r2/
#
#  a=$(bc <<< "$a+0.1")
#done

