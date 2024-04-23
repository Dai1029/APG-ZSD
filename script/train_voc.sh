#python trainer_VOC.py --manualSeed 806 \
#--cls_weight 0 --cls_weight_unseen 0.1 --nclass_all 21 --syn_num 500 --val_every 1 \
#--cuda --netG_name MLP_G --netD_name MLP_D \
#--nepoch 140  --nepoch_cls 25 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
#--dataset voc --batch_size 64 --nz 300 --attSize 300 --resSize 1024 --lr 0.00001 \
#--lr_step 20 --gan_epoch_budget 38000 --lr_cls 0.00005 \
#--pretrain_classifier faster-pth/PASCALVOC_epoch_4.pth \
#--testsplit test_0.7_0.3 --trainsplit train_0.7_0.3 --lz_ratio 0 \
#--class_embedding VOC/fasttext.npy \
#--pretrain_classifier_unseen VOC/fasttext_trv_tev.pth \
#--datafeature_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_feats.npy \
#--datalabel_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_labels.npy \
#--outname VOC_Base/ORI/r3

#
#python trainer_VOC.py --manualSeed 806 \
#--cls_weight 0 --cls_weight_unseen 0.1 --nclass_all 21 --syn_num 500 --val_every 1 \
#--cuda --netG_name MLP_G --netD_name MLP_D \
#--nepoch 140  --nepoch_cls 25 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
#--dataset voc --batch_size 64 --nz 300 --attSize 300 --resSize 1024 --lr 0.00001 \
#--lr_step 20 --gan_epoch_budget 38000 --lr_cls 0.00005 \
#--pretrain_classifier faster-pth/PASCALVOC_epoch_4.pth \
#--testsplit test_0.7_0.3 --trainsplit train_0.7_0.3 --lz_ratio 0 \
#--class_embedding VOC/fasttext.npy \
#--pretrain_classifier_unseen VOC/fasttext_trv_tev.pth \
#--datafeature_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_feats.npy \
#--datalabel_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_labels.npy \
#--outname VOC_Base/ORI/r2
#--cfxs 0.3 --unseen_num 5 --bl 0.8 \




a=1.2
while (($(bc <<< "$a <2.0")))
do
  a=$(bc <<< "$a+0.2")

  python trainer_VOC.py --manualSeed 806 \
  --cls_weight 0 --cls_weight_unseen 0.1 --nclass_all 21 --syn_num 500 --val_every 1 \
  --cuda --netG_name MLP_G --netD_name MLP_D \
  --nepoch 140  --nepoch_cls 25 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
  --dataset voc --batch_size 64 --nz 300 --attSize 300 --resSize 1024 --lr 0.00001 \
  --lr_step 20 --gan_epoch_budget 38000 --lr_cls 0.00005 \
  --pretrain_classifier faster-pth/PASCALVOC_epoch_4.pth \
  --testsplit test_0.7_0.3 --trainsplit train_0.7_0.3 --lz_ratio 0 \
  --class_embedding VOC/fasttext.npy \
  --pretrain_classifier_unseen VOC/fasttext_trv_tev.pth \
  --datafeature_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_feats.npy \
  --datalabel_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_labels.npy \
  --cfxs "$a" --unseen_num 15 \
  --outname VOC_BU/15_U$a/r1

done
#
#a=0.1
#while (($(bc <<< "$a <0.4")))
#do
#  a=$(bc <<< "$a+0.2")
#
#  python trainer_VOC.py --manualSeed 806 \
#  --cls_weight 0 --cls_weight_unseen 0.1 --nclass_all 21 --syn_num 500 --val_every 1 \
#  --cuda --netG_name MLP_G --netD_name MLP_D \
#  --nepoch 150  --nepoch_cls 25 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
#  --dataset voc --batch_size 64 --nz 300 --attSize 300 --resSize 1024 --lr 0.00001 \
#  --lr_step 20 --gan_epoch_budget 38000 --lr_cls 0.00005 \
#  --pretrain_classifier faster-pth/PASCALVOC_epoch_4.pth \
#  --testsplit test_0.7_0.3 --trainsplit train_0.7_0.3 --lz_ratio 0 \
#  --class_embedding VOC/fasttext.npy \
#  --pretrain_classifier_unseen VOC/fasttext_trv_tev.pth \
#  --datafeature_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_feats.npy \
#  --datalabel_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_labels.npy \
#  --cfxs "$a" --unseen_num 5 \
#  --outname VOC_BU/New/5_U$a/r1
#
#    python trainer_VOC.py --manualSeed 806 \
#  --cls_weight 0 --cls_weight_unseen 0.1 --nclass_all 21 --syn_num 500 --val_every 1 \
#  --cuda --netG_name MLP_G --netD_name MLP_D \
#  --nepoch 150  --nepoch_cls 25 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
#  --dataset voc --batch_size 64 --nz 300 --attSize 300 --resSize 1024 --lr 0.00001 \
#  --lr_step 20 --gan_epoch_budget 38000 --lr_cls 0.00005 \
#  --pretrain_classifier faster-pth/PASCALVOC_epoch_4.pth \
#  --testsplit test_0.7_0.3 --trainsplit train_0.7_0.3 --lz_ratio 0 \
#  --class_embedding VOC/fasttext.npy \
#  --pretrain_classifier_unseen VOC/fasttext_trv_tev.pth \
#  --datafeature_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_feats.npy \
#  --datalabel_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_labels.npy \
#  --cfxs "$a" --unseen_num 5 \
#  --outname VOC_BU/New/5_U$a/r2
#
#  python trainer_VOC.py --manualSeed 806 \
#  --cls_weight 0 --cls_weight_unseen 0.1 --nclass_all 21 --syn_num 500 --val_every 1 \
#  --cuda --netG_name MLP_G --netD_name MLP_D \
#  --nepoch 150  --nepoch_cls 25 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
#  --dataset voc --batch_size 64 --nz 300 --attSize 300 --resSize 1024 --lr 0.00001 \
#  --lr_step 20 --gan_epoch_budget 38000 --lr_cls 0.00005 \
#  --pretrain_classifier faster-pth/PASCALVOC_epoch_4.pth \
#  --testsplit test_0.7_0.3 --trainsplit train_0.7_0.3 --lz_ratio 0 \
#  --class_embedding VOC/fasttext.npy \
#  --pretrain_classifier_unseen VOC/fasttext_trv_tev.pth \
#  --datafeature_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_feats.npy \
#  --datalabel_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_labels.npy \
#  --cfxs "$a" --unseen_num 5 \
#  --outname VOC_BU/New/5_U$a/r3
#
#done

#a=0.75
#while (($(bc <<< "$a >=0.65")))
#do
#  python trainer_VOC.py --manualSeed 806 \
#  --cls_weight 0 --cls_weight_unseen 0.1 --nclass_all 21 --syn_num 500 --val_every 1 \
#  --cuda --netG_name MLP_G --netD_name MLP_D \
#  --nepoch 140  --nepoch_cls 25 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
#  --dataset voc --batch_size 64 --nz 300 --attSize 300 --resSize 1024 --lr 0.00001 \
#  --lr_step 20 --gan_epoch_budget 38000 --lr_cls 0.00005 \
#  --pretrain_classifier faster-pth/PASCALVOC_epoch_4.pth \
#  --testsplit test_0.7_0.3 --trainsplit train_0.7_0.3 --lz_ratio 0 \
#  --class_embedding VOC/fasttext.npy \
#  --pretrain_classifier_unseen VOC/fasttext_trv_tev.pth \
#  --datafeature_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_feats.npy \
#  --datalabel_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_labels.npy \
#  --bl "$a" --outname VOC_BC/B75C0005/r1
#
#  python trainer_VOC.py --manualSeed 806 \
#  --cls_weight 0 --cls_weight_unseen 0.1 --nclass_all 21 --syn_num 500 --val_every 1 \
#  --cuda --netG_name MLP_G --netD_name MLP_D \
#  --nepoch 140  --nepoch_cls 25 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
#  --dataset voc --batch_size 64 --nz 300 --attSize 300 --resSize 1024 --lr 0.00001 \
#  --lr_step 20 --gan_epoch_budget 38000 --lr_cls 0.00005 \
#  --pretrain_classifier faster-pth/PASCALVOC_epoch_4.pth \
#  --testsplit test_0.7_0.3 --trainsplit train_0.7_0.3 --lz_ratio 0 \
#  --class_embedding VOC/fasttext.npy \
#  --pretrain_classifier_unseen VOC/fasttext_trv_tev.pth \
#  --datafeature_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_feats.npy \
#  --datalabel_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_labels.npy \
#  --bl "$a" --outname VOC_BC/B75C0005/r2
#
#  a=$(bc <<< "$a-0.05")
#done
#
#python trainer_VOC.py --manualSeed 806 \
#--cls_weight 0 --cls_weight_unseen 0.1 --nclass_all 21 --syn_num 500 --val_every 1 \
#--cuda --netG_name MLP_G --netD_name MLP_D \
#--nepoch 140  --nepoch_cls 25 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
#--dataset voc --batch_size 64 --nz 300 --attSize 300 --resSize 1024 --lr 0.00001 \
#--lr_step 20 --gan_epoch_budget 38000 --lr_cls 0.00005 \
#--pretrain_classifier faster-pth/PASCALVOC_epoch_4.pth \
#--testsplit test_0.7_0.3 --trainsplit train_0.7_0.3 --lz_ratio 0 \
#--class_embedding VOC/fasttext.npy \
#--pretrain_classifier_unseen VOC/fasttext_trv_tev.pth \
#--datafeature_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_feats.npy \
#--datalabel_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_labels.npy \
#--bl 0.8 --outname VOC_BS/B.80/r4
#
#python trainer_VOC.py --manualSeed 806 \
#--cls_weight 0 --cls_weight_unseen 0.1 --nclass_all 21 --syn_num 500 --val_every 1 \
#--cuda --netG_name MLP_G --netD_name MLP_D \
#--nepoch 140  --nepoch_cls 25 --ngh 4096 --ndh 4096 --lambda1 10 --critic_iter 5 \
#--dataset voc --batch_size 64 --nz 300 --attSize 300 --resSize 1024 --lr 0.00001 \
#--lr_step 20 --gan_epoch_budget 38000 --lr_cls 0.00005 \
#--pretrain_classifier faster-pth/PASCALVOC_epoch_4.pth \
#--testsplit test_0.7_0.3 --trainsplit train_0.7_0.3 --lz_ratio 0 \
#--class_embedding VOC/fasttext.npy \
#--pretrain_classifier_unseen VOC/fasttext_trv_tev.pth \
#--datafeature_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_feats.npy \
#--datalabel_root /media/dxm/D/DXM/Feature/VOC_TV_T/train_0.7_0.3_labels.npy \
#--bl 0.8 --outname VOC_BS/B.80/r5


