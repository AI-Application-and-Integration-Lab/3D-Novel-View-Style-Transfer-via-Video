for i in {1..20}
do
	python exp.py --net fixed_vgg16unet3_unet4.64.3 --cmd eval --iter 79999 --eval-dsets tat-subseq --eval-scale 0.25 --eval-style-id "$i"
	# mkdir ../../../results/for_quantitative_results/1/"$i"
	# mv ./experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_Auditorium-col_0.25_n4 /media/ai2lab/Data/results/for_quantitative_results/0/t_1/"$i"
	# mv ./experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_Family-col_0.25_n4 /media/ai2lab/Data/results/for_quantitative_results/0/t_1/"$i"
	# mv ./experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_Francis-col_0.25_n4 /media/ai2lab/Data/results/for_quantitative_results/0/t_1/"$i"
	# mv ./experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_Meetingroom-col_0.25_n4 /media/ai2lab/Data/results/for_quantitative_results/0/t_1/"$i"
	# mv ./experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_Palace-col_0.25_n4 /media/ai2lab/Data/results/for_quantitative_results/0/t_1/"$i"
	# mv ./experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_Panther-col_0.25_n4 /media/ai2lab/Data/results/for_quantitative_results/0/t_1/"$i"
	# mv ./experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_Courtroom-col_0.25_n4 /media/ai2lab/Data/results/for_quantitative_results/1/"$i"
	# mv ./experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_Horse-col_0.25_n4 /media/ai2lab/Data/results/for_quantitative_results/1/"$i"
	# mv ./experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_Ignatius-col_0.25_n4 /media/ai2lab/Data/results/for_quantitative_results/1/"$i"
	# mv ./experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_Playground-col_0.25_n4 /media/ai2lab/Data/results/for_quantitative_results/1/"$i"
	# mv ./experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_Truck-col_0.25_n4 /media/ai2lab/Data/results/for_quantitative_results/1/"$i"
	# mv ./experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_Train-col_0.25_n4 /media/ai2lab/Data/results/for_quantitative_results/1/"$i"
	# mv ./experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_Panther-col_0.25_n4 /media/ai2lab/Data/results/for_quantitative_results/1/"$i"

done


# 40 46 52 71 92 107 119
# mv ./experiments/tat_nbs5_s0.25_p192_fixed_vgg16unet3_unet4.64.3/tat_subseq_Courtroom-col_0.25_n4 /media/ai2lab/Data/results/for_user_preference/0/"$i"
