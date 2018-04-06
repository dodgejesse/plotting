
# spaces: arch, reg, reg_bad_lr, reg_half_lr, reg_good_lr, dropl2learn, dropl2learn_bad_lr, dropl2learn_half_lr, dropl2learn_good_lr
# search algos: bayes_opt, mixed_dpp_rbf, random, spearmint_seq spearmint_kover2 sobol_noise
# location:
# sobol_noise random mixed_dpp_rbf bayes_opt

ITERS=40

for SPACE in arch; do
    for SEARCH_TYPE in mixed_dpp_rbf; do
	DATA_NAME="stanford_sentiment_binary,nmodels=1,mdl_tpe=cnn,srch_tpe=${SEARCH_TYPE},spce=${SPACE},iters=${ITERS}"
	DATA_DIR="/home/jessedd/projects/ARKcat/output/archive_${DATA_NAME}"
	echo ${DATA_DIR}
	OUT_LOC="/home/jessedd/projects/plotting/icml_2018/data/${DATA_NAME}.txt"
	
	tail -n $((${ITERS}+1)) ${DATA_DIR}/${SEARCH_TYPE}*/outfile.txt | sed '/^=.*/d' | sed '/total train and eval time/d' > ${OUT_LOC}

    done
done
