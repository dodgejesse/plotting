
# spaces: arch, reg, reg_bad_lr, reg_half_lr, reg_good_lr, dropl2learn, dropl2learn_bad_lr, dropl2learn_half_lr, dropl2learn_good_lr
# search algos: bayes_opt, mixed_dpp_rbf, random, spearmint_seq spearmint_kover2 sobol_noise
# location:


ITERS=20

for SPACE in arch; do
    for SEARCH_TYPE in spearmint_2; do
	DATA_NAME="stanford_sentiment_binary,nmodels=1,mdl_tpe=cnn,srch_tpe=${SEARCH_TYPE},spce=${SPACE},iters=${ITERS}"
	DATA_DIR="/home/jessedd/projects/ARKcat/output/archive_${DATA_NAME}"
	OUT_LOC="/home/jessedd/projects/plotting/icml_2018/data/durations_and_accs/${DATA_NAME}.txt"

	rm -f ${OUT_LOC}
	for CUR_IN_FILE in ${DATA_DIR}/${SEARCH_TYPE}*/outfile.txt; do
	    grep -A1 "durations for training each model" ${CUR_IN_FILE} | tail -n 1 >> ${OUT_LOC}
	    tail -n $((${ITERS}+1)) ${CUR_IN_FILE} | sed '/^=.*/d' | sed '/total train and eval time/d' | awk '{print $1}' >> ${OUT_LOC}
	    echo "" >> ${OUT_LOC}
	    #echo $CUR_IN_FILE
	done
	echo ${DATA_DIR}
    done
done

