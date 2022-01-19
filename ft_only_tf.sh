python main.py --mode "fine"\
      --data "SST-2"\
      --train\
      --eval\
      --mlm\
      --mlm_loss 0.2\
      --seed 12\
      --weights_name_or_path "(SST2)Token_important_scores_with_mask_2021_05_04_abs.score"\
      --weighted_masking_v3\
      --tfidf\