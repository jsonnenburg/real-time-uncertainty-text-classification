export PYTHONPATH="/Users/johann/Documents/Uni/real-time-uncertainty-text-classification/"

python train_bert_teacher.py --input_data_dir ../../data/robustness_study/preprocessed --output_dir out/bert_teacher  \
--learning_rate 0.00002 --batch_size 16 --epochs 1 --max_length 48 --mc_dropout_inference --seed 42 \
--save_datasets --cleanup