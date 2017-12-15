from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys
import random
import tensorflow as tf
import numpy as np
import csv

BASE_PATH = "/Users/Benjamin/Desktop/byu/Semester 8/CS478/group-project"
_CSV_COLUMNS = [
		"year",
		"_Ever_been_told_you_have_asthma",
		"_Doctor_ever_said_you_were_over",
		"_Ever_told_you_had_a_stroke",
		"_Ever_told_you_had_emphysema",
		"_Ever_told_you_had_chronic_bron",
		"_Ever_told_you_had_any_liver_co",
		"_Ever_told_you_had_cancer_or_ma",
		"_Pulse_regular_or_irregular_",
		"_Pulse_type_",
		"_Gender",
		"_Ever_told_you_had_a_thyroid_pr",
		"_Close_relative_had_heart_attac",
		"_Breathing_problem_require_oxyg",
		"_Problem_taking_deep_breath_",
		"_Ever_been_told_you_have_psoria",
		"_Ever_been_told_you_have_celiac",
		"_Are_you_on_a_gluten_free_diet_",
		"_Doctor_told_you_to_lose_weight",
		"_Doctor_told_you_to_exercise",
		"_Doctor_told_you_to_reduce_salt",
		"_Doctor_told_you_to_reduce_fat_",
		"_Are_you_now_controlling_or_los",
		"_Are_you_now_increasing_exercis",
		"_Are_you_now_reducing_salt_in_d",
		"_Are_you_now_reducing_fat_in_di",
		"_Ever_been_told_you_have_jaundi",
		"_Systolic_Blood_pres_1st_rdg_mm",
		"_Diastolic_Blood_pres_1st_rdg_m",
		"_Sagittal_Abdominal_Diameter_1s",
		"_C_reactive_protein_mg_dL_",
		"_Lead_ug_dL_",
		"_Cadmium_ug_L_",
		"_Mercury_total_ug_L_",
		"_Mercury_Inorganic_ug_L_",
		"_Total_cholesterol_mmol_L_",
		"_60_sec_pulse_30_sec_pulse_2_",
		"_Weight_kg_",
		"_Standing_Height_cm_",
		"_Body_Mass_Index_kg_m_2_",
		"_Waist_Circumference_cm_",
		"_Age_at_Screening_Adjudicated_R",
		"_Age_in_Months_Recode",
		"_Annual_Family_Income",
		"_Direct_HDL_Cholesterol_mg_dL_",
		"_Folic_acid_serum_nmol_L_",
		"_Mercury_ethyl_ug_L_",
		"_Mercury_methyl_ug_L_",
		"_Blood_selenium_ug_L_",
		"_Blood_manganese_ug_L_",
		"_Average_Sagittal_Abdominal_Dia",
		"_Increased_fatigue",
		"_High_cholesterol",
		"_5_10_Methenyl_tethrofolic_acid",
		"_5_Formyl_tetrahydrofolic_acid_",
		"__hours_watch_TV_or_videos_past",
		"__of_hours_use_computer_past_30",
		"_Ever_told_you_had_coronary_hea"
]

_CSV_COLUMN_DEFAULTS = [["?"]] * 27
_CSV_COLUMN_DEFAULTS2 = [[0.0]] * 30
_CSV_COLUMN_DEFAULTS += _CSV_COLUMN_DEFAULTS2
_CSV_COLUMN_DEFAULTS.append(["?"])

parser = argparse.ArgumentParser()

parser.add_argument(
		'--model_dir', type=str, default='model',
		help='Base directory for the model.')

parser.add_argument(
		'--model_type', type=str, default='deep',
		help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
		'--train_epochs', type=int, default=15, help='Number of training epochs.')

parser.add_argument(
		'--epochs_per_eval', type=int, default=1,
		help='The number of training epochs to run between evaluations.')

parser.add_argument(
		'--batch_size', type=int, default=5, help='Number of examples per batch.')

parser.add_argument(
		'--train_data', type=str, default=BASE_PATH+'/data/master.data',
		help='Path to the training data.')

parser.add_argument(
		'--test_data', type=str, default=BASE_PATH+'/data/master.test',
		help='Path to the test data.')

_NUM_EXAMPLES = {
		'train': 30498,
		'validation': 13070
}

_NUM_EXAMPLES = {
		'train': 100,
		'validation': 10
}

def build_model_columns():
	# Builds a set of deep feature columns.

	year = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("year" , ["1999","2001","2003","2005","2007","2009","2011","2013"]))
	_Ever_been_told_you_have_asthma = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Ever_been_told_you_have_asthma" , ["1","2"]))
	_Doctor_ever_said_you_were_over = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Doctor_ever_said_you_were_over" , ["1","2"]))
	_Ever_told_you_had_a_stroke = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Ever_told_you_had_a_stroke" , ["1","2"]))
	_Ever_told_you_had_emphysema = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Ever_told_you_had_emphysema" , ["1","2"]))
	_Ever_told_you_had_chronic_bron = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Ever_told_you_had_chronic_bron" , ["1","2"]))
	_Ever_told_you_had_any_liver_co = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Ever_told_you_had_any_liver_co" , ["1","2"]))
	_Ever_told_you_had_cancer_or_ma = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Ever_told_you_had_cancer_or_ma" , ["1","2"]))
	_Pulse_regular_or_irregular_ = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Pulse_regular_or_irregular_" , ["1","2"]))
	_Pulse_type_ = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Pulse_type_" , ["1","2"]))
	_Gender = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Gender" , ["1","2"]))
	_Ever_told_you_had_a_thyroid_pr = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Ever_told_you_had_a_thyroid_pr" , ["1","2"]))
	_Close_relative_had_heart_attac = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Close_relative_had_heart_attac" , ["1","2"]))
	_Breathing_problem_require_oxyg = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Breathing_problem_require_oxyg" , ["1","2"]))
	_Problem_taking_deep_breath_ = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Problem_taking_deep_breath_" , ["1","2"]))
	_Ever_been_told_you_have_psoria = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Ever_been_told_you_have_psoria" , ["1","2"]))
	_Ever_been_told_you_have_celiac = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Ever_been_told_you_have_celiac" , ["1","2"]))
	_Are_you_on_a_gluten_free_diet_ = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Are_you_on_a_gluten_free_diet_" , ["1","2"]))
	_Doctor_told_you_to_lose_weight = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Doctor_told_you_to_lose_weight" , ["1","2"]))
	_Doctor_told_you_to_exercise = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Doctor_told_you_to_exercise" , ["1","2"]))
	_Doctor_told_you_to_reduce_salt = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Doctor_told_you_to_reduce_salt" , ["1","2"]))
	_Doctor_told_you_to_reduce_fat_ = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Doctor_told_you_to_reduce_fat_" , ["1","2"]))
	_Are_you_now_controlling_or_los = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Are_you_now_controlling_or_los" , ["1","2"]))
	_Are_you_now_increasing_exercis = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Are_you_now_increasing_exercis" , ["1","2"]))
	_Are_you_now_reducing_salt_in_d = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Are_you_now_reducing_salt_in_d" , ["1","2"]))
	_Are_you_now_reducing_fat_in_di = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Are_you_now_reducing_fat_in_di" , ["1","2"]))
	_Ever_been_told_you_have_jaundi = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list("_Ever_been_told_you_have_jaundi" , ["1","2"]))

	_Systolic_Blood_pres_1st_rdg_mm = tf.feature_column.numeric_column("_Systolic_Blood_pres_1st_rdg_mm")
	_Diastolic_Blood_pres_1st_rdg_m = tf.feature_column.numeric_column("_Diastolic_Blood_pres_1st_rdg_m")
	_Sagittal_Abdominal_Diameter_1s = tf.feature_column.numeric_column("_Sagittal_Abdominal_Diameter_1s")
	_C_reactive_protein_mg_dL_ = tf.feature_column.numeric_column("_C_reactive_protein_mg_dL_")
	_Lead_ug_dL_ = tf.feature_column.numeric_column("_Lead_ug_dL_")
	_Cadmium_ug_L_ = tf.feature_column.numeric_column("_Cadmium_ug_L_")
	_Mercury_total_ug_L_ = tf.feature_column.numeric_column("_Mercury_total_ug_L_")
	_Mercury_Inorganic_ug_L_ = tf.feature_column.numeric_column("_Mercury_Inorganic_ug_L_")
	_Total_cholesterol_mmol_L_ = tf.feature_column.numeric_column("_Total_cholesterol_mmol_L_")
	_60_sec_pulse_30_sec_pulse_2_ = tf.feature_column.numeric_column("_60_sec_pulse_30_sec_pulse_2_")
	_Weight_kg_ = tf.feature_column.numeric_column("_Weight_kg_")
	_Standing_Height_cm_ = tf.feature_column.numeric_column("_Standing_Height_cm_")
	_Body_Mass_Index_kg_m_2_ = tf.feature_column.numeric_column("_Body_Mass_Index_kg_m_2_")
	_Waist_Circumference_cm_ = tf.feature_column.numeric_column("_Waist_Circumference_cm_")
	_Age_at_Screening_Adjudicated_R = tf.feature_column.numeric_column("_Age_at_Screening_Adjudicated_R")
	_Age_in_Months_Recode = tf.feature_column.numeric_column("_Age_in_Months_Recode")
	_Annual_Family_Income = tf.feature_column.numeric_column("_Annual_Family_Income")
	_Direct_HDL_Cholesterol_mg_dL_ = tf.feature_column.numeric_column("_Direct_HDL_Cholesterol_mg_dL_")
	_Folic_acid_serum_nmol_L_ = tf.feature_column.numeric_column("_Folic_acid_serum_nmol_L_")
	_Mercury_ethyl_ug_L_ = tf.feature_column.numeric_column("_Mercury_ethyl_ug_L_")
	_Mercury_methyl_ug_L_ = tf.feature_column.numeric_column("_Mercury_methyl_ug_L_")
	_Blood_selenium_ug_L_ = tf.feature_column.numeric_column("_Blood_selenium_ug_L_")
	_Blood_manganese_ug_L_ = tf.feature_column.numeric_column("_Blood_manganese_ug_L_")
	_Average_Sagittal_Abdominal_Dia = tf.feature_column.numeric_column("_Average_Sagittal_Abdominal_Dia")
	_Increased_fatigue = tf.feature_column.numeric_column("_Increased_fatigue")
	_High_cholesterol = tf.feature_column.numeric_column("_High_cholesterol")
	_5_10_Methenyl_tethrofolic_acid = tf.feature_column.numeric_column("_5_10_Methenyl_tethrofolic_acid")
	_5_Formyl_tetrahydrofolic_acid_ = tf.feature_column.numeric_column("_5_Formyl_tetrahydrofolic_acid_")
	__hours_watch_TV_or_videos_past = tf.feature_column.numeric_column("__hours_watch_TV_or_videos_past")
	__of_hours_use_computer_past_30 = tf.feature_column.numeric_column("__of_hours_use_computer_past_30")
	_Ever_told_you_had_coronary_hea = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_vocabulary_list(
			'_Ever_told_you_had_coronary_hea', [
					'1','2','?'
					]))

	deep_columns = [
		year,
		_Ever_been_told_you_have_asthma,
		_Doctor_ever_said_you_were_over,
		_Ever_told_you_had_a_stroke,
		_Ever_told_you_had_emphysema,
		_Ever_told_you_had_chronic_bron,
		_Ever_told_you_had_any_liver_co,
		_Ever_told_you_had_cancer_or_ma,
		_Pulse_regular_or_irregular_,
		_Pulse_type_,
		_Gender,
		_Ever_told_you_had_a_thyroid_pr,
		_Close_relative_had_heart_attac,
		_Breathing_problem_require_oxyg,
		_Problem_taking_deep_breath_,
		_Ever_been_told_you_have_psoria,
		_Ever_been_told_you_have_celiac,
		_Are_you_on_a_gluten_free_diet_,
		_Doctor_told_you_to_lose_weight,
		_Doctor_told_you_to_exercise,
		_Doctor_told_you_to_reduce_salt,
		_Doctor_told_you_to_reduce_fat_,
		_Are_you_now_controlling_or_los,
		_Are_you_now_increasing_exercis,
		_Are_you_now_reducing_salt_in_d,
		_Are_you_now_reducing_fat_in_di,
		_Ever_been_told_you_have_jaundi,
		_Systolic_Blood_pres_1st_rdg_mm,
		_Diastolic_Blood_pres_1st_rdg_m,
		_Sagittal_Abdominal_Diameter_1s,
		_C_reactive_protein_mg_dL_,
		_Lead_ug_dL_,
		_Cadmium_ug_L_,
		_Mercury_total_ug_L_,
		_Mercury_Inorganic_ug_L_,
		_Total_cholesterol_mmol_L_,
		_60_sec_pulse_30_sec_pulse_2_,
		_Weight_kg_,
		_Standing_Height_cm_,
		_Body_Mass_Index_kg_m_2_,
		_Waist_Circumference_cm_,
		_Age_at_Screening_Adjudicated_R,
		_Age_in_Months_Recode,
		_Annual_Family_Income,
		_Direct_HDL_Cholesterol_mg_dL_,
		_Folic_acid_serum_nmol_L_,
		_Mercury_ethyl_ug_L_,
		_Mercury_methyl_ug_L_,
		_Blood_selenium_ug_L_,
		_Blood_manganese_ug_L_,
		_Average_Sagittal_Abdominal_Dia,
		_Increased_fatigue,
		_High_cholesterol,
		_5_10_Methenyl_tethrofolic_acid,
		_5_Formyl_tetrahydrofolic_acid_,
		__hours_watch_TV_or_videos_past,
		__of_hours_use_computer_past_30
	]

	return deep_columns


def build_estimator(model_dir, model_type):
	"""Build an estimator appropriate for the given model type."""
	deep_columns = build_model_columns()
	hidden_units = [25]

	# Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
	# trains faster than GPU for this model.
	run_config = tf.estimator.RunConfig().replace(
			session_config=tf.ConfigProto(device_count={'GPU': 0}))

	if model_type == 'wide':
		return tf.estimator.LinearClassifier(
				model_dir=model_dir,
				feature_columns=wide_columns,
				config=run_config)
	elif model_type == 'deep':
		return tf.estimator.DNNClassifier(
				model_dir=model_dir,
				feature_columns=deep_columns,
				hidden_units=hidden_units,
				config=run_config,
				dropout=0.5,)
	else:
		return tf.estimator.DNNLinearCombinedClassifier(
				model_dir=model_dir,
				linear_feature_columns=wide_columns,
				dnn_feature_columns=deep_columns,
				dnn_hidden_units=hidden_units,
				config=run_config)

def input_fn(data_file, num_epochs, shuffle, batch_size):
	"""Generate an input function for the Estimator."""
	assert tf.gfile.Exists(data_file), (
			'%s not found. Please make sure you have '
			'set both arguments --train_data and --test_data.' % data_file)

	def parse_csv(value):
		# print('Parsing', data_file)
		columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
		features = dict(zip(_CSV_COLUMNS, columns))
		# print("features: {}".format(features))
		# labels = features.pop(_CSV_COLUMNS[-1])
		labels = features.pop('_Ever_told_you_had_coronary_hea')
		return features, tf.equal(labels, '1')

	# Extract lines from input files using the Dataset API.
	dataset = tf.data.TextLineDataset(data_file)

	if shuffle:
		dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])

	dataset = dataset.map(parse_csv, num_parallel_calls=5)
	# print("dataset: {}".format(dataset))

	# We call repeat after shuffling, rather than before, to prevent separate
	# epochs from blending together.
	dataset = dataset.repeat(num_epochs)
	dataset = dataset.batch(batch_size)

	iterator = dataset.make_one_shot_iterator()
	features, labels = iterator.get_next()
	# print("features: {}".format(features))
	# print("labels: {}".format(labels))
	return features, labels

def write_list(myList, myFilePath = "out.csv"):
	if type(np.asarray([])) == type(myList):
		 myList = myList.tolist()
	with open(myFilePath, 'ab') as myfile:
		 wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		 if type(myList[0]) == type([]):
			  for x in myList:
					wr.writerow(x)
		 else:
			  wr.writerow(myList)

def main(unused_argv):
	# Clean up the model directory if present
	shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
	model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

	# predictions = list(model.predict(input_fn=lambda: input_fn(FLAGS.test_data, 1, False, FLAGS.batch_size)))
	# print(type(predictions))
	# print("Predictions: {}".format(predictions))

	# Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
	for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
		model.train(input_fn=lambda: input_fn(FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

		# print("variable names: {}".format(model.get_variable_names()))
		# bias_logits = model.get_variable_value('dnn/logits/bias')
		# bias_logits_adagrad = model.get_variable_value('dnn/logits/bias/t_0/Adagrad')
		# kernel_logits = model.get_variable_value('dnn/logits/kernel')
		# kernel_logits_adagrad = model.get_variable_value('dnn/logits/kernel/t_0/Adagrad')
		# write_list(bias_logits, "bias_logits.csv")
		# write_list(bias_logits_adagrad, "bias_logits_adagrad.csv")
		# write_list(kernel_logits, "kernel_logits.csv")
		# write_list(kernel_logits_adagrad, "kernel_logits_adagrad.csv")

		results = model.evaluate(input_fn=lambda: input_fn(FLAGS.test_data, 1, False, FLAGS.batch_size))
		# predictions = model.predict(input_fn=lambda: input_fn(FLAGS.test_data, 1, False, FLAGS.batch_size), predict_keys=_CSV_COLUMNS[0:-2])
		predictions = list(model.predict(input_fn=lambda: input_fn(FLAGS.test_data, 1, False, FLAGS.batch_size)))
		#print(type(predictions))
		print("Predictions: {}".format(predictions))
		pred = [x["probabilities"] for x in predictions]
		write_list(pred, "probabilities {}.csv".format(n))
		class_ids = [x["class_ids"] for x in predictions]
		write_list(class_ids, "class_ids {}.csv".format(n))
		# Display evaluation metrics
		print('\n\nResults at epoch', (n + 1) * FLAGS.epochs_per_eval)
		print('-' * 60)

		for key in sorted(results):
			print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
