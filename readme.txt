This folder contains the data for the Drug Screening Classification project for ABR 02-450/750 at CMU

There are 9 csv files, organized into three categories

"Easy"
EASY_TRAIN.csv: This is a 4000 by 1001 matrix containing the features and labels for 4000 training compounds. 
		Each compound is described by 1000 features, which corresponds to the first 1000 columns in the 
		matrix. The final column is the class label. Use this file as your pool (or stream) for active
		learning. 

EASY_TEST.csv: 	This is a 1000 by 1001 matrix containing the features and labels for 1000 test compounds. 
		Each compound is described by 1000 features, which corresponds to the first 1000 columns in the 
		matrix. The final column is the class label. Use this file to compute test errors.

EASY_BLINDED.csv:  This is a 1000 by 1001 matrix containing the features and labels for 1000 test compounds. 
		   The first column is a unique id. The remaining 1000 columns are the features. There is 
		   no label in this file. Use this file to make blinded predictions. Your predictions should
		   be in a text file named EASY_BLINDED.csv. Each line of that file should have the following
		   format:  <ID>, prediction


"Moderate"
MODERATE_TRAIN.csv: This is a 4000 by 1001 matrix containing the features and labels for 4000 training compounds. 
		Each compound is described by 1000 features, which corresponds to the first 1000 columns in the 
		matrix. The final column is the class label. Use this file as your pool (or stream) for active
		learning. 

MODERATE_TEST.csv: This is a 1000 by 1001 matrix containing the features and labels for 1000 test compounds. 
		   Each compound is described by 1000 features, which corresponds to the first 1000 columns in the 
		   matrix. The final column is the class label. Use this file to compute test errors.

MODERATE_BLINDED.csv:   This is a 1000 by 1001 matrix containing the features and labels for 1000 test compounds. 
		   	The first column is a unique id. The remaining 1000 columns are the features. There is 
		   	no label in this file. Use this file to make blinded predictions. Your predictions should
		   	be in a text file named MODERATE_BLINDED.csv. Each line of that file should have the following
		   	format:  <ID>, prediction



"Difficult"
DIFFICULT_TRAIN.csv: This is a 4000 by 1001 matrix containing the features and labels for 4000 training compounds. 
		Each compound is described by 1000 features, which corresponds to the first 1000 columns in the 
		matrix. The final column is the class label. Use this file as your pool (or stream) for active
		learning. 
DIFFICULT_TEST.csv: This is a 1000 by 1001 matrix containing the features and labels for 1000 test compounds. 
		   Each compound is described by 1000 features, which corresponds to the first 1000 columns in the 
		   matrix. The final column is the class label. Use this file to compute test errors.

DIFFICULT_BLINDED.csv:   This is a 1000 by 1001 matrix containing the features and labels for 1000 test compounds. 
		   	The first column is a unique id. The remaining 1000 columns are the features. There is 
		   	no label in this file. Use this file to make blinded predictions. Your predictions should
		   	be in a text file named DIFFICULT_BLINDED.csv. Each line of that file should have the following
		   	format:  <ID>, prediction
