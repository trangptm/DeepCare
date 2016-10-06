
command: python training.py -opt [value]
options: 
 		-d : dataset (diabetes or mental)
		-t : task (
            readm - predict readmission
            high_risk - predict high risk patient
						next_diag - next diagnoses
            curr_pm - intervention recommendation)
		-e : pooling at embedding (sqrt, mean, max)
		-r : regularization (
                  norm - L1 + L2
                  drin - drop at input layer (before embedding)
									drhid - drop at hidden layer
                  drfeat (after embedding)
									multi-regularization: -r reg1_reg2_reg3 - e.g., -r norm_drfeat)
