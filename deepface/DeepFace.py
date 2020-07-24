import warnings
warnings.filterwarnings("ignore")
import numpy as np
from tqdm import tqdm
import json
from deepface.extendedmodels import Age, Race
from deepface.commons import functions

def analyze(img_path, actions = [], models = {}, enforce_detection = True):

	if type(img_path) == list:
		img_paths = img_path.copy()
		bulkProcess = True
	else:
		img_paths = [img_path]
		bulkProcess = False

	#---------------------------------

	#if a specific target is not passed, then find them all
	if len(actions) == 0:
		actions= ['age', 'race']

	print("Actions to do: ", actions)

	#---------------------------------
	if 'age' in actions:
		if 'age' in models:
			print("already built age model is passed")
			age_model = models['age']
		else:
			age_model = Age.loadModel()

	if 'race' in actions:
		if 'race' in models:
			print("already built race model is passed")
			race_model = models['race']
		else:
			race_model = Race.loadModel()
	#---------------------------------

	resp_objects = []
	
	global_pbar = tqdm(range(0,len(img_paths)), desc='Analyzing')
	
	#for img_path in img_paths:
	for j in global_pbar:
		img_path = img_paths[j]

		resp_obj = "{"

		#TO-DO: do this in parallel

		pbar = tqdm(range(0,len(actions)), desc='Finding actions')

		action_idx = 0
		img_224 = None # Set to prevent re-detection
		#for action in actions:
		for index in pbar:
			action = actions[index]
			pbar.set_description("Action: %s" % (action))

			if action_idx > 0:
				resp_obj += ", "

			if action == 'age':
				if img_224 is None:
					img_224 = functions.detectFace(img_path, target_size = (224, 224), grayscale = False, enforce_detection = enforce_detection) #just emotion model expects grayscale images
				#print("age prediction")
				age_predictions = age_model.predict(img_224)[0,:]
				apparent_age = Age.findApparentAge(age_predictions)

				resp_obj += "\"age\": %s" % (apparent_age)

			elif action == 'race':
				if img_224 is None:
					img_224 = functions.detectFace(img_path, target_size = (224, 224), grayscale = False, enforce_detection = enforce_detection) #just emotion model expects grayscale images
				race_predictions = race_model.predict(img_224)[0,:]
				race_labels = ['asian', 'indian', 'black', 'white', 'middle eastern', 'latino hispanic']

				sum_of_predictions = race_predictions.sum()

				race_obj = "\"race\": {"
				for i in range(0, len(race_labels)):
					race_label = race_labels[i]
					race_prediction = 100 * race_predictions[i] / sum_of_predictions

					if i > 0: race_obj += ", "

					race_obj += "\"%s\": %s" % (race_label, race_prediction)

				race_obj += "}"
				race_obj += ", \"dominant_race\": \"%s\"" % (race_labels[np.argmax(race_predictions)])

				resp_obj += race_obj

			action_idx = action_idx + 1

		resp_obj += "}"

		resp_obj = json.loads(resp_obj)

		if bulkProcess == True:
			resp_objects.append(resp_obj)
		else:
			return resp_obj

	if bulkProcess == True:
		resp_obj = "{"

		for i in range(0, len(resp_objects)):
			resp_item = json.dumps(resp_objects[i])

			if i > 0:
				resp_obj += ", "

			resp_obj += "\"instance_"+str(i+1)+"\": "+resp_item
		resp_obj += "}"
		resp_obj = json.loads(resp_obj)
		return resp_obj

functions.initializeFolder()

#---------------------------

