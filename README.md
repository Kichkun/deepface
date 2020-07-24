# Age and Race predictor from single selfie

## Installation

```bash
git clone #https://github.com/Kichkun/deepface.git
cd deepface
pip install .
```

**Age and Race recognition** 
including asian, white, middle eastern, indian, latino and black predictions. 
```python
from deepface import DeepFace
demography = DeepFace.analyze("img4.jpg", actions = ['age', 'race'])
#demographies = DeepFace.analyze(["img1.jpg", "img2.jpg", "img3.jpg"]) #analyzing multiple faces same time
print("Age: ", demography["age"])
print("Race: ", demography["dominant_race"])
```

<p align="center"><img src="https://makeameme.org/media/templates/250/the_most_interesting_man_in_the_world.jpg" width="95%" height="95%"></p>

```bash
{'age': 50.75593731981769,
 'dominant_race': 'white',
 'race': {'asian': 1.3692333378223509e-08,
  'black': 1.682867603949867e-10,
  'indian': 1.8962782455389515e-07,
  'latino hispanic': 0.0014391415788850281,
  'middle eastern': 0.01635842927498743,
  'white': 99.98220801353455}}
```