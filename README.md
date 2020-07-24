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
