To do
=====

Base framework
--------------
 - Write prediction (in case of labels)
   -> Research: prediction based on SOM (how to detect small clusters?)
   SOM.predict(test_vector)
   GridlessSOM


Validation
----------
 - Perform some research for a decent evaluation function (see decisions.txt)


Extensions
----------
 - Toroïdal extensions
   - update are\_neighbours
 - Extend to SOM with free positioning
   - Neighborhood function?
   - Prediction? Fuzzy adhv closest centroid of KNN?


Finetuning
----------
 - Finetune learning rate & neighbourhood function
