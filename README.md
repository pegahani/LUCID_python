# LUCID_python
Ap, Ae, F, S

Learn on COM, test on real :
Mean Absolute Error: 654.6823121748392
Mean Squared Error: 607051.5243247674
Root Mean Squared Error: 779.135113009783

Learn on NCSIMUL, test on Real:
Mean Absolute Error: 580.3428640869012
Mean Squared Error: 481947.7065515925
Root Mean Squared Error: 694.2245361204057

'Ae','Ap','F','S','AD','Angle','fz','Vc','V','Q','h','InteractMode','ContactMode','AeEquiv','ApEquiv'

Learn on NCSIMUL, test on Real:
Mean Absolute Error: 479.1842727778793
Mean Squared Error: 262089.361562187
Root Mean Squared Error: 511.9466393699513

Learn on Real (80%), test on real (20%):
Mean Squared Error : 3400

Learn on NCSIMUL 90% train 10% evaluation:
478823/478823 [==============================] - 7s 14us/step - loss: 24.6186 - mean_squared_error: 24.6186 - val_loss: 18.4117 - val_mean_squared_error: 18.4117
Relearn on Real (80%) train, 20% evaluation:
425620/425620 [==============================] - 8s 19us/step - loss: 2964.8376 - mean_squared_error: 2964.8376 - val_loss: 3046.6828 - val_mean_squared_error: 3046.6828


'Ae','Ap','F','S':
Learn on NCSIMUL 80% train 20% test:
6s 14us/step - loss: 15496.3613 - mean_squared_error: 15496.3613 - val_loss: 14889.1224 - val_mean_squared_error: 14889.1224
Retrain it on NCSIMUL:
loss: 1047568.6875 - mean_squared_error: 1047568.6875 - val_loss: 692578.6875 - val_mean_squared_error: 692578.6875
