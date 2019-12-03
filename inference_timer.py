# -*- coding: utf-8 -*-
"""
Assumes model and X_test are loaded as variables.
"""
from timeit import default_timer as timer

times=[]

for i in range(100):  
    test_image=X_test[i].reshape(1,X_test[i].shape[0],X_test[i].shape[1],X_test[i].shape[2])
    start_time = timer()
    y_est=model.predict(test_image)
    end_time = timer()
    times.append(end_time-start_time)
    print(end_time-start_time)
    
print('Average (ms): '+str(sum(times)/len(times)))
print('Max. (ms): '+str(max(times)))
print('Min. (ms): '+str(min(times)))