# Using the Fall Detection Service

We assume that you have finished the tasks in How to run the code.md.

This file will explain how to understand the service.

Once you have copied over the model_pos.pkl and scaler_pos.pkl files please start up the UWB positioning service.

All of the relevant fall detection services will be running in the command terminal that started the  ```backend``` code.

### Fall Detection Loop
1) Thread Starts - A daemon thread is started that runs the fall detection service
2) Tag Lockout - A small queue stores a tuple (tag,unlock time). Expired tags are cleared after the time passes
3) Per tag processing 
```
for all keys in the redis cache:
    if key is locked: pass
    
    Consume all the data (positonal data) for the tag from the resis queue
        If tag already has a queue extend the queue
        If not create a new queue

```
4) Once a tag has more than MIN_SAMPLES amount of datat then tag is now monitored for falls
5) We operate on maintaining a sliding window BUFFER
6) During the buffer we preprocess the data according to our preprocessing pipeline (lid our in how to run the code)
7) Threshold check -See if the peak z minus 25th percentile of next Z values >= 0.4. This means that a drop of 0.4 meters has been detected
8) If the threshold is passed then the classification using the trained model is activated
9) If a fall has been detected then a notification is sent out and the tag is locked for 30 seconds, also the redis list and the queue is cleared.


Each tag has a minimum 4 second loop period.


### Notification
If a fall has been detected we send out a notification.

Description : "Tag :{tag number}, has fallen"

Then we send a fall notifcation and create a new incident event which can be viewed on the front end.
