from configure import out_label, configure_verbose_mode
import numpy as np


def write_predictions(filename,event_id,classes_prob):
    assert len(classes_prob[0]) == 2 #probabilities of classes
    assert len(event_id) == len(classes_prob)
    if configure_verbose_mode:
        print("Start writing in the file %s" % (filename))
    
    cp = np.array([row[1]-row[0] for row in classes_prob])
    cp_sorted_inv = cp.argsort()

    cp_sorted = list(cp_sorted_inv)
    for i1,i2 in zip(range(len(cp_sorted_inv)),
                  cp_sorted_inv):
        cp_sorted[i2] = i1
    
    ids = list(map(int,event_id))
    submission = np.array([[str(ids[i]),str(cp_sorted[i]+1),
                       's' if cp[i] >= 0 else 'b'] 
            for i in range(len(event_id))])
    submission = np.append([out_label], submission, axis=0)

    np.savetxt(filename,submission,fmt='%s',delimiter=',')


def write_predictions2(filename, event_id, regressions, threshold):
    assert len(event_id) == len(regressions)
    if configure_verbose_mode:
        print("Start writing in the file %s" % (filename))
    
    rgs = np.array(regressions)
    sSelector = np.array([v <= threshold for v in rgs])

    cp_sorted_inv = rgs.argsort()[::-1]

    cp_sorted = list(cp_sorted_inv)
    for tI,tII in zip(range(len(cp_sorted_inv)),
                  cp_sorted_inv):
        cp_sorted[tII] = tI
    
    ids = list(map(int,event_id))
    submission = np.array([[str(ids[i]),str(cp_sorted[i]+1),
                       's' if sSelector[i] else 'b']
            for i in range(len(event_id))])
    submission = np.append([out_label], submission, axis=0)

    np.savetxt(filename,submission,fmt='%s',delimiter=',')