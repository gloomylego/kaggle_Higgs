from configure import out_label, configure_verbose_mode
from numpy import argsort
def write_predictions(filename,event_id,probabilities,classes):

    if configure_verbose_mode:
        print("Preparing values for writing")
    indices_bs, indices_ss = index_selection(classes, 0)
    bs = probabilities[indices_bs] #backgrounds
    ss = probabilities[indices_ss] #signals
    s_ibs = argsort(bs,kind='heapsort')[::-1]
    s_iss = argsort(ss,kind='heapsort')
    if configure_verbose_mode:
        print("Start writing in the file %s" % (filename))
    
    with open(filename,'w') as f:
        f.write("%s,%s,%s\n" % (out_label[0],out_label[1],out_label[2]))
        for i in range(len(bs)):
            arg = indices_bs[s_ibs[i]]
            #EventId, RankOrder, Class
            f.write("%i,%i,%c\n" % (event_id[arg],i+1,'b'))
        
        for i in range(len(ss)):
            arg = indices_ss[s_iss[i]]
            f.write("%i,%i,%c\n" % (event_id[arg],i+1+len(bs),'s'))
        if configure_verbose_mode:
            print("Success")

def index_selection(select_option, select_val):
    result = []
    others = []
    for i in range(len(select_option)):
        if select_option[i] == select_val:
            result += [i]
        else:
            others += [i]
    return result, others

def selection(selection, select_option, select_val):
    result = []
    for i in range(len(select_option)):
        if select_option[i] == select_val:
            result += selection[i]
    return result