import csv

filename_train = "../../data/training.csv"
filename_test = "../../data/test.csv"
args_total = 33#len(EventId,DER_mass_MMC,DER_mass_transverse_met_lep,DER_mass_vis,DER_pt_h,DER_deltaeta_jet_jet,DER_mass_jet_jet,DER_prodeta_jet_jet,DER_deltar_tau_lep,DER_pt_tot,DER_sum_pt,DER_pt_ratio_lep_tau,DER_met_phi_centrality,DER_lep_eta_centrality,PRI_tau_pt,PRI_tau_eta,PRI_tau_phi,PRI_lep_pt,PRI_lep_eta,PRI_lep_phi,PRI_met,PRI_met_phi,PRI_met_sumet,PRI_jet_num,PRI_jet_leading_pt,PRI_jet_leading_eta,PRI_jet_leading_phi,PRI_jet_subleading_pt,PRI_jet_subleading_eta,PRI_jet_subleading_phi,PRI_jet_all_pt,Weight,Label)
args_x = 31 #first is EventId
args_y = 2


with open(filename_train, 'r') as f:
    sub = csv.reader(f)
    next(sub)#sub.next()#
    x_train = sub[:,0:args_x-1]
    y_train = sub[:,args_x:args_total-1]
