#python 3.5


configure_release_mode = True#False #for time saving
configure_verbose_mode = True


filename_train = "../../data/training.csv"
filename_test = "../../data/test.csv"

if not configure_release_mode:
    filename_train = "" "../../data/training_part.csv"
    filename_test = "" "../../data/test_part.csv"
    
args_y = ["Weight","Label"]
args_x = ["EventId","DER_mass_MMC","DER_mass_transverse_met_lep","DER_mass_vis","DER_pt_h","DER_deltaeta_jet_jet","DER_mass_jet_jet","DER_prodeta_jet_jet","DER_deltar_tau_lep","DER_pt_tot","DER_sum_pt","DER_pt_ratio_lep_tau","DER_met_phi_centrality","DER_lep_eta_centrality","PRI_tau_pt","PRI_tau_eta","PRI_tau_phi","PRI_lep_pt","PRI_lep_eta","PRI_lep_phi","PRI_met","PRI_met_phi","PRI_met_sumet","PRI_jet_num","PRI_jet_leading_pt","PRI_jet_leading_eta","PRI_jet_leading_phi","PRI_jet_subleading_pt","PRI_jet_subleading_eta","PRI_jet_subleading_phi","PRI_jet_all_pt"]
args_total_amount = len(args_x)+len(args_y)


out_label = ["EventId","RankOrder","Class"]