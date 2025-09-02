library(readr)
library(cosmosR)
library(dplyr)
library(ggplot2)
library(reshape2)
library(glue)

drug_target_relationships_biomni <- as.data.frame(
  read_csv("data/GDPx2-Annotations/GDPx2-Annotations/drug_target_relationships_biomni.csv"))

TFs_scores <- as.data.frame(read_csv("results/TFs_scores.csv"))

repeat_num = 10
remove_edge_num = 1
for(r in 1:repeat_num) {
  # meta_network <- as.data.frame(
  #   read_csv("support/corrected_PKN_with_BRAF_fixes.csv"))
  # meta_network <- as.data.frame(
  #   read_csv("support/updated_omnipath_PKN_with_corrected_MTOR.csv"))
  # meta_network <- as.data.frame(
  #   read_csv("support/clean_omnipath_PKN.csv"))
  # meta_network <- as.data.frame(
  #   read_csv(glue("support/correct_one_edge/qwen3_8b/corrected_PKN_{r-1}.csv")))
  meta_network <- as.data.frame(
    read_csv(glue("support/correct_{remove_edge_num}_edge/qwen3_8b/corrected_PKN_{r-1}.csv")))
  # meta_network <- as.data.frame(
  #   read_csv("support/qwen3_8b_corrected_PKN.csv"))
  # meta_network <- as.data.frame(
  #   read_csv("support/claude_sonnet_4_corrected_PKN.csv"))
  # meta_network <- as.data.frame(
  #   read_csv("support/correct_one_edge/claude_sonnet_4/corrected_PKN_1.csv"))
  
  
  DEA <- as.data.frame(read_csv("data/GDPx2-ProcessedData/DEA.csv"))
  
  row.names(DEA) <- DEA$gene
  DEA <- DEA[,-which(names(DEA) == "gene")]
  
  meta_network <- meta_network_cleanup(meta_network)
  #Remove genes that are not expressed from the condition#Remove genes that are not expressed from the meta_network
  meta_network <- cosmosR:::filter_pkn_expressed_genes(row.names(DEA), meta_pkn = meta_network)
  # meta_network <- meta_network[-which(meta_network$source == "MAP2K2" & meta_network$target == "PPARG"),] # RW: Comments out because corrected_PKN_with_BRAF_fixes.csv has no this condition
  # meta_network <- meta_network[-which(meta_network$source == "MTOR" & !(meta_network$target %in% c("AKT1", "RPS6KB1", "RPS6KB2", "SGK1", "MAF1"))),]
  
  
  # meta_network[which(meta_network$source == "AURKA" & meta_network$target == "BRCA1"),3] <- -1
  # meta_network <- meta_network[!meta_network$source == "PIK3CA",]
  
  # sub_net <- as.data.frame(rbind(c("PIK3CA","MTOR",1),
  #                                c("PIK3CA","PDPK1",1)))
  # names(sub_net) <- c("source","target","interaction")
  # meta_network <- as.data.frame(rbind(meta_network,sub_net))
  
  drug_target_relationships_biomni <- drug_target_relationships_biomni[drug_target_relationships_biomni$Gene_Symbol %in% meta_network$source,]
  
  # drug_of_interest <- "Torin2"
  # conditions_of_interest <- names(DEA)[grepl(drug_of_interest,names(DEA))]
  # target_of_interest <- "MTOR"
  
  drug_of_interest <- "Dabrafenib"
  conditions_of_interest <- names(DEA)[grepl(drug_of_interest,names(DEA))]
  target_of_interest <- "BRAF"
  
  # condition <- "human_epithelial_melanocytes_Dabrafenib_900"
  moon_res_list <- list()
  target_res_list <- list()
  for(condition in conditions_of_interest)
  {
    RNA_input <- DEA[,names(DEA) == condition]
    names(RNA_input) <- row.names(DEA)
    
    TF_input <- TFs_scores[,names(TFs_scores) == condition]
    names(TF_input) <- TFs_scores$source
    
    #Filter inputs and prune the meta_network to only keep nodes that can be found downstream of the inputs
    #The number of step is quite flexible, 7 steps already covers most of the network
    
    n_steps <- 6
    
    # in this step we prune the network to keep only the relevant part between upstream and downstream nodes
    TF_input <- cosmosR:::filter_input_nodes_not_in_pkn(TF_input, meta_network)
    meta_network <- cosmosR:::keep_observable_neighbours(meta_network, n_steps, names(TF_input))
    
    
    #compress the network
    meta_network_compressed_list <- compress_same_children(meta_network, sig_input = NULL, metab_input = TF_input)
    
    meta_network_compressed <- meta_network_compressed_list$compressed_network
    
    node_signatures <- meta_network_compressed_list$node_signatures
    
    duplicated_parents <- meta_network_compressed_list$duplicated_signatures
    
    meta_network_compressed <- meta_network_cleanup(meta_network_compressed)
    
    # load("support/collectri_regulon_R.RData")
    collectTRI <- as.data.frame(
      read_csv("support/collectTRI.csv"))[,c(1,2,3)]
    
    names(collectTRI)[3] <- "mor"
    
    
    meta_network_TF <- meta_network_compressed
    
    before <- 1
    after <- 0
    i <- 1
    while (before != after & i < 10) {
      before <- length(meta_network_TF[,1])
      moon_res <- moon(upstream_input = NULL, 
                       downstream_input = TF_input, 
                       meta_network = meta_network_TF, 
                       n_layers = n_steps, 
                       statistic = "ulm") 
      
      meta_network_TF <- filter_incohrent_TF_target(moon_res, collectTRI, meta_network_TF, RNA_input)
      after <- length(meta_network_TF[,1])
      i <- i + 1
    }
    
    if(i < 10)
    {
      print(paste("Converged after ",paste(i-1," iterations", sep = ""),sep = ""))
    } else
    {
      print(paste("Interupted after ",paste(i," iterations. Convergence uncertain.", sep = ""),sep = ""))
    }
    
    moon_res_list[[condition]] <- moon_res
    target_res_list[[condition]] <- moon_res[moon_res$source == target_of_interest,"score"]
  }
  
  target_res_df <- data.frame(unlist(target_res_list))
  target_res_df$conc <- gsub(".*_","",row.names(target_res_df))
  names(target_res_df)[1] <- "moon_score"
  target_res_df$conc <- as.numeric(target_res_df$conc)
  
  DEA_conditions_of_interest <- DEA[,names(DEA) %in% conditions_of_interest]
  DEA_target <- DEA_conditions_of_interest[target_of_interest,]
  DEA_target <- as.data.frame(t(DEA_target))
  # DEA_target$conc <- as.numeric( gsub(".*_","",row.names(DEA_target)))
  names(DEA_target) <- "expression_zscore"
  
  target_res_df <- merge(target_res_df, DEA_target, by = "row.names")
  
  # ggplot(target_res_df, aes(conc, moon_score)) +
  #   geom_point() +
  #   stat_smooth(method = "loess", span = 0.5) +
  #   ggtitle(paste(target_of_interest,drug_of_interest,sep = "_")) + xlab("Concentration")
    
  
  scoring_net <- get_moon_scoring_network(target_of_interest,meta_network_TF,moon_res)$ATT
  
  #number of TF and upstream regs
  sum(scoring_net$level == 0)
  sum(scoring_net$level == 1)
  sum(scoring_net$level == 2)
  
  TF_targets_considered <- collectTRI[collectTRI$source %in% scoring_net[scoring_net$level == 0,"source"],"target"]
  TF_targets_considered <- TF_targets_considered[TF_targets_considered %in% names(RNA_input)]
  
  
  target_res_df_long <- melt(target_res_df, id.vars = c("Row.names","conc"))
  
  # ggplot(target_res_df_long, aes(x = conc, y = value, group = variable, color = variable)) +
  #   geom_point() +
  #   stat_smooth(method = "loess", span = 0.5) +
  #   ggtitle(paste(target_of_interest,drug_of_interest,sep = "_")) + xlab("Concentration")
  
  # ggplot(target_res_df_long, aes(x = conc, y = value, group = variable, color = variable)) +
  #   geom_point() +
  #   stat_smooth(method = "loess", span = 0.5) +
  #   ggtitle(paste(target_of_interest,drug_of_interest,"Biomni_remove_9_edges",sep = "_")) + xlab("Concentration")
  
  # write.csv(target_res_df_long, "results/moon_score/remove_1_edge/biomni.csv", row.names = FALSE)
  write.csv(target_res_df_long, glue("results/moon_score/remove_{remove_edge_num}_edge/qwen3_8b_{r-1}", ".csv"), row.names = FALSE)
}
# ggplot(target_res_df_long, aes(x = conc, y = value, group FALSE# ggplot(target_res_df_long, aes(x = conc, y = value, group = variable, color = variable)) +
#   geom_point() +
#   stat_smooth(method = "loess", span = 0.5) +
#   ggtitle(paste(target_of_interest,drug_of_interest,"qwen3_8b_remove_MAP2K2_PPARG",sep = "_")) + xlab("Concentration")

# qwen3_8b_remove_MAP2K1_IRS1
# qwen3_8b_remove_MAP2K2_CASP9

# [REMOVE, MAP2K1, TAL1, -1]*7
# [REMOVE, MAP2K2, CASP9, -1]*9
# [REMOVE, MAP2K1, IRS1, -1]*6
# [REMOVE, MAP2K1, MAPK6, 1]
# [REMOVE, MAP2K2, PPARG, 1] -> This one worked. corrected_PKN_8.csv
# [REMOVE, MAP4K1, MAPK1, 1]*3 + 1 -> Claude sonnet 4 normal
# [REMOVE, MAP2K1, CASP9, -1]
# [REMOVE, MAP4K1, MAP3K1, 1]
# [REMOVE, BRAF, MAP4K1, 1] -> Claude sonnet 4 research

# Edges remove by Biomni:
# 1 BRAF target removed:
# BRAF → MAP4K1
# 6 MAP2K1 targets removed:
#   MAP2K1 → TAL1
#   MAP2K1 → IRS1
#   MAP2K1 → GSK3B
#   MAP2K1 → CASP9
#   MAP2K1 → ARRB2
#   MAP2K1 → RPS6KA4
# 2 MAP2K2 targets removed:
#   MAP2K2 → CASP9
#   MAP2K2 → PPARG