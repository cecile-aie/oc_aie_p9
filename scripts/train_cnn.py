from p9dg.histo_dataset import HistoDataset, BalancedRoundRobinSampler

train_ds = HistoDataset(root_data="/data",
                        split="train",
                        output_size=256,
                        thresholds_json_path="configs/seuils_par_classe.json")
