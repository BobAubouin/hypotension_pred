from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from tqdm.contrib.concurrent import process_map


class Extract_feature_from_segment():
    def __init__(self, segment_data):
        self.segment_data = segment_data

        self.matrice = np.zeros((121, 2))
        self.matrice[:59, 0] = 1
        self.matrice[60:120, 1] = 2
        self.matrice1 = [1, 0]
        self.matrice2 = [0, 1]

    def liste_temps_reel_hypotension(self,
                                     parametre: str,
                                     seuil: float,
                                     largeur_window: int):
        list_sub_feature = []
        data = self.segment_data[[col for col in self.segment_data.columns if col.startswith(parametre)]].copy()
        data = data.transpose()
        data.reset_index(drop=True, inplace=True)
        for i in range(len(data)-largeur_window+1):
            if (data.iloc[i:i+largeur_window].values < seuil).all():
                sub_feat = 2
            elif (data.iloc[i:i+largeur_window].values < seuil).any():
                sub_feat = 1
            else:
                sub_feat = 0
            list_sub_feature.append(sub_feat)

        temp = np.dot(list_sub_feature, self.matrice)
        feat_1 = np.dot(temp, self.matrice1)
        feat_2 = np.dot(temp, self.matrice2)
        df_features = pd.DataFrame({f"{parametre}_1": feat_1,
                                    f"{parametre}_2": feat_2},
                                   index=[0])
        if parametre == 'mbp':
            phd = self.compter_occurrences(list_sub_feature)
            df_features = pd.concat([df_features, phd], axis=1)

        return df_features

    def liste_temps_reel_hypotension_v2(self, parametre: str,
                                        seuil1: float,
                                        seuil2: float,
                                        largeur_window: int):
        list_sub_feature = []
        data = self.segment_data[[col for col in self.segment_data.columns if col.startswith(parametre)]].copy()
        data = data.transpose()
        data.reset_index(drop=True, inplace=True)
        for i in range(len(data)-largeur_window+1):
            if (data.iloc[i:i+largeur_window].values < seuil2).all():
                sub_feat = 5
            elif (data.iloc[i:i+largeur_window].values < seuil1).all():
                sub_feat = 4
            elif (data.iloc[i:i+largeur_window].values < seuil2).any():
                sub_feat = 3
            elif (data.iloc[i:i+largeur_window].values < seuil1).any():
                sub_feat = 2
            elif (data.iloc[i:i+largeur_window].values >= seuil1).all():
                sub_feat = 1
            else:
                sub_feat = 0  # this case is impossible
            list_sub_feature.append(sub_feat)

        temp = np.dot(list_sub_feature, self.matrice)
        feat_1 = np.dot(temp, self.matrice1)
        feat_2 = np.dot(temp, self.matrice2)
        df_features = pd.DataFrame({f"{parametre}_1_v2": feat_1,
                                    f"{parametre}_2_v2": feat_2},
                                   index=[0])
        return df_features

    @staticmethod
    def compter_occurrences(list_value: pd.DataFrame):
        df_features = pd.DataFrame(index=[0])
        if len(list_value) != 0:
            ph_feat = list_value.count(2)/len(list_value)
            pd_feat = list_value.count(1)/len(list_value)
            df_features['ph'] = ph_feat
            df_features['pd'] = pd_feat
            return df_features
        else:
            return None

    @staticmethod
    def central_tendency_measure(map_signal: list):
        count = 0
        variations = np.diff(map_signal)
        rho = 1.5*np.std(variations)
        for i in range(len(map_signal)-2):
            racine = np.sqrt((map_signal[i+2]-map_signal[i+1])**2+(map_signal[i+1]-map_signal[i])**2)
            if racine < rho:
                count += 1
        return count/(len(map_signal)-2)

    @staticmethod
    def phi(map_signal: list, m: int):
        N = len(map_signal)
        if N <= m:
            return 0
        std_dev = np.std(map_signal)
        r = 0.2 * std_dev
        patterns = [map_signal[i:i + m] for i in range(N - m + 1)]
        patterns = np.array(patterns)
        distances = np.abs(patterns[:, None, :] - patterns[None, :, :]).max(axis=2)
        counts = np.sum(distances <= r, axis=0)
        counts = np.where(counts == 0, 1, counts)
        return np.sum(np.log(counts / (N - m + 1))) / (N - m + 1)

    def approximative_entropy(self, map_signal: pd.DataFrame, m=2):
        map_signal = map_signal.values.squeeze()
        phi_m = self.phi(map_signal, m)
        phi_m_plus_1 = self.phi(map_signal, m + 1)
        return phi_m - phi_m_plus_1

    @staticmethod
    def map_hypotension_index(map_signal: pd.DataFrame,
                              fs: float,
                              seuil: float):
        temps_total = len(map_signal)/fs
        hypotensions = 0
        for i in range(1, len(map_signal)):
            if map_signal.iloc[i-1].item() >= seuil and map_signal.iloc[i].item() <= seuil:
                hypotensions += 1
        return hypotensions/(temps_total/3600)

    @staticmethod
    def detect_hypotension_episodes(data_map: pd.DataFrame,
                                    seuil: float = 65,
                                    duration: int = 30):
        episodes = []
        inf_seuil = data_map < seuil
        i = 0
        while i < len(inf_seuil):
            if inf_seuil.iloc[i].item():
                debut = i
                while i < len(inf_seuil) and inf_seuil.iloc[i].item():
                    i += 1
                fin = i
                if (fin-debut) >= duration:
                    episodes.append((debut, fin))
            i += 1
        return episodes

    @staticmethod
    def calculer_puissance(map_signal: pd.DataFrame,
                           episodes: list):

        n = len(map_signal)
        map_signal = map_signal.values.squeeze()
        p_m = np.sum(np.square(map_signal))/n
        p_h_i = []
        for debut, fin in episodes:
            p_i = np.sum(np.square(map_signal[debut:fin]))/(fin-debut)
            p_h_i.append(p_i)
        n_h = len(p_h_i)
        p_h = np.mean(p_h_i) if n_h > 0 else 0
        r = p_h / p_m if p_m != 0 else 0

        df_features = pd.DataFrame({'p_m': p_m,
                                    'p_h': p_h,
                                    'r': r},
                                   index=[0])
        return df_features

    def puissance(self):
        data = self.segment_data[[col for col in self.segment_data.columns if col.startswith('mbp')]].copy()
        data = data.transpose()
        data.reset_index(drop=True, inplace=True)

        episodes = self.detect_hypotension_episodes(data)
        df_features = self.calculer_puissance(data, episodes)
        return df_features

    def parametre_statistique(self):
        data = self.segment_data[[col for col in self.segment_data.columns if col.startswith('mbp')]]
        data_mbp = data.transpose()
        data_mbp.reset_index(drop=True, inplace=True)
        seuil = 65
        fs = 0.5
        n = 3
        var1 = len(data_mbp)//n
        n = int(n)
        dict_features = {}
        for i in range(n):
            # select sub-segment
            i1 = i*var1
            i2 = (i+1)*var1 if i < n else len(data_mbp)
            map_signal = data_mbp.iloc[i1:i2].copy()
            map_signal.reset_index(drop=True, inplace=True)
            valeur_moyenne1 = map_signal.mean()
            map_signal = map_signal.fillna(valeur_moyenne1)

            # compute features
            r1 = map_signal.diff().abs().mean()
            r2 = self.central_tendency_measure(map_signal.values.squeeze())
            r3 = 100 * (map_signal < seuil).sum(axis=0).item() / len(map_signal)
            r4 = self.approximative_entropy(map_signal)
            r5 = self.map_hypotension_index(map_signal, fs, seuil)
            dict_features[f'mbp_{i}_r1'] = r1
            dict_features[f'mbp_{i}_r2'] = r2
            dict_features[f'mbp_{i}_r3'] = r3
            dict_features[f'mbp_{i}_r4'] = r4
            dict_features[f'mbp_{i}_r5'] = r5
        df_features = pd.DataFrame(dict_features, index=[0])
        return df_features

    def reg_polynomiale(self,
                        parametre: str = 'mbp',
                        ordre: int = 3):
        n = 3  # self.minutes//50
        data_mbp = self.segment_data[[col for col in self.segment_data.columns if col.startswith(parametre)]]
        data_mbp = data_mbp.transpose()
        data_mbp.reset_index(drop=True, inplace=True)

        var1 = len(data_mbp)//n
        var1 = int(var1)
        n = int(n)
        dict_features = {}
        for i in range(n):
            i1 = i*var1
            i2 = (i+1)*var1 if i < n else len(data_mbp)
            sub_segment = data_mbp.iloc[i1:i2].copy()
            if sub_segment.isnull().any().item():
                valeur_moyenne = sub_segment.mean()
                sub_segment = sub_segment.fillna(valeur_moyenne)
            x = np.linspace(0, (len(sub_segment)-1)*2, len(sub_segment))
            x2 = x.reshape(-1, 1)
            y = sub_segment.values.squeeze()
            poly = PolynomialFeatures(degree=ordre)
            x_poly = poly.fit_transform(x2)
            model = LinearRegression()
            model.fit(x_poly, y)
            coefs = model.coef_
            intercept = model.intercept_
            for j in range(ordre):
                dict_features[f'{parametre}_{i}_coef_{j}'] = coefs[j]
            dict_features[f'{parametre}_{i}_intercept'] = intercept
        df_features = pd.DataFrame(dict_features, index=[0])
        return df_features

    def extract_all_features(self):
        df_map = self.liste_temps_reel_hypotension("mbp", 65, 30)
        df_bis = self.liste_temps_reel_hypotension("bis", 40, 30)
        df_sap = self.liste_temps_reel_hypotension("sbp", 90, 30)
        df_dap = self.liste_temps_reel_hypotension("dbp", 40, 30)
        df_map2 = self.liste_temps_reel_hypotension_v2("mbp", 65, 50, 30)
        df_bis2 = self.liste_temps_reel_hypotension_v2("bis", 40, 20, 30)
        df_sap2 = self.liste_temps_reel_hypotension_v2("sbp", 90, 55, 30)
        df_dap2 = self.liste_temps_reel_hypotension_v2("dbp", 40, 25, 30)
        puissance_feat = self.puissance()
        statistique_liste = self.parametre_statistique()
        reg_liste = self.reg_polynomiale()

        df_features = pd.concat([df_map, df_bis, df_sap, df_dap, df_map2, df_bis2, df_sap2,
                                df_dap2, puissance_feat, statistique_liste, reg_liste], axis=1)

        return df_features


def extract_features_from_raw_dataset(raw_dataset: pd.DataFrame):
    features_cases = []
    for id, segment_data in raw_dataset.iterrows():
        segment_data = pd.DataFrame(segment_data).T
        segment_data.index = [0]
        feature_extractor = Extract_feature_from_segment(segment_data)
        features = feature_extractor.extract_all_features()
        features = pd.concat(
            [features, segment_data[["caseid", "time", "label", "intervention",	"time_before_IOH",	"label_id"]]],
            axis=1)
        features_cases.append(features)

    return pd.concat(features_cases)


def test_number_sup(str, max_time):
    if 'time' in str or 'label' in str:
        return True
    if '_' in str:
        number = str.split('_')[1]
        if int(number) < max_time:
            return True
        else:
            return False
    else:
        return True


def select_segments(data_raw, max_time):
    data_raw = data_raw[[col for col in data_raw.columns if test_number_sup(col, max_time)]]
    return data_raw


def process_case(args):
    """
    Function to process a single case.
    Args:
        args (tuple): Contains caseid, case_data, dataset_path, output_folder, and progress queue.
    Returns:
        None
    """
    caseid, case_data, dataset_path, output_folder = args
    features = extract_features_from_raw_dataset(case_data)
    output_path = dataset_path / output_folder / f'cases_{caseid:04d}.parquet'
    features.to_parquet(output_path)


def main():
    dataset_name = 'signal_dataset_haythem'
    dataset_path = Path(f'data/datasets/{dataset_name}')
    output_folder = 'features'

    if not (dataset_path / output_folder).exists():
        (dataset_path / output_folder).mkdir(parents=True)

    data_raw = pd.read_parquet(dataset_path / 'cases/')
    print(f"Number of segments: {len(data_raw)}")

    # Restrict the data to 5 minutes (example filtering logic)
    data_raw = data_raw[[col for col in data_raw.columns if test_number_sup(col, 150)]]
    print(f"Number of segments after dropping NaN: {len(data_raw)}")

    # Prepare tasks for parallel processing
    tasks = [
        (caseid, case_data, dataset_path, output_folder)
        for caseid, case_data in data_raw.groupby('caseid')
    ]

    # Use process_map for parallel processing with progress bar
    process_map(process_case, tasks, max_workers=None)


if __name__ == '__main__':
    main()
