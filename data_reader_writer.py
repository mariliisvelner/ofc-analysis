import pandas as pd
import glob
import os
from scipy.io import loadmat
import pickle

PROJECT_DIR = "C:\\Users\\MariLiis\\Documents\\Ylikool\\Magister\\Thesis\\ofc-analysis"
IMAGE_DIR = "C:\\Users\\MariLiis\\Documents\\Ylikool\\Magister\\Thesis\\ofc-analysis\\images"
VAR_DIR = "C:\\Users\\MariLiis\\Documents\\Ylikool\\Magister\\Thesis\\ofc-analysis\\vars"
DATA_BEHAV_DIR = "C:\\Users\\MariLiis\\Documents\\Ylikool\\Magister\\Thesis\\ofc-analysis\\data\\data_behav"
DATA_EPHYS_DIR = "C:\\Users\\MariLiis\\Documents\\Ylikool\\Magister\\Thesis\\ofc-analysis\\data\\data_ephys"
DATA_REGR_DIR = "C:\\Users\\MariLiis\\Documents\\Ylikool\\Magister\\Thesis\\ofc-analysis\\ofc_behav_data"

EPHYS_ATTRIBUTES = ["buttonpress_events_hg", "buttonpress_window_events_hg", "game_events_hg", "game_window_events_hg"]
REGR_ATTRIBUTES = ["exputil", "gamble_ind", "loss_ind", "regret", "risk", "rpe", "win_ind", "winprob",
                   "previous_exputil",
                   "previous_gamble_ind", "previous_loss_ind", "previous_regret", "previous_risk", "previous_rpe",
                   "previous_win_ind", "previous_winprob"]
CURRENT_TRIAL_REGRESSORS = ["exputil", "gamble_ind", "loss_ind", "regret", "risk", "rpe", "win_ind", "winprob"]
PREVIOUS_TRIAL_REGRESSORS = ["previous_exputil", "previous_gamble_ind", "previous_loss_ind", "previous_regret",
                             "previous_risk", "previous_rpe", "previous_win_ind", "previous_winprob"]

SUBJECTS = ["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]
N_ELEC = {"s1": 5, "s2": 6, "s3": 59, "s4": 5, "s5": 61, "s6": 7, "s7": 11, "s8": 10, "s9": 19, "s10": 16}
N_TRIALS = {"s1": 180, "s2": 188, "s3": 194, "s4": 108, "s5": 179, "s6": 187, "s7": 181, "s8": 200, "s9": 200,
            "s10": 136}

AFTER_OUTCOME = "1.5s after outcome reveal"
GAMEPRES_BUTTONPRESS = "between game presentation and buttonpress"
time_window_types = [AFTER_OUTCOME, GAMEPRES_BUTTONPRESS]

GAMBLE = "gamble"
LOSS = "loss"
WIN = "win"
ALL = "all"
trial_types = [GAMBLE, LOSS, ALL, WIN]

EPHYS_FILENAME = "ephys_data"
BEHAVIOR_FILENAME = "behavior_data"
REGRESSOR_FILENAME = "regressor_data"
GAMBLE_CHOICES_FILENAME = "gamble_choices"


def get_ephys_data():
    ephys_data = {}

    os.chdir(DATA_EPHYS_DIR)
    s_nr = 1
    for file in glob.glob("*.mat"):
        data = loadmat(file)
        ephys_data["s" + str(s_nr)] = data
        s_nr += 1

    return ephys_data


def get_gamble_choices_data():
    os.chdir(DATA_BEHAV_DIR)
    return pd.read_csv("gamble_choices.csv")


def get_bad_trials_data():
    os.chdir(DATA_BEHAV_DIR)
    return loadmat("bad_trials_OFC.mat")["bad_trials_OFC"]


def get_behavior_data():
    behav_data = {}

    os.chdir(DATA_BEHAV_DIR)
    s_nr = 1
    for file in glob.glob("*.csv"):
        if file != "gamble_choices.csv":
            df = pd.read_csv(file)
            behav_data["s" + str(s_nr)] = df
            s_nr += 1

    bad_trials_data = get_bad_trials_data()

    for i in range(10):
        s = "s" + str(i + 1)
        # Get only the indices from bad_trials_data which denote excluded trials
        idx_1 = [j for j in range(200) if bad_trials_data[i][j] == 1]
        behav = behav_data[s]
        # Get the indices of the timeout trials
        idx_2 = list(behav[behav.outcome == "Timeout"].index)
        # Combine the indices
        idx_all = sorted(list(set(idx_1 + idx_2)))
        # Make a new feature "trial.included" which is 1 if the trial is included in the analysis and 0 otherwise
        behav_data[s]["trial.included"] = 1
        behav_data[s]["trial.included"].iloc[idx_all] = 0

    return behav_data


def get_regressor_data():
    regressor_data = {}

    os.chdir(DATA_REGR_DIR)
    s_nr = 1
    for file in glob.glob("*.mat"):
        data = loadmat(file)
        regressor_data["s" + str(s_nr)] = data
        s_nr += 1

    return regressor_data


def data_preprocess():
    ephys_data = get_ephys_data()
    gamble_choices = get_gamble_choices_data()
    behavior_data = get_behavior_data()
    regressor_data = get_regressor_data()

    save_var(ephys_data, EPHYS_FILENAME)
    save_var(gamble_choices, GAMBLE_CHOICES_FILENAME)
    save_var(behavior_data, BEHAVIOR_FILENAME)
    save_var(regressor_data, REGRESSOR_FILENAME)


# Save a variable to <filename>.p file
def save_var(var, filename):
    os.chdir(VAR_DIR)
    pickle.dump(var, open(filename + ".p", "wb"))


# Load and return a variable from a <filename>.p file
def load_var(filename):
    os.chdir(VAR_DIR)
    return pickle.load(open(filename + ".p", "rb"))


def save_fig(fig, figname):
    os.chdir(IMAGE_DIR)
    fig.savefig(figname + ".png", bbox_inches='tight')


def load_ephys_data():
    return load_var(EPHYS_FILENAME)


def load_behavior_data():
    return load_var(BEHAVIOR_FILENAME)


def load_regressor_data():
    return load_var(REGRESSOR_FILENAME)


def load_gamble_choices_data():
    return load_var(GAMBLE_CHOICES_FILENAME)
