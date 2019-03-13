import numpy as np

# Extracts the data for the nth electrode (n = 1, 2, ...)
# The resulting array should be in the shape (n_trials, 3001)
def extract_electrode_data(data, n_elec):
    new_data = []
    for trial in data:
        trial_data = []
        for timepoint in trial:
            trial_data.append(timepoint[n_elec - 1])
        new_data.append(trial_data)
    return np.array(new_data)


# Convert trials so that they are [i1, i2] indices around each outcome reveal
# ephys data should be shaped (n_trials, 3001)
# NB! ephys_data is buttonpress i.e. [-1, 2]s around buttonpress event!
def time_lock_outcome(ephys_data, behav_data, is_win, i1, i2):
    result = []
    experiment = [item for sublist in ephys_data for item in
                  sublist]  # the measurements of the entire experiment in one list
    # should have n_trials x 3001 elements

    for index, row in behav_data.iterrows():
        if not is_win and row['outcome'] not in ['Loss']: #, 'WouldHaveWon']:#, 'WouldHaveLost']:
            continue
        elif is_win and row['outcome'] != 'Win':
            continue

        delay = round((row['reveal.time'] - row[
            'buttonpress.time']) * 1000)  # the time between buttonpress event and outcome reveal in ms
        outcome_moment = index * 3001 + 1000 + delay  # the buttonpress is 1s = 1000ms from the beginning of the trial
        if outcome_moment + i2 >= len(experiment):
            break
        trial = experiment[outcome_moment - i1:outcome_moment + i2]
        result.append(trial)

    return result

