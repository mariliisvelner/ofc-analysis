import numpy as np

elec_nrs_1 = {'s1': [3, 5], 's2': [4, 6], 's3': [2, 6, 19, 22, 24, 26, 29, 31, 35, 38, 41, 43, 44], 's4': [2, 5],
            's5': [12, 20, 27, 32, 36, 38, 53], 's6': [5, 6, 7], 's7': [1, 4, 5, 6, 10], 's8': [3, 6, 7, 9],
            's9': [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, 15, 18], 's10': [3, 7, 9, 14, 16]}


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
def time_lock_outcome(ephys_data, behav_data, i1=500, i2=1500):
    result = []
    experiment = [item for sublist in ephys_data for item in
                  sublist]  # the measurements of the entire experiment in one list
    # should have n_trials x 3001 elements

    for index, row in behav_data.iterrows():
        delay = round((row['reveal.time'] - row['buttonpress.time'])
                      * 1000)  # the time between buttonpress event and outcome reveal in ms
        outcome_moment = index * 3001 + 1000 + delay  # the buttonpress is 1s = 1000ms from the beginning of the trial
        if outcome_moment + i2 >= len(experiment):
            # We don't include the last trial because the indices will probably be out of range
            break
        elif outcome_moment - i1 < 0:
            # May happen in the case of the first trial, we'll just skip it
            continue
        else:
            trial = experiment[outcome_moment - i1:outcome_moment + i2]
        result.append(np.array(trial))

    return result

# Time-locks the given ephys trials to [-i1;i2]ms from the buttonpress
# NB! The ephys data has to be game_events_hg!
def time_lock_buttonpress(ephys_data, behav_data, i1=2000, i2=500):
    result = []
    experiment = [item for sublist in ephys_data for item in
                  sublist]  # the measurements of the entire experiment in one list
    # should have n_trials x 3001 elements

    for index, row in behav_data.iterrows():
        delay = round((row['buttonpress.time'] - row['choice.time'])
                      * 1000)  # the time between buttonpress event and game presentation in ms
        buttonpress_moment = index * 3001 + 1000 + delay  # the game presentation is 1s = 1000ms from the beginning of the trial
        if buttonpress_moment + i2 >= len(experiment):
            # We don't include the last trial because the indices will probably be out of range
            break
        elif buttonpress_moment - i1 < 0:
            # May happen in the case of the first trial, we'll just skip it
            continue
        else:
            trial = experiment[buttonpress_moment - i1:buttonpress_moment + i2]
        result.append(np.array(trial))

    return result
