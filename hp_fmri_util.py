import numpy as np

#@title fMRI util

def delay_one(mat, d):
  """delays a matrix by a delay d. Positive d ==> row t has row t-d."""
  new_mat = np.zeros_like(mat)
  if d>0:
      new_mat[d:] = mat[:-d]
  elif d<0:
      new_mat[:d] = mat[-d:]
  else:
      new_mat = mat
  return new_mat

def delay_mat(mat, delays):
  """delays a matrix by a set of delays d.
    a row t in the returned matrix has the concatenated:
    row(t-delays[0],t-delays[1]...t-delays[last] ).
  """
  new_mat = np.concatenate([delay_one(mat, d) for d in delays],axis = -1)
  return new_mat

def prepare_nlp_features(train_features, test_features, word_train_indicator, TR_train_indicator, 
                         SKIP_WORDS=20, END_WORDS=5176, path='hp_fmri_data'):
        
    time = np.load(f'{path}/time_fmri.npy')
    runs = np.load(f'{path}/runs_fmri.npy') 
    time_words = np.load(f'{path}/time_words_fmri.npy')
    time_words = time_words[SKIP_WORDS:END_WORDS]
        
    words_id = np.zeros([len(time_words)])
    # w=find what TR each word belongs to
    for i in range(len(time_words)):
        words_id[i] = np.where(time_words[i]> time)[0][-1]
        
    all_features = np.zeros([time_words.shape[0], train_features.shape[1]])
    all_features[word_train_indicator] = train_features
    all_features[~word_train_indicator] = test_features
        
    p = all_features.shape[1]
    tmp = np.zeros([time.shape[0], p])
    for i in range(time.shape[0]):
        tmp[i] = np.mean(all_features[(words_id<=i)*(words_id>i-1)],0)
    tmp = delay_mat(tmp, np.arange(1,5))

    # remove the edges of each run
    tmp = np.vstack([zscore(tmp[runs==i][20:-15]) for i in range(1,5)])
    tmp = np.nan_to_num(tmp)
        
    return tmp[TR_train_indicator], tmp[~TR_train_indicator]

def get_fold_flags(n, n_folds):
    flags = np.zeros((n))
    num_items_in_each_fold = int(np.floor(n/n_folds))
    for i in range(0,n_folds -1):
        flags[i*num_items_in_each_fold:(i+1)*num_items_in_each_fold] = i
    flags[(n_folds-1)*num_items_in_each_fold:] = (n_folds-1)
    return flags

def tr_to_word_train_indicator(tr_train_indicator, skip_words=20, end_words=5176, path='hp_fmri_data'):
    time = np.load(f'{path}/time_fmri.npy')
    runs = np.load(f'{path}/runs_fmri.npy') 
    time_words = np.load(f'{path}/time_words_fmri.npy')
    time_words = time_words[skip_words:end_words]
        
    word_train_indicator = np.zeros([len(time_words)], dtype=bool)    
    words_id = np.zeros([len(time_words)],dtype=int)
    # Find what TR each word belongs to.
    for i in range(len(time_words)):                
        words_id[i] = np.where(time_words[i]> time)[0][-1]
        
        if words_id[i] <= len(runs) - 15:
            offset = runs[int(words_id[i])]*20 + (runs[int(words_id[i])]-1)*15
            if tr_train_indicator[int(words_id[i])-offset-1] == 1:
                word_train_indicator[i] = True
    return word_train_indicator 
