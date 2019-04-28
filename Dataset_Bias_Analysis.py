import pandas as pd
from prettytable import PrettyTable


# An overview of the dataset considering the columns defined in groups as revealing of identities
# and also looking the text to detect the identities.


file_path ="train.csv"

# Filter the dataset based on some identity. An identity is identified if its grade is greater than 0.5.
def filter_frame_v2(frame, keyword=None, length=None):

    if keyword:
        frame = frame[frame[keyword] > 0.5]
    if length:
        frame = frame[frame['length'] <= length]
    return frame

# Compute the rate of some identity is identified as toxic amongst the comments.
def class_balance_v2(frame, keyword, length=None):
    frame = filter_frame_v2(frame, keyword, length)
    return len(frame.query('toxic')) / len(frame)


wiki_data = pd.read_csv(file_path, sep = ',')
#create a new column to determine toxicity
wiki_data['toxic'] = wiki_data['target'] > 0.5
wiki_data['length'] = wiki_data['comment_text'].str.len()

#these are the groups in the dataset that have more than 500 examples in test
groups = ['black','christian','female',
          'homosexual_gay_or_lesbian','jewish','male','muslim',
          'psychiatric_or_mental_illness','white']

wiki_data = wiki_data.loc[:,['comment_text','toxic']+groups]

print('overall fraction of comments labeled toxic:', class_balance_v2(wiki_data, keyword=None))
print('overall class balance {:.1f}%\t{} examples'.format(
    100 * class_balance_v2(wiki_data, keyword=None), len(wiki_data)))
print('\n')
for term in groups:
  fraction =  class_balance_v2(wiki_data, term)
  frame = filter_frame_v2(wiki_data, term)
  num = len(frame)
  print('Proportion of identity {:s} in train: {:.3f}%'.format(term,100*(len(frame)/len(wiki_data.index))))
  print('Proportion of toxic examples for {:10s} {:.1f}%\t{} examples'.format(str(term), 100 * fraction, num))
  print('\n')

# This is other way to compute it
# for each group, print the number of cases that are toxic and not toxic
# for x in groups:
#   aux = wiki_data[ wiki_data[x] > 0.5 ] # for each group, select only elements that belong to that group
#   wiki_grouped = aux.loc[:,['toxic',x]].groupby('toxic', as_index=False).count()
#   print(wiki_grouped.to_string()+'\n')


# The same as above but instead of using the columns to determine identity search the text comment for occurring
# the identity term.

def filter_frame(frame, keyword=None, length=None):
    """Filters DataFrame to comments that contain the keyword as a substring and fit within length."""
    if keyword:
        if isinstance(keyword,list):
          frame = frame[frame['comment_text'].str.contains('|'.join(keyword), case=False)]
        else:
          frame = frame[frame['comment_text'].str.contains(keyword, case=False)]
    if length:
        frame = frame[frame['length'] <= length]
    return frame

def class_balance(frame, keyword, length=None):
    """Returns the fraction of the dataset labeled toxic."""
    frame = filter_frame(frame, keyword, length)
    return len(frame.query('toxic')) / len(frame)

# print('\n')
# print('overall fraction of comments labeled toxic:', class_balance(wiki_data, keyword=None))
# print('overall class balance {:.1f}%\t{} examples'.format(
#     100 * class_balance(wiki_data, keyword=None), len(wiki_data)))
#
# for term in groups:
#   if term == 'homosexual_gay_or_lesbian':
#       term = ['homosexual','gay','lesbian']
#   elif term == 'psychiatric_or_mental_illness':
#       continue
#   fraction =  class_balance(wiki_data, term)
#   num = len(filter_frame(wiki_data, term))
#   print('class balance for {:10s} {:.1f}%\t{} examples'.format(str(term), 100 * fraction, num))

