import os

def get_unique_filepath(path, pattern):
    
    all_files = os.listdir(path)

    assert isinstance(pattern, str)
    occurrences = [el for el in all_files if pattern in el]
    
    if occurrences:
        # in case we are processing .mat files
        if '.mat' in occurrences[0]:
            if len(occurrences) > 2:
                raise ValueError(pattern + ' not unique!')
            else: 
                if 'LF' in occurrences[0]:
                    return os.path.join(path, occurrences[0]), os.path.join(path, occurrences[1])
                elif 'HF' in occurrences[0]:
                    return os.path.join(path, occurrences[1]), os.path.join(path, occurrences[0])
        # in case of other files (.nii.gz)
        else:
            if len(occurrences) > 1:
                raise ValueError(pattern + ' not unique!')
            else:
                return os.path.join(path, occurrences[0])
    else:
        raise Exception('No file found for \'{}\' in {}'.format(pattern, path))
