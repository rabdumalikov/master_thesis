import os
import shutil

def find_best_checkpoint(id: int):
    dir_path = f'.builds/{id}/models'

    all_checkpoint_names = []
    checkpoint_name = ''
    best_em = -1
    # iterate over all the directories in the specified directory
    for dir_name in os.listdir(dir_path):

        if os.path.isdir(os.path.join(dir_path, dir_name)):

            with open(dir_path + '/' + dir_name  + '/' + 'results.txt', 'r') as f:
                d = {}
                for l in f.readlines():
                    key, value = l.split('=')
                    d[key] = value
                    
                all_checkpoint_names.append(dir_path + '/' + dir_name)

                if float(d['em']) > best_em:
                    best_em = float(d['em'])
                    checkpoint_name = dir_name

                    print(f'Best checkpoint {checkpoint_name} with em={best_em}')


    best_checkpoint = dir_path + '/' + checkpoint_name
    if best_checkpoint in all_checkpoint_names:
        all_checkpoint_names.remove(best_checkpoint)
    return best_checkpoint, all_checkpoint_names

# DANGEROUS SCRIPT
def main():

    path = '.builds/'

    counter = 10
    for dr in sorted(os.listdir(path)):        
        if not os.path.isdir(path + dr):
            continue
        best_checkpoint, all_checkpoints = find_best_checkpoint(int(dr))
        print(dr, best_checkpoint, all_checkpoints)

        for to_remove in all_checkpoints:
            print(f'Remove {to_remove}')
            shutil.rmtree(to_remove)
        
        if len(all_checkpoints) > 0:
            if counter <= 0:
                break

            counter -= 1


if __name__ == '__main__':
    main()


