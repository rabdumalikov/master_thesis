import wandb
import utils


def main():
    dataset_types_choices = ['s(f)', 's(f+cf)', 's(f+a)', 's(f+cf+a)']

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="MasterThesis",
        id=str(11111),
        group='Experiment_1',
        name=f'DisentQA-T5L-[finetuning]-baseline'
    )

    for dtc in dataset_types_choices:
        results = utils.get_DisentQA_results(dtc)

        if results is None:
            print(f"No results for {dtc}")
            continue

        for test_data_type in results:
            exact_match_acc = results[test_data_type]

            for current_epoch in range(1, 21):
                wandb.log({'epoch': current_epoch,
                           f'{test_data_type}_EM_acc': exact_match_acc})

    wandb.finish()


if __name__ == "__main__":
    main()
