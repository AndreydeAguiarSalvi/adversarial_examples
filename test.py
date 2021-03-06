# using https://github.com/Harry24k/adversarial-attacks-pytorch
from utils.utils import *
from utils.datasets import get_Dataset


def test(model, args, loader):
    epsilons = [args['eps']] if args['eps'] else [0, .05, .1, .15, .2, .25, .3]
    examples = []
    accuracies = []

    for eps in epsilons:
        args['eps'] = eps
        result = attack(model, args, loader)
        accuracies.append(result['total_acc'])
        examples.append(result['adv_examples'])

        f = open(args['folder'] + f"results_attack-{args['attack']}_eps-{eps}.txt", 'w')
        for (key, value) in result.items():
            if 'examples' not in key: f.write(f"{key}: {value}\n")
        f.close()

    save_accuracies(epsilons, accuracies, args)
    save_examples(epsilons, examples, args)


if __name__ == "__main__":
    args = create_argparser()
    if args['device'] != 'cpu': args['device'] = 'cuda:' + args['device']
    get_folder(args)
    _, _, test_loader = get_Dataset(args, False, True)

    model = create_model(args).to(args['device'])
    chkpt = torch.load(args['weights'], map_location=args['device'])
    model.load_state_dict(chkpt)

    
    test(model, args, test_loader)