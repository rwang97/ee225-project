import glob
import matplotlib.pyplot as plt
import json

if __name__ == '__main__':

    f = open("outputs/exp_/history.json")
    data = json.load(f)

    # print(data)
    
    trains = []
    vals = []
    bests = []
    for stats in data:
        train = stats['train']
        validation = stats['valid']
        best = stats['best']
        trains.append(train)
        vals.append(validation)
        bests.append(best)

    X = list(range(len(trains)))

    plt.plot(X, trains,  color='r', ls='-')
    # plt.plot(X, vals,  color='g', ls='-')
    # plt.plot(X, bests, color='b', ls='-')
    # plt.plot(X, Y1, color='r', ls='-', label='random-shooting')
    # plt.plot(X, Y2, color='g', ls='-', label='CEM iterations2')
    # plt.plot(X, Y3, color='b', ls='-', label='CEM iterations4')
    plt.xlabel('Epochs')
    plt.ylabel('Train Loss') 
    # plt.legend()
    plt.grid()
    plt.title('Training Loss on 10 Samples')
    # plt.show()
    plt.savefig("train_loss.pdf", dpi=300, transparent=False, bbox_inches='tight')