from config.imports import *

def main():
    
    with open('datasets/unlabeled/data_w30_unlabeled.pickle', 'rb') as data:
        X = pickle.load(data)


    with open('models/classifiers/intuitive_feature_based_rater.pickle', 'rb') as model:
        rater = pickle.load(model)

    for x in X:
        y = rater.rate(x)
        plt.plot(x)
        plt.title('Predicted label: {:.2f}'.format(y[0]))
        plt.show()


if __name__ == "__main__":
    main()