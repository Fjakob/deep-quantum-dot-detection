from config.imports import *

from src.lib.peakDetectors.os_cfar import OS_CFAR


def visualize():
    with open('reports/nelder_mead_history.pickle', 'rb') as file:
        opt_hist = pickle.load(file)

    plt.subplot(4,1,1)
    plt.plot(opt_hist[:,0],'-*')
    plt.ylabel('Cost')
    plt.ylim(0,25)
    plt.grid()
    plt.subplot(4,1,2)
    plt.plot(opt_hist[:,1],'-*')
    plt.ylabel('N')
    plt.grid()
    plt.subplot(4,1,3)
    plt.plot(opt_hist[:,2],'-*')
    plt.ylabel('T')
    plt.grid()
    plt.subplot(4,1,4)
    plt.plot(opt_hist[:,3],'-*')
    plt.ylabel('Nprot')
    plt.xlabel('Iteration')
    plt.grid()
    plt.show()



def train():

    ### Readout (x,p) dataset
    path = 'datasets/labeled/data_w30_labeled_with_peaks.pickle'
    with open(path, 'rb') as file:
        X, _, P = pickle.load(file)

    ### instanciate
    os_cfar = OS_CFAR(N=193, T=6.54, N_protect=25)

    ### run optimization and save results
    os_cfar.optimize_parameters((X,P))
    opt_hist = os_cfar.opt_hist
    opt_hist = np.asarray(opt_hist)
    with open('reports/nelder_mead_history.pickle', 'wb') as file:
        pickle.dump(opt_hist, file)

    ### sample plots
    idx=17
    peaks, n_peaks, thresh = os_cfar.detect(X[idx])
    p = P[idx]
    plt.plot(X[idx])
    plt.plot(thresh)
    plt.plot(peaks, X[idx][peaks], '*')
    plt.title(f'Peaks: {p}, detected: {n_peaks}')
    print(idx)
    plt.show()

if __name__ == "__main__":
    #train()
    visualize()