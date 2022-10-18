from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn.model_selection import KFold

######################################################################
# DO NOT EXECUTE
######################################################################

if __name__ == "__main__":

    kernel_list = {"Linear": kernels.DotProduct() + kernels.WhiteKernel(), 
                   "Gaussian" :kernels.RBF(),
                   "Matérn": kernels.Matern(nu=1/2),
                   "Rational quadratic": kernels.RationalQuadratic()}

    for name, kernel in kernel_list.items():
        k_fold = KFold(n_splits=5)
        R2 = []
        idx = 1
        for train_index, test_index in k_fold.split(V):
            V_train, V_test = V[train_index], V[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            gpr = GaussianProcessRegressor(kernel=kernel).fit(V_train, Y_train)
            fit = gpr.score(V_test, Y_test)
            print("R2 score on fold {}: {:.2f}".format(idx, fit))
            R2.append(fit)
            idx += 1
        print("Mean of R2 scores for " + name +  " kernel: {:.2f} \n".format(np.mean(R2)))

    for idx in range(10):
        x_test, y_test = X[idx], Y[idx]
        plt.plot(x_test)
        v = np.expand_dims((feature_extracter.extract_from_spectrum(x_test) - mean) / scale, axis=0)
        y_bar, var = gpr.predict(v, return_std=True)
        plt.title("Real label: {:.2f} \nPredicted: {:.2f} (with variance: {:.2f})".format(y_test, y_bar[0], var[0]))
        plt.show()